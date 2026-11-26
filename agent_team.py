"""내 주제로 Multi-Agent 팀을 만드는 틀.

전문가 목록(이름, 역할, 도구)만 적으면 아래 세 가지가 따라온다.
  1) 지휘자: 질문마다 필요한 전문가를 고르고, 동시에 일을 시키고, 답을 하나로 합친다   → team.ask()
  2) 증류: "전문가 고르기"를 큰 모델(스승)에게 시켜 자료를 만들고 작은 모델(제자)을 학습시킨다
           → team.build_distill_data(), team.train_student(), team.load_student()
  3) 서비스: team_service.py가 이 팀을 웹 화면이나 MCP 서버로 띄운다

완성 예시는 xray_team.py, 설명은 example/2-3_my_multi_agent.ipynb 에 있다.
교육·연구용이며 진료나 임상 판단을 대신하지 않는다.
"""
import hashlib
import itertools
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")      # 모델을 불러올 때 진행 막대를 숨긴다

import ollama
from strands import Agent, tool
from strands.hooks import BeforeToolCallEvent

from lab_ui import (USAGE, Status, chat_counted, count_tokens, md, run_together, settle_tokens, stream_answer, tool_log,
                    usage_summary, wait_for, watch_tokens)
from strands.models.ollama import OllamaModel

ROOT = Path(__file__).resolve().parent
EXPERT_MODEL = os.getenv("EXPERT_MODEL", "gemma4:e4b")           # 전문가, 지휘자, 스승 역할
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
STUDENT_MODEL = os.getenv("STUDENT_MODEL", "mlx-community/gemma-3-1b-it-4bit")   # 증류할 작은 모델

RULES = """
- 반드시 도구를 써서 근거를 찾는다. 도구 결과에 없는 숫자와 사실은 쓰지 않는다.
- 한국어로, 소제목 3개 이내, 전체 10줄 이내로 답한다.
- 마지막 줄에 확신 정도(높음, 보통, 낮음)를 적는다."""
REPORT = "(1) 한 줄 결론 (2) 근거 3개 이내 (3) 주의할 점 (4) 다음에 할 일"
DISCLAIMER = "※ 교육·연구용 실습 결과이며 진료나 임상 판단을 대신하지 않습니다."


JOSA = {"가": "이", "와": "과", "를": "을", "는": "은", "로": "으로"}      # 받침이 없을 때: 받침이 있을 때


def fill(template, word):
    """질문 틀의 {d} 자리에 낱말을 넣는다. 바로 뒤의 조사(가, 와, 를, 는, 로)는 받침에 맞게 고친다."""
    def josa(m):
        last = word[-1] if word else ""
        if not "가" <= last <= "힣":
            return word + m.group(1)
        final = (ord(last) - 0xAC00) % 28                # 0이면 받침 없음, 8이면 ㄹ 받침
        plain = final == 0 or (m.group(1) == "로" and final == 8)
        return word + (m.group(1) if plain else JOSA[m.group(1)])
    return re.sub(r"\{d\}(가|와|를|는|로)(?=[\s,?.]|$)", josa, template).replace("{d}", word)


# 실습에 쓰는 두 모델의 정식 이름. 둘 다 Google이 가중치를 공개한 Gemma 계열이다.
STUDENT_BASE = "Google Gemma 3 1B"      # 제자의 바탕 모델. 이 위에 LoRA Adapter만 학습해 얹는다 (바탕 모델은 고치지 않는다)
TEACHER_BASE = "Google Gemma 4 E4B"     # 스승

# 견줄 기준이 되는 모델들 (2026년 11월 기준). 위에서 아래로 갈수록 큰 모델이다.
#   index : Artificial Analysis의 종합 성능 지수(Intelligence Index). 여러 시험 점수를 합친 값으로, 높을수록 어려운 일을 잘한다.
#           같은 표에서 확인한 값만 적었고, 찾지 못한 모델은 None이다.
#   price : 남의 서버에서 빌려 쓸 때 100만 token당 달러 (입력, 출력). 직접 내려받아 돌리는 모델은 None
OPEN_MODELS = [
    {"name": "gpt-oss-20b", "by": "OpenAI가 공개한 모델 (내려받아 직접 돌림)", "params": "약 210억 개", "file_gb": 14, "index": None, "price": None,
     "memory": "약 16GB", "where": "메모리 16GB 이상인 컴퓨터"},
    {"name": "gpt-oss-120b", "by": "OpenAI가 공개한 모델 (내려받아 직접 돌림)", "params": "약 1,170억 개", "file_gb": 65, "index": 4, "price": None,
     "memory": "약 80GB", "where": "80GB짜리 GPU 한 장 (데이터센터용)"},
    {"name": "GPT-6 Luna", "by": "OpenAI의 최신 모델 가운데 가장 싼 것", "params": "공개하지 않음", "file_gb": None, "index": 28, "price": (0.10, 0.50),
     "memory": "공개하지 않음", "where": "OpenAI의 서버. 쓴 만큼 요금"},
    {"name": "GPT-6.1 Terra", "by": "OpenAI의 최신 모델 가운데 싼 축", "params": "공개하지 않음", "file_gb": None, "index": 51, "price": (1, 5),
     "memory": "공개하지 않음", "where": "OpenAI의 서버. 쓴 만큼 요금"},
]


def free_gpu():
    """Ollama가 메모리에 올려 둔 모델을 내린다. 학습과 큰 모델이 GPU를 함께 쓰면 학습이 멈출 수 있다."""
    try:
        for m in ollama.ps().models:
            ollama.generate(model=m.model, prompt="", keep_alive=0)
    except Exception:
        pass


def gpu_was_busy(log):
    """학습이 'GPU가 바빠서' 멈춘 것인지 본다 (macOS가 화면이 버벅이지 않게 작업을 끊는 경우)."""
    return any("Impacting Interactivity" in line or "Command buffer execution failed" in line for line in log)


def follow_training(proc, iters):
    """학습 프로그램(mlx_lm lora)이 찍는 줄을 읽어 진행 상황을 보여 준다. 읽은 줄 전체를 돌려준다."""
    status, log = Status(f"⏳ 학습을 시작합니다. {iters}번 반복하고 5분쯤 걸립니다. 모델을 불러오는 중"), []
    for line in proc.stdout:
        line = re.sub(r"\x1b\[[0-9;]*m", "", line).rstrip()
        log.append(line)
        m = re.match(r"\s*(\d+)\s+(val\s+)?([\d.]+)", line)
        if "Trainable" in line:
            print("   " + line.strip())
        elif m and m[2]:        # 중간 시험: 학습에 쓰지 않은 문제로 틀린 정도를 잰다
            print(f"   {int(m[1]):>4}/{iters}번  중간 시험에서 틀린 정도(val loss) {float(m[3]):.3f}  ({status.seconds():.0f}초)")
        elif m:
            status.update(f"⏳ 학습 중: {int(m[1])}/{iters}번  틀린 정도(loss) {float(m[3]):.3f}  ({status.seconds():.0f}초)")
    status.done(f"✅ {iters}번 반복을 마쳤습니다  ({status.seconds():.0f}초)")
    return log


class Team:
    """전문가 여러 명과 지휘자 한 명으로 된 팀.

    name    : 영문 이름. 증류 자료를 distill/<name>/ 폴더에 저장할 때 쓴다
    goal    : 이 팀이 하는 일을 한 줄로 (예: "흉부 엑스레이 판독을 돕는 팀")
    experts : {"영문키": {"title": 직함, "role": 맡는 일(지휘자가 이걸 보고 고른다),
                          "prompt": 전문가에게 주는 지시문, "tools": [파이썬 함수, ...]}}
    report  : 통합 답변의 형식
    prepare : (선택) 검토 대상(case)을 받아 모든 전문가에게 줄 기초 자료를 글로 돌려주는 함수
    cases   : (선택) 화면에서 고를 수 있게 보여 줄 검토 대상 예시
    """

    def __init__(self, name, goal, experts, report=REPORT, rules=RULES, prepare=None, cases=(), disclaimer=DISCLAIMER):
        self.name, self.goal, self.report, self.rules = name, goal, report, rules
        self.prepare, self.cases, self.disclaimer = prepare, list(cases), disclaimer
        self.experts = {k: {**v, "tools": [t if hasattr(t, "tool_spec") else tool(t) for t in v["tools"]]}
                        for k, v in experts.items()}
        self.keys = list(self.experts)
        self.careful = True
        self.folder = ROOT / "distill" / name
        self._student = {}

    # ---------- 1) 지휘자: 고르기 → 동시에 시키기 → 합치기 ----------
    @property
    def router_prompt(self):
        lines = "\n".join(f"- {k}: {v['role']}" for k, v in self.experts.items())
        return (f"너는 {self.goal}의 지휘자다. 질문에 답하는 데 꼭 필요한 전문가만 고른다.\n{lines}\n"
                f"질문 하나에 1명에서 {len(self.keys)}명까지 고를 수 있다.")

    def _sorted(self, picked):
        return sorted(set(picked), key=self.keys.index)

    def route_with_teacher(self, question, model=None):
        """큰 모델이 필요한 전문가를 고른다. 답은 정해진 형식(JSON)으로만 받는다.

        model에 다른 Ollama 모델 이름을 주면 그 모델에게 같은 지시문으로 시킨다 (학습 없이 지시문만으로 시키는 방식).
        """
        schema = {"type": "object", "required": ["experts"], "properties": {"experts": {
            "type": "array", "minItems": 1, "maxItems": len(self.keys), "items": {"type": "string", "enum": self.keys}}}}
        kw = dict(model=model or EXPERT_MODEL, format=schema, options={"temperature": 0},
                  messages=[{"role": "system", "content": self.router_prompt}, {"role": "user", "content": question}])
        try:
            r = ollama.chat(think=False, **kw)     # 생각 과정을 끄면 빨라진다
        except Exception:
            r = ollama.chat(**kw)
        count_tokens("지휘자 (전문가 고르기)", r)
        return self._sorted(json.loads(r["message"]["content"])["experts"])

    @staticmethod
    def _facts(agent):
        """Agent가 도구를 불러 돌려받은 결과를 모은다: [{"tool", "input", "output"}]. 전문가의 말이 아니라 도구가 준 값이다."""
        calls, facts = {}, []
        for m in agent.messages:
            for b in m.get("content", []):
                if "toolUse" in b:
                    calls[b["toolUse"].get("toolUseId")] = b["toolUse"]
                if "toolResult" in b:
                    use = calls.get(b["toolResult"].get("toolUseId"), {})
                    out = " ".join(str(c.get("text", c.get("json", ""))) for c in b["toolResult"].get("content", []))
                    facts.append({"tool": use.get("name", "?"), "input": json.dumps(use.get("input", {}), ensure_ascii=False)[:120], "output": out[:400]})
        return facts

    @staticmethod
    def _fact_lines(rows):
        """여러 전문가의 도구 결과를 지시문에 넣을 글로 만든다."""
        lines = [f"- {r['title']}의 도구 {f['tool']}({f['input']}) → {f['output']}" for r in rows for f in r.get("facts", [])]
        return "\n".join(lines) or "(도구 결과 없음)"

    # 대화할 때 지킬 것 (careful=True일 때 지시문에 넣는다). 도구로 확인한 값과 짐작을 같은 무게로 섞지 않게 한다
    TALK_RULE = ("지켜야 할 것: 다른 전문가가 도구로 얻은 결과(계산, 기준표 조회, 검색)는 받아들인다. 네 기억이나 짐작으로 그것을 뒤집지 않는다. "
                 "네 도구로 확인할 수 없는 항목에는 판정을 내리지 말고 '내 도구로는 확인할 수 없다'고 말한다.\n")

    def ask_expert(self, key, question, context="", quiet=True, on_event=None):
        """전문가 한 명에게 묻는다. 매번 새 Agent를 만들어 다른 질문의 기억이 섞이지 않게 한다."""
        spec = self.experts[key]
        agent = Agent(model=OllamaModel(host=OLLAMA_HOST, model_id=EXPERT_MODEL, temperature=0.2),
                      system_prompt=spec["prompt"] + self.rules, tools=spec["tools"], callback_handler=None)
        watch_tokens(agent, spec["title"])      # 쓰는 token을 실시간으로 센다
        if not quiet:
            tool_log(agent, spec["title"])      # 도구를 부를 때마다 화면에 한 줄 알려 준다
        if on_event:                            # 웹 화면에도 같은 소식을 보낸다
            agent.hooks.add_callback(BeforeToolCallEvent, lambda e: on_event(
                {"type": "tool", "expert": key, "title": spec["title"], "name": e.tool_use.get("name"),
                 "args": ", ".join(f"{k}={str(v)[:50]!r}" for k, v in (e.tool_use.get("input") or {}).items())}))
        used = lambda: [b["toolUse"]["name"] for m in agent.messages for b in m.get("content", []) if "toolUse" in b]
        start = time.time()
        try:
            out = agent((f"[기초 자료]\n{context}\n\n" if context else "") + f"[질문]\n{question}")
            if not used():      # 도구 없이 기억만으로 답했으면 한 번 더 시킨다
                out = agent("도구를 한 번도 쓰지 않았다. 먼저 도구를 불러 근거를 찾고, 그 결과만으로 다시 답하라.")
            text = str(out).strip()
            settle_tokens(agent)
        except Exception as e:
            text = f"(분석 실패: {e})"
        return {"expert": key, "title": spec["title"], "answer": text, "tools": used(), "seconds": round(time.time() - start, 1),
                "facts": self._facts(agent)}

    def comment(self, key, question, results):
        """전문가 한 명이 다른 전문가들의 답을 읽고 의견을 말한다. 도구는 쓰지 않고 짧게 말한다."""
        mine = next(r for r in results if r["expert"] == key)
        others = "\n\n".join(f"[{r['title']}의 답]\n{r['answer'][:900]}" for r in results if r["expert"] != key)
        prompt = (f"너는 {self.goal}의 {mine['title']}다. 같은 질문에 다른 전문가들이 아래와 같이 답했다.\n"
                  "네가 찾은 근거에 비추어, 다른 전문가 한 명 한 명에게 이름을 부르며 말을 건다.\n"
                  "각 전문가에게 (1) 네 근거와 맞는 점 또는 어긋나는 점 하나, (2) 그 전문가가 더 확인해 주면 좋을 것 하나를 말한다.\n"
                  "한국어로, 전문가 한 명당 두 문장 이내. 네 답과 다른 전문가의 답에 없는 사실은 새로 지어내지 않는다.\n"
                  + (self.TALK_RULE + f"\n[전문가들이 도구로 얻은 결과]\n{self._fact_lines(results)}\n" if self.careful else "")
                  + f"\n[질문]\n{question}\n\n[네가 한 답]\n{mine['answer'][:900]}\n\n{others}")
        kw = dict(model=EXPERT_MODEL, messages=[{"role": "user", "content": prompt}], options={"temperature": 0.2})
        start = time.time()
        try:
            try:
                text = chat_counted(mine["title"], think=False, **kw)
            except Exception:
                text = chat_counted(mine["title"], **kw)
        except Exception as e:
            text = f"(의견을 받지 못함: {e})"
        return {"expert": key, "title": mine["title"], "to": [r["title"] for r in results if r["expert"] != key],
                "text": text, "seconds": round(time.time() - start, 1), "round": 1, "tools": []}

    def reply(self, key, question, context, results, talks, quiet=True, on_event=None):
        """전문가 한 명이 다른 전문가들에게 받은 질문과 지적에 답한다. 필요하면 자기 도구를 다시 써서 확인한다."""
        spec, mine = self.experts[key], next(r for r in results if r["expert"] == key)
        said = "\n\n".join(f"[{t['title']}가 한 말]\n{t['text']}" for t in talks if t["expert"] != key)
        agent = Agent(model=OllamaModel(host=OLLAMA_HOST, model_id=EXPERT_MODEL, temperature=0.2),
                      system_prompt=spec["prompt"] + self.rules, tools=spec["tools"], callback_handler=None)
        watch_tokens(agent, spec["title"])      # 쓰는 token을 실시간으로 센다
        if not quiet:
            tool_log(agent, spec["title"])
        if on_event:
            agent.hooks.add_callback(BeforeToolCallEvent, lambda e: on_event(
                {"type": "tool", "expert": key, "title": spec["title"], "name": e.tool_use.get("name"),
                 "args": ", ".join(f"{k}={str(v)[:50]!r}" for k, v in (e.tool_use.get("input") or {}).items())}))
        used = lambda: [b["toolUse"]["name"] for m in agent.messages for b in m.get("content", []) if "toolUse" in b]
        start = time.time()
        try:
            out = agent((f"[기초 자료]\n{context}\n\n" if context else "") + f"[질문]\n{question}\n\n[네가 처음 한 답]\n{mine['answer'][:900]}\n\n"
                             f"아래는 지금까지 다른 전문가들이 한 말이다. 이 가운데 너({spec['title']})에게 한 질문과 지적, "
                             f"그리고 '남은 것'으로 적힌 쟁점 가운데 네 도구로 확인할 수 있는 것에 답하라. 이미 답한 것은 되풀이하지 않는다.\n{said[-3500:]}\n\n"
                             + ("답하는 법: 의견이 갈린 항목이나 너에게 확인을 요청한 항목은 네 도구로 다시 확인한 뒤 답한다. "
                                "네 도구로 확인할 수 없는 것은 판정하지 말고 '내 도구로는 확인할 수 없다'고 말한다. "
                                "다른 전문가가 도구로 얻은 결과는 받아들이고, 네 기억이나 짐작으로 뒤집지 않는다. " if self.careful else
                                "답하는 법: 확인이 필요하면 네 도구를 한 번만 더 써서 확인한다(꼭 필요할 때만). ")
                             + "질문한 전문가의 이름을 부르며 한 명당 두세 문장으로 답한다. "
                             "네 처음 답을 고쳐야 하면 고친다고 말한다. 확인하지 못한 것은 확인하지 못했다고 쓴다. 한국어로 쓴다.")
            text = str(out).strip()
            settle_tokens(agent)
        except Exception as e:
            text = f"(답을 받지 못함: {e})"
        return {"expert": key, "title": spec["title"], "to": [r["title"] for r in results if r["expert"] != key],
                "text": text, "seconds": round(time.time() - start, 1), "round": 2, "tools": used(), "facts": self._facts(agent)}

    def stance(self, key, question, results, talks):
        """대화를 다 읽은 뒤 전문가 한 명이 마지막 입장을 밝힌다: 합의하는지, 팀의 결론은 무엇인지, 남은 쟁점은 무엇인지."""
        mine = next(r for r in results if r["expert"] == key)
        log = "\n\n".join(f"[{t['title']}]\n{t['text'][:700]}" for t in talks)
        prompt = (f"너는 {self.goal}의 {mine['title']}다. 아래는 같은 질문을 두고 전문가들이 주고받은 대화다.\n"
                  "대화를 읽고 마지막 입장을 세 줄로 쓴다.\n"
                  "첫 줄: 다른 전문가들과 결론이 같으면 '동의', 다르면 '이견' 한 낱말만.\n"
                  "둘째 줄: '결론: ' 뒤에 팀이 내릴 결론을 한 문장으로.\n"
                  "셋째 줄: '남은 것: ' 뒤에 아직 풀리지 않은 쟁점이나 사람이 꼭 확인할 것을 한 문장으로.\n"
                  "대화에 없는 사실은 지어내지 않는다. 한국어로 쓴다.\n"
                  + (self.TALK_RULE + "도구로 얻은 결과가 있는 항목은 그 결과를 결론으로 삼는다. 그런 항목을 두고 '이견'이라고 하지 않는다.\n"
                     f"\n[전문가들이 도구로 얻은 결과]\n{self._fact_lines(list(results) + [t for t in talks if t.get('facts')])}\n" if self.careful else "")
                  + f"\n[질문]\n{question}\n\n[네가 처음 한 답]\n{mine['answer'][:600]}\n\n[대화]\n{log}")
        kw = dict(model=EXPERT_MODEL, messages=[{"role": "user", "content": prompt}], options={"temperature": 0.2})
        start = time.time()
        try:
            try:
                text = chat_counted(mine["title"], think=False, **kw)
            except Exception:
                text = chat_counted(mine["title"], **kw)
        except Exception as e:
            text = f"(입장을 받지 못함: {e})"
        return {"expert": key, "title": mine["title"], "to": [r["title"] for r in results if r["expert"] != key], "text": text,
                "seconds": round(time.time() - start, 1), "round": 3, "tools": [], "agree": text.lstrip("*# ").startswith("동의")}

    def synthesize(self, question, context, results, quiet=True, on_text=None, talks=()):
        """지휘자가 전문가들의 답을 모아 하나로 쓴다. quiet=False면 쓰는 대로 화면에 보여 준다."""
        parts = "\n\n".join(f"### {r['title']}의 분석\n{r['answer']}" for r in results)
        if talks:
            parts += ("\n\n### 전문가끼리 주고받은 대화 (뒤로 갈수록 나중에 한 말이다. 처음 분석과 다르면 대화에서 고친 내용을 따른다)\n"
                      + "\n".join(f"- [{t['round']}차] {t['title']}: {t['text'][:600]}" for t in talks))
        rule = ""
        if self.careful:                        # 도구가 돌려준 결과를 전문가의 의견보다 앞세우게 한다
            facts = self._fact_lines(list(results) + [t for t in talks if t.get("facts")])
            rule = ("근거에는 순서가 있다. (가) 도구가 돌려준 결과 (나) 전문가의 설명과 의견.\n"
                    "도구가 돌려준 결과가 있는 항목은 그 결과대로 적는다. 다른 전문가의 의견으로 바꾸거나 '판정할 수 없다'고 쓰지 않는다. "
                    "도구 결과가 있는 항목은 하나도 빠뜨리지 않는다. 의견이 달랐다면 도구 결과를 먼저 적고 다른 의견이 있었다고 덧붙인다.\n"
                    "단, 도구가 AI의 판단을 돌려준 것(사진을 보고 쓴 설명 등)은 틀릴 수 있다고 적는다.\n\n"
                    f"[도구가 돌려준 결과]\n{facts}\n\n")
        prompt = (f"너는 {self.goal}의 지휘자다. 아래 전문가 분석만 근거로 한국어 통합 답변을 쓴다.\n"
                  f"형식: {self.report}. 전문가 분석에 없는 내용은 쓰지 않는다. 전문가끼리 말이 다르면 다르다고 적는다. 전체 {'16' if self.careful else '12'}줄 이내.\n\n"
                  + rule + (f"[기초 자료]\n{context}\n\n" if context else "") + f"[질문]\n{question}\n\n{parts}")
        messages = [{"role": "user", "content": prompt}]
        if not quiet:
            return stream_answer(EXPERT_MODEL, messages, who="지휘자")
        if on_text:                             # 웹 화면: 쓰는 대로 지금까지의 글을 넘겨 준다
            text = ""
            for chunk in ollama.chat(model=EXPERT_MODEL, messages=messages, options={"temperature": 0.2}, stream=True):
                text += chunk.message.content or ""
                on_text(text)
                if chunk.done:
                    count_tokens("지휘자 (답 합치기)", chunk)
            return text.strip()
        r = ollama.chat(model=EXPERT_MODEL, messages=messages, options={"temperature": 0.2})
        count_tokens("지휘자 (답 합치기)", r)
        return r["message"]["content"].strip()

    def _ask_file(self, question, case, depth, router):
        """같은 질문을 전에 돌린 결과를 넣어 두는 파일. 질문, 대상, 대화하는 노력, 고르는 모델이 같으면 같은 파일이다."""
        key = json.dumps([question, case, depth, "student" if router else "teacher"], ensure_ascii=False)
        return self.folder / "asks" / (hashlib.sha1(key.encode()).hexdigest()[:16] + ".json")

    def _replay(self, r, quiet, tell):
        """전에 돌려 둔 결과를 다시 보여 준다. 모델을 부르지 않으므로 바로 끝난다."""
        say = (lambda *a: None) if quiet else print
        labels = {1: "서로의 답을 읽고 묻고 지적합니다", 2: "받은 질문과 남은 쟁점에 답합니다 (필요하면 도구를 다시 씁니다)", 3: "대화를 읽고 지금의 입장을 밝힙니다"}
        say(f"✅ 전에 돌려 둔 결과를 그대로 보여 줍니다. 모델을 다시 부르지 않습니다. (그때 걸린 시간 {r['seconds']}초)")
        say(f"👤 질문: {r['question']}" + (f"   (대상: {r['case']})" if r["case"] else ""))
        USAGE.clear()
        USAGE.extend({**u, "live": 0} for u in r["usage"]["by"])
        tell({"type": "steps", "names": [x["name"] for x in r["steps"]]})
        for x in r["steps"]:
            tell({"type": "step", "n": x["n"], "total": len(r["steps"]), "name": x["name"]})
            say(f"\n[{x['n']}/{len(r['steps'])}] {x['name']}  ({x['seconds']}초" + (f", {x['note']}" if x["note"] else "") + ")")
            if x["name"] == "전문가 고르기":
                tell({"type": "route", "experts": r["experts"], "seconds": r["route_seconds"]})
            if x["name"] == "전문가들이 일하기":
                for e in r["results"]:
                    tell({"type": "expert", "result": e})
                    if not quiet:
                        md(f"**🧑‍🔬 {e['title']}**  ({e['seconds']}초, 도구: {' → '.join(e['tools']) or '안 씀'})\n\n{e['answer']}")
            if x["name"] == "전문가끼리 의견 나누기":
                for i, t in enumerate(r["talks"]):
                    if i % max(1, len(r["experts"])) == 0:
                        n = i // max(1, len(r["experts"])) + 1
                        tell({"type": "round", "n": n, "label": labels.get(t["round"], "")})
                        say(f"\n  대화 {n}: {labels.get(t['round'], '')}")
                    tell({"type": "talk", **t})
                    if not quiet:
                        md(f"**💬 {t['title']}** → {', '.join(t['to'])}  ({t['seconds']}초" + (f", 도구: {' → '.join(t['tools'])}" if t.get("tools") else "") + f")\n\n{t['text']}")
            if x["name"] == "지휘자가 답을 합치기" and not quiet:
                md("**📋 통합 답변**\n\n" + r["answer"])
            tell({"type": "step_done", **x})
            if quiet:
                time.sleep(0.3)                 # 웹 화면에서 단계가 차례로 보이게 아주 잠깐 쉰다
        if not quiet:
            u = r["usage"]
            say(f"\n⏱️ 그때 걸린 시간 {r['seconds']}초  (" + ", ".join(f"{x['name']} {x['seconds']}초" for x in r["steps"]) + ")")
            md("\n".join(["**쓴 token**", "", "| 누가 | 모델을 부른 횟수 | 보낸 token | 받은 token |", "|---|---|---|---|"]
                         + [f"| {x['who']} | {x['calls']} | {x['in']:,} | {x['out']:,} |" for x in u["by"]]
                         + [f"| **합계** | {u['total']['calls']} | **{u['total']['in']:,}** | **{u['total']['out']:,}** |"]))
        return {**r, "replayed": True}

    def ask(self, question, case="", router=None, quiet=False, on_event=None, discuss="medium", reuse=False, careful=True):
        """질문 하나를 처음부터 끝까지 처리한다.

        case   : 검토 대상 (예: 사진 파일 이름). prepare 함수가 있을 때만 쓴다
        router : 질문을 받아 전문가 키 목록을 돌려주는 함수. 비워 두면 큰 모델이 고른다.
                 증류한 작은 모델로 바꾸려면 router=team.load_student()
        on_event : 진행 소식을 받을 함수 (웹 화면이 쓴다). 사진 읽기, 고른 전문가, 도구 호출, 끝난 전문가의 답, 쓰고 있는 통합 답변
        discuss : 전문가들이 각자 답한 뒤 서로 얼마나 깊이 대화할지.
                  "low"    서로의 답을 읽고 한 번씩 묻고 지적한다
                  "medium" 묻기 → 받은 질문에 답하기(도구를 다시 쓸 수 있다) → 입장 밝히기
                  "high"   답하기와 입장 밝히기를 모두 동의할 때까지 되풀이한다 (많아야 세 바퀴. 가장 오래 걸린다)
                  False    대화하지 않고 바로 지휘자에게 넘긴다
        careful : True(기본)면 도구가 돌려준 결과를 전문가의 의견보다 앞세운다. 대화할 때 남이 도구로 확인한 값을 짐작으로 뒤집지 않고,
                  의견이 갈리면 도구로 다시 확인하며, 자기 도구로 확인할 수 없는 것에는 판정하지 않는다.
                  False면 이런 규칙 없이 전문가의 말을 모두 같은 무게로 다룬다 (구성에 따라 결과가 어떻게 달라지는지 견줘 볼 때 쓴다)
        reuse   : True면 같은 질문을 전에 돌린 결과가 있을 때 모델을 다시 부르지 않고 그 결과를 보여 준다 (수업 시연용).
                  결과는 돌릴 때마다 distill/<팀 이름>/asks/ 에 저장된다
        """
        tell = on_event or (lambda e: None)
        say = (lambda *a: None) if quiet else print
        depth = {False: "", 0: "", "off": "", 1: "low", "low": "low", True: "medium", 3: "medium", "medium": "medium", "high": "high"}.get(discuss, "medium")
        self.careful = careful
        saved = self._ask_file(question, case, depth + ("|careful" if careful else ""), router)
        if reuse and saved.exists():
            return self._replay(json.loads(saved.read_text(encoding="utf-8")), quiet, tell)
        start = time.time()
        names = ((["기초 자료 만들기"] if self.prepare else []) + ["전문가 고르기", "전문가들이 일하기"]
                 + (["전문가끼리 의견 나누기"] if discuss not in (False, 0, "off") else []) + ["지휘자가 답을 합치기"])
        steps = []                              # 단계마다 걸린 시간을 적어 둔다
        USAGE.clear()                           # 이번 질문에서 쓴 token을 처음부터 다시 센다

        def step(name):
            """단계를 시작한다고 알리고, 끝났을 때 부를 함수를 돌려준다."""
            n, t = names.index(name) + 1, time.time()
            tell({"type": "step", "n": n, "total": len(names), "name": name})
            say(f"\n[{n}/{len(names)}] {name}")

            def done(note=""):
                steps.append({"n": n, "name": name, "seconds": round(time.time() - t, 1), "note": note})
                tell({"type": "step_done", **steps[-1]})
                tell({"type": "usage", **usage_summary()})
            return done

        say(f"👤 질문: {question}" + (f"   (대상: {case})" if case else ""))
        tell({"type": "steps", "names": names})
        context = ""
        if self.prepare:                        # 기초 자료 만들기(사진 읽기 등)도 오래 걸리므로 진행을 보여 준다
            done = step("기초 자료 만들기")
            context = self.prepare(case) if quiet else wait_for("기초 자료 만들기", lambda: self.prepare(case))
            done()

        done = step("전문가 고르기")
        t0 = time.time()
        keys = [k for k in (router or self.route_with_teacher)(question) if k in self.keys] or self.keys
        route_seconds = round(time.time() - t0, 2)
        say(f"🧭 고른 전문가: {', '.join(self.experts[k]['title'] for k in keys)}  (고르는 데 {route_seconds}초)")
        tell({"type": "route", "experts": keys, "seconds": route_seconds})
        done(", ".join(self.experts[k]["title"] for k in keys))

        done = step("전문가들이 일하기")
        if quiet:
            def work(k):
                r = self.ask_expert(k, question, context, on_event=on_event)
                tell({"type": "expert", "result": r})
                return r
            with ThreadPoolExecutor(max_workers=len(keys)) as pool:      # 고른 전문가들이 동시에 일한다
                results = list(pool.map(work, keys))
        else:                                   # 같은 일을 하되, 도구 호출과 끝난 사람의 답을 바로바로 보여 준다
            show = lambda name, r: md(f"**🧑‍🔬 {r['title']}**  ({r['seconds']}초, 도구: {' → '.join(r['tools']) or '안 씀'})\n\n{r['answer']}")
            results = run_together({self.experts[k]["title"]: (lambda k=k: self.ask_expert(k, question, context, quiet=False))
                                    for k in keys}, on_done=show)
        done(f"{len(keys)}명")

        talks = []
        if depth:                             # 각자 답한 뒤 대화한다: 묻고 지적하기 → 답하기(도구를 다시 쓸 수 있다) → 마지막 입장
            done = step("전문가끼리 의견 나누기")
            if len(results) < 2:
                say("전문가가 한 명이라 의견을 나눌 상대가 없습니다.")
                done("전문가가 한 명이라 건너뜀")
            else:
                heard = lambda name, t: md(f"**💬 {t['title']}** → {', '.join(t['to'])}  ({t['seconds']}초"
                                           + (f", 도구: {' → '.join(t['tools'])}" if t["tools"] else "") + f")\n\n{t['text']}")
                ask_each = ("서로의 답을 읽고 묻고 지적합니다", lambda k: self.comment(k, question, results))
                answer_each = ("받은 질문과 남은 쟁점에 답합니다 (필요하면 도구를 다시 씁니다)",
                               lambda k: self.reply(k, question, context, results, talks, quiet=quiet, on_event=on_event))
                stand = ("대화를 읽고 지금의 입장을 밝힙니다", lambda k: self.stance(k, question, results, talks))
                # 깊이에 따라 대화의 길이가 달라진다. high는 모두 동의할 때까지(많아야 세 바퀴) 답하기와 입장 밝히기를 되풀이한다
                plan = {"low": [ask_each], "medium": [ask_each, answer_each, stand],
                        "high": [ask_each, answer_each, stand, answer_each, stand, answer_each, stand]}[depth]
                n = 0
                for label, speak in plan:
                    last = [t for t in talks if t["round"] == 3][-len(keys):]
                    if depth == "high" and speak is answer_each[1] and n >= 5 and last and all(t.get("agree") for t in last):
                        say("\n  → 모두 같은 결론에 동의해서 대화를 마칩니다.")
                        break                   # 두 바퀴를 돌았고 모두 동의하면 더 묻지 않는다
                    n += 1
                    say(f"\n  대화 {n}: {label}")
                    tell({"type": "round", "n": n, "label": label})
                    if quiet:
                        def one(k, speak=speak):
                            t = speak(k)
                            tell({"type": "talk", **t})
                            return t
                        with ThreadPoolExecutor(max_workers=len(keys)) as pool:
                            said = list(pool.map(one, keys))
                    else:
                        said = run_together({self.experts[k]["title"]: (lambda k=k, speak=speak: speak(k)) for k in keys}, on_done=heard)
                    talks += said               # 다음 차례의 전문가들은 여기까지의 대화를 모두 읽는다
                last = [t for t in talks if t["round"] == 3][-len(keys):]
                agree = [t for t in last if t.get("agree")]
                note = (f"대화 {n}차례, {len(agree)}/{len(last)}명이 같은 결론에 동의" if last else f"대화 {n}차례")
                say(f"\n  → {note}")
                done(note)

        done = step("지휘자가 답을 합치기")
        if len(results) > 1:
            answer = self.synthesize(question, context, results, quiet=quiet, talks=talks,
                                     on_text=(lambda t: tell({"type": "draft", "text": t})) if on_event else None)
            done()
        else:                                   # 한 명이면 그 답이 곧 답이다
            answer = results[0]["answer"]
            say("전문가가 한 명이라 합칠 것이 없습니다. 그 답이 곧 답입니다.")
            done("전문가가 한 명이라 건너뜀")
        answer += "\n\n" + self.disclaimer
        seconds = round(time.time() - start)
        usage = usage_summary()
        say(f"\n{self.disclaimer}\n\n⏱️ 전체 {seconds}초  (" + ", ".join(f"{x['name']} {x['seconds']}초" for x in steps) + ")")
        if not quiet:
            md("\n".join(["**쓴 token** (모델에게 보낸 글과 받은 글의 양. 인터넷 너머의 모델이라면 이만큼 요금을 냅니다)", "",
                          "| 누가 | 모델을 부른 횟수 | 보낸 token | 받은 token |", "|---|---|---|---|"]
                         + [f"| {u['who']} | {u['calls']} | {u['in']:,} | {u['out']:,} |" for u in usage["by"]]
                         + [f"| **합계** | {usage['total']['calls']} | **{usage['total']['in']:,}** | **{usage['total']['out']:,}** |"]))
        result = {"question": question, "case": case, "context": context, "experts": keys, "route_seconds": route_seconds,
                  "results": results, "talks": talks, "answer": answer, "seconds": seconds, "steps": steps, "usage": usage}
        try:                                    # 다음에 reuse=True로 바로 보여 줄 수 있게 저장해 둔다
            saved.parent.mkdir(parents=True, exist_ok=True)
            saved.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass
        return result

    # ---------- 2) 증류: "전문가 고르기"를 작은 모델에게 가르치기 ----------
    @property
    def instruction(self):
        """제자에게 주는 짧은 지시문. 긴 역할 설명 없이도 답하도록 가르치는 것이 목표다."""
        return f"질문에 답하는 데 필요한 전문가를 {', '.join(self.keys)} 중에서 골라 쉼표로만 답하라.\n질문: "

    def build_distill_data(self, train_templates, test_templates, subjects=("",), n_valid=40, seed=7, reuse=False):
        """질문 틀과 낱말을 조합해 질문을 만들고, 스승(큰 모델)이 고른 전문가를 정답으로 붙여 저장한다.

        질문 틀의 {d} 자리에 subjects의 낱말이 하나씩 들어간다. {d}가 없는 틀은 질문이 하나뿐이라
        시험용으로 빠지지 않게 항상 학습에 넣고, 묻히지 않도록 세 번 넣는다.
        test_templates는 학습에 쓰지 않는다. 처음 보는 유형에도 맞는지 시험하는 데 쓴다.
        reuse=True면 전에 만들어 둔 학습 자료가 있을 때 스승에게 다시 묻지 않고 그것을 쓴다.
        """
        def expand(templates):
            qs = [fill(t, s) if "{d}" in t else t for t, s in itertools.product(templates, subjects)]
            return list(dict.fromkeys(q.strip() for q in qs))        # 같은 질문은 한 번만

        def label(questions, what):
            rows, status = [], Status(f"⏳ 스승이 {what} 질문 {len(questions)}개에 답을 답니다")
            for i, q in enumerate(questions, 1):
                rows.append({"question": q, "experts": ",".join(self.route_with_teacher(q))})
                status.update(f"⏳ 스승이 {what} 질문에 답을 다는 중: {i}/{len(questions)}  ({status.seconds():.0f}초)")
            status.done(f"✅ {what} 질문 {len(questions)}개에 답을 달았습니다  ({status.seconds():.0f}초)")
            return rows

        rng = random.Random(seed)
        train_q, test_q = expand(train_templates), expand(test_templates)
        rng.shuffle(train_q), rng.shuffle(test_q)
        self._rebuilt = False
        if reuse and (self.folder / "data" / "train.jsonl").exists():
            def read(name):
                rows = [json.loads(l)["messages"] for l in open(self.folder / "data" / f"{name}.jsonl", encoding="utf-8")]
                return [{"question": u["content"].split("질문: ", 1)[1], "experts": a["content"]} for u, a in rows]
            train_rows, test_rows = read("train") + read("valid"), read("test")
            same_questions = {r["question"] for r in train_rows + test_rows} == set(train_q + test_q)
            same_experts = all(e in self.keys for r in train_rows + test_rows for e in r["experts"].split(","))
            if same_questions and same_experts:     # 팀 파일이 그때와 같을 때만 다시 쓴다
                print(f"✅ 전에 만들어 둔 학습 자료를 그대로 씁니다. 스승에게 다시 묻지 않습니다.  ({self.folder / 'data'})")
                print(f"   학습용 {len(train_rows)}개, 처음 보는 유형의 시험용 {len(test_rows)}개")
                return train_rows, test_rows
            print("⚠️  팀 파일의 전문가나 질문 틀이 전에 학습 자료를 만들 때와 달라졌습니다. 학습 자료를 새로 만들고, 학습도 다시 합니다.")
            self._rebuilt = True
        print(f"학습용 질문 {len(train_q)}개, 처음 보는 유형의 시험용 질문 {len(test_q)}개에 스승({EXPERT_MODEL})이 답을 답니다.")
        train_rows, test_rows = label(train_q, "학습용"), label(test_q, "시험용")
        fixed = {t.strip() for t in train_templates if "{d}" not in t}       # 낱말 자리가 없는 질문
        varied = [r for r in train_rows if r["question"] not in fixed]
        n_valid = min(n_valid, len(varied) // 5)
        splits = {"train": varied[n_valid:] + [r for r in train_rows if r["question"] in fixed] * 3,
                  "valid": varied[:n_valid], "test": test_rows}
        rng.shuffle(splits["train"])
        (self.folder / "data").mkdir(parents=True, exist_ok=True)
        for name, rows in splits.items():
            with open(self.folder / "data" / f"{name}.jsonl", "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps({"messages": [{"role": "user", "content": self.instruction + r["question"]},
                                                     {"role": "assistant", "content": r["experts"]}]}, ensure_ascii=False) + "\n")
        meta = {"instruction": self.instruction, "teacher": EXPERT_MODEL, "student": STUDENT_MODEL, "keys": self.keys}
        (self.folder / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        print(f"✅ 저장: {self.folder / 'data'}  (학습 {len(splits['train'])}, 본 유형 시험 {len(splits['valid'])}, 처음 보는 유형 시험 {len(splits['test'])})")
        print("\n스승이 고른 조합별 개수 (한쪽으로 쏠리면 질문 틀을 고르게 늘리세요):")
        for combo, n in Counter(r["experts"] for r in train_rows).most_common():
            print(f"   {n:>4}  {combo}")
        return train_rows, test_rows

    def show_data(self, n=6):
        """학습 자료가 어떻게 생겼는지 보여 준다: 파일 위치와 개수, 표로 본 몇 줄, 파일에 실제로 적힌 한 줄."""
        self._need_data()
        read = lambda name: [json.loads(l) for l in open(self.folder / "data" / f"{name}.jsonl", encoding="utf-8")]
        train, valid, test = read("train"), read("valid"), read("test")
        picks = random.Random(3).sample(train, min(n, len(train)))
        lines = [f"학습 자료는 파일 세 개입니다 (`{self.folder.relative_to(ROOT)}/data/`). 한 줄이 문제 하나입니다.", "",
                 "| 파일 | 줄 수 | 쓰임 |", "|---|---|---|",
                 f"| `train.jsonl` | {len(train)} | 제자가 보고 배우는 문제 |",
                 f"| `valid.jsonl` | {len(valid)} | 학습 도중 중간 시험. 학습용과 같은 유형이지만 배우지는 않은 문제 |",
                 f"| `test.jsonl` | {len(test)} | 처음 보는 유형의 시험. 학습에 쓰지 않은 질문 틀로 만든 문제 |", "",
                 "**표로 보면** 문제(입력)와 정답(출력)의 짝입니다. 정답은 사람이 아니라 스승 모델이 달았습니다.", "",
                 "| 입력: 제자에게 주는 글 | 출력: 제자가 따라 써야 할 정답 |", "|---|---|"]
        for row in picks:
            user, assistant = row["messages"]
            lines.append(f"| {user['content'].replace(chr(10), '<br>')} | `{assistant['content']}` |")
        lines += ["", "**파일에 실제로 적힌 한 줄** (JSON). `user`가 입력, `assistant`가 정답입니다.", "",
                  "```json", json.dumps(picks[0], ensure_ascii=False, indent=2), "```"]
        md("\n".join(lines))

    def _need_data(self):
        if not (self.folder / "data" / "train.jsonl").exists():
            raise SystemExit(f"❌ 학습 자료가 없습니다. 먼저 build_distill_data()를 실행해 주세요. ({self.folder})")

    def train_student(self, iters=200, learning_rate=1e-4, reuse=False):
        """스승의 답으로 작은 모델을 학습시킨다 (MLX LoRA). 200번 반복에 아이맥에서 2분쯤 걸린다.

        100번 반복할 때마다 그때의 모델을 따로 저장해 둔다. compare_checkpoints()로 서로 비교할 수 있다.
        reuse=True면 전에 학습해 둔 모델이 있을 때 학습을 건너뛰고 그것을 쓴다.
        """
        self._need_data()
        adapter = self.folder / "adapter"
        if reuse and self.has_student() and not getattr(self, "_rebuilt", False):
            saved = ", ".join(str(n) for n in self.checkpoints())
            print(f"✅ 전에 학습해 둔 모델을 그대로 씁니다. 학습을 건너뜁니다.  ({adapter})")
            print(f"   저장된 중간 모델: {saved}번 반복 시점" if saved else "")
            return
        self._student.clear()
        for old in adapter.glob("*_adapters.safetensors"):      # 지난 학습의 중간 모델은 지운다
            old.unlink()
        shutil.rmtree(self.folder / "checkpoints", ignore_errors=True)
        for old in self.folder.glob("compare_*.json"):          # 지난 모델로 재 둔 비교 결과도 지운다
            old.unlink()
        cmd = [sys.executable, "-m", "mlx_lm", "lora", "--model", STUDENT_MODEL, "--train", "--data", str(self.folder / "data"),
               "--iters", str(iters), "--batch-size", "8", "--num-layers", "8", "--learning-rate", str(learning_rate),
               "--adapter-path", str(adapter), "--steps-per-report", "10", "--steps-per-eval", "100", "--save-every", "100", "--mask-prompt"]
        env = {**os.environ, "NO_COLOR": "1", "TERM": "dumb", "HF_HUB_DISABLE_PROGRESS_BARS": "1", "PYTHONUNBUFFERED": "1"}
        start = time.time()
        for attempt in (1, 2):
            free_gpu()                          # 큰 모델을 메모리에서 내려 학습에 자리를 내준다
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
            log = follow_training(proc, iters)
            if proc.wait() == 0:
                break
            if attempt == 1 and gpu_was_busy(log):
                print("⚠️  그래픽 장치(GPU)가 바빠서 학습이 중간에 멈췄습니다. 웹 화면이나 다른 노트북에서 모델을 쓰고 있다면 잠시 멈춰 주세요. 20초 뒤 한 번 더 해 봅니다.")
                time.sleep(20)
                continue
            raise SystemExit("❌ 학습이 실패했습니다.\n" + "\n".join(log[-15:]))
        size_mb = (adapter / "adapters.safetensors").stat().st_size / 1e6
        print(f"\n✅ 학습 완료  {time.time() - start:.0f}초   Adapter 파일 {size_mb:.1f}MB  ({adapter})")

    def has_student(self):
        return (self.folder / "adapter" / "adapters.safetensors").exists()

    def sizes(self, measure=False):
        """스승(큰 모델)과 제자(증류한 작은 모델)의 크기를 숫자로 돌려준다. "작다"가 얼마나 작은지 보여 주기 위한 것이다.

        measure=True면 제자를 실제로 한 번 돌려 메모리를 얼마나 쓰는지도 잰다.
        """
        gb = lambda n: round(n / 1e9, 2)
        teacher = {"name": EXPERT_MODEL, "params": "", "file_gb": None, "memory_gb": None}
        try:
            teacher["params"] = ollama.show(EXPERT_MODEL).details.parameter_size
            teacher["file_gb"] = next((gb(m.size) for m in ollama.list().models if m.model == EXPERT_MODEL), None)
            teacher["memory_gb"] = next((gb(m.size) for m in ollama.ps().models if m.model == EXPERT_MODEL), None)
        except Exception:
            pass
        m = re.search(r"-(\d+(?:\.\d+)?)([bm])-", STUDENT_MODEL.lower())
        student = {"name": STUDENT_MODEL.split("/")[-1], "params": (m[1] + ("B" if m[2] == "b" else "M")) if m else "", "file_gb": None, "memory_gb": None,
                   "adapter_mb": round((self.folder / "adapter" / "adapters.safetensors").stat().st_size / 1e6, 1) if self.has_student() else None}
        try:
            from huggingface_hub import snapshot_download
            folder = Path(snapshot_download(STUDENT_MODEL, local_files_only=True))
            student["file_gb"] = gb(sum(f.stat().st_size for f in folder.rglob("*") if f.is_file()))
        except Exception:
            pass
        if measure and self.has_student():
            import mlx.core as mx
            self._generate("준비", True)
            mx.reset_peak_memory()
            self._generate("이 질문은 누구에게 보낼까?", True)
            student["memory_gb"] = gb(mx.get_peak_memory())
        teacher["base"], student["base"] = TEACHER_BASE, STUDENT_BASE
        return {"teacher": teacher, "student": student, "others": OPEN_MODELS}

    def show_sizes(self):
        """스승, 제자, 그리고 인터넷 너머의 큰 모델(GPT, Claude 등)을 한 표로 비교해 보여 준다."""
        z = self.sizes(measure=True)
        t, st = z["teacher"], z["student"]
        ko = lambda p: {"B": "억 개", "M": "만 개"}[p[-1]].join([str(round(float(p[:-1]) * (10 if p[-1] == "B" else 100))), ""]) if p else "?"
        show = lambda v, unit="GB": f"{v}{unit}" if v is not None else "?"
        price = lambda p: f"100만 token당 ${p[0]} / ${p[1]}" if p else "0원 (전기 요금뿐)"
        rows = [f"| **제자** (증류한 작은 모델) | **{STUDENT_BASE}** + 내가 학습시킨 Adapter | 약 {ko(st['params'])} | {show(st['file_gb'])} + {show(st['adapter_mb'], 'MB')} | 약 {show(st['memory_gb'])} (방금 잰 값). 내 컴퓨터 | 지수 없음. **배운 일 하나만** 함 | 0원 (전기 요금뿐) |",
                f"| **스승** (큰 모델) | **{TEACHER_BASE}** | 약 {ko(t['params'])} | {show(t['file_gb'])} | 약 {show(t['memory_gb'] or t['file_gb'])}. 내 컴퓨터 | 찾지 못함 | 0원 (전기 요금뿐) |"]
        for o in OPEN_MODELS:
            size = "내려받을 수 없음" if o["file_gb"] is None else f"약 {o['file_gb']}GB" + (f" (제자의 약 {o['file_gb'] / st['file_gb']:,.0f}배)" if st["file_gb"] else "")
            rows.append(f"| {o['by']} | {o['name']} | {o['params']} | {size} | {o['where']} | {'찾지 못함' if o['index'] is None else o['index']} | {price(o['price'])} |")
        md("\n".join(["| | 모델 | 파라미터 수 | 파일 크기 | 돌리는 곳과 메모리 | 종합 성능 지수 | 비용 |", "|---|---|---|---|---|---|---|"] + rows + ["",
            "종합 성능 지수는 Artificial Analysis가 여러 시험 점수를 합쳐 매긴 값입니다(2026년 11월에 본 값. 높을수록 어려운 일을 잘함). "
            "지수가 13배라고 해서 13배 똑똑하다는 뜻은 아니지만, 최신 모델과 직접 돌릴 수 있는 공개 모델 사이의 차이가 크다는 것은 보여 줍니다."]))
        return z

    def show_cost(self, prices=None, computer_usd=600, years=3, watts_idle=7, watts_busy=30, kwh_usd=0.12):
        """전문가 고르기를 "싼 클라우드 모델에 지시문으로 시킬 때"와 "증류한 제자를 내 컴퓨터에서 돌릴 때"의 비용을 비교한다.

        질문 하나의 token 수와 제자가 답하는 시간은 이 컴퓨터에서 실제로 잰다. 나머지는 가정이므로 바꿔서 다시 계산해 본다.
          prices       : {모델 이름: (입력, 출력)} 100만 token당 달러. 비워 두면 2026년 11월에 알려진 요금 네 가지.
                         첫 줄은 실습의 스승과 같은 모델을 남의 서버에서 token당 요금으로 빌려 쓸 때의 값이다
          computer_usd : 제자를 돌릴 컴퓨터 값 (기본값: 메모리 16GB 맥 미니 정도)
          years        : 그 컴퓨터를 몇 년 쓰고 바꾸는지 (부품은 수명이 있으므로 값을 이 기간에 나눠 낸다고 본다)
          watts_idle, watts_busy : 쉴 때와 일할 때 쓰는 전력(W).  kwh_usd : 전기 1kWh 요금(달러)
        """
        prices = prices or {"Gemma 4 E4B 호스팅 (실습의 스승과 같은 모델, Fireworks)": (0.20, 0.20),
                            "GPT-6 Luna (가장 싼 축)": (0.10, 0.50), "GPT-6.1 Terra": (1, 5), "Claude Opus 5.5": (4, 20)}
        question = "이 질문은 누구에게 보내야 할까?"
        # 1) 질문 하나에 token이 몇 개 드는지: 큰 모델에게 지시문과 함께 물어서 센다
        schema = {"type": "object", "required": ["experts"], "properties": {"experts": {
            "type": "array", "minItems": 1, "maxItems": len(self.keys), "items": {"type": "string", "enum": self.keys}}}}
        kw = dict(model=EXPERT_MODEL, format=schema, options={"temperature": 0},
                  messages=[{"role": "system", "content": self.router_prompt}, {"role": "user", "content": question}])
        try:
            r = ollama.chat(think=False, **kw)
        except Exception:
            r = ollama.chat(**kw)
        tok_in, tok_out = r.prompt_eval_count, r.eval_count
        # 2) 제자가 질문 하나에 몇 초 걸리는지: 다섯 번 돌려 평균을 낸다
        self._generate(question, True)
        t0 = time.time()
        for _ in range(5):
            self._generate(question, True)
        sec = (time.time() - t0) / 5
        # 3) 계산
        own_call = sec / 3600 * (watts_busy - watts_idle) / 1000 * kwh_usd       # 질문 하나에 더 드는 전기 요금
        own_day = computer_usd / (years * 365) + 24 * watts_idle / 1000 * kwh_usd  # 컴퓨터를 새로 사서 켜 두는 하루 고정비
        capacity = int(86400 / sec)
        usd = lambda v: f"${v:,.2f}" if v >= 0.01 else f"${v:.6f}".rstrip("0") if v >= 0.000001 else "$0.000001 미만"
        rows = []
        for name, (p_in, p_out) in prices.items():
            call = (tok_in * p_in + tok_out * p_out) / 1e6
            even = own_day / (call - own_call) if call > own_call else None
            rows.append({"model": name, "call": call, "break_even_per_day": even})
        lines = [f"질문 하나: 지시문과 질문 **{tok_in} token** + 답 **{tok_out} token**. 제자는 질문 하나에 **{sec:.2f}초**, 하루에 최대 약 {capacity:,}번.", "",
                 "| | 질문 하나 | 하루 1,000번 | 하루 100,000번 | 새 컴퓨터를 사도 본전인 횟수 |", "|---|---|---|---|---|"]
        for x in rows:
            even = "본전을 못 뽑음" if x["break_even_per_day"] is None else f"하루 약 {x['break_even_per_day']:,.0f}번부터" + (" (제자 한 대의 한계를 넘음)" if x["break_even_per_day"] > capacity else "")
            lines.append(f"| {x['model']}에 지시문으로 | {usd(x['call'])} | {usd(x['call'] * 1000)} | {usd(x['call'] * 100000)} | {even} |")
        lines.append(f"| **증류한 제자** (이미 있는 컴퓨터) | {usd(own_call)} | {usd(own_call * 1000)} | {usd(own_call * 100000)} | 첫 질문부터 더 쌈 (전기 요금뿐) |")
        lines.append(f"| **증류한 제자** (컴퓨터를 새로 삼) | | {usd(own_day + own_call * 1000)} | {usd(own_day + own_call * 100000)} | 하루 고정비 {usd(own_day)} |")
        lines += ["", f"가정: 컴퓨터 ${computer_usd:,}를 {years}년 쓰고 바꿈, 전력은 쉴 때 {watts_idle}W와 일할 때 {watts_busy}W, 전기 1kWh에 ${kwh_usd}. "
                      "요금은 100만 token당 달러(입력, 출력): " + ", ".join(f"{n} {a}/{b}" for n, (a, b) in prices.items()) + ". 값을 바꿔 다시 계산해 보세요."]
        md("\n".join(lines))
        return {"tokens_in": tok_in, "tokens_out": tok_out, "student_seconds": round(sec, 3), "own_per_call": own_call, "own_per_day": own_day, "rows": rows}

    def _saved(self, name, reuse, show):
        """reuse=True이고 전에 저장해 둔 결과가 있으면 그것을 보여 주고 돌려준다. 없으면 None."""
        path = self.folder / f"{name}.json"
        if reuse and path.exists() and not getattr(self, "_rebuilt", False):
            rows = json.loads(path.read_text(encoding="utf-8"))
            print(f"✅ 전에 재 둔 결과를 그대로 보여 줍니다. 다시 재지 않습니다.  ({path.relative_to(ROOT)})")
            show(rows)
            return rows
        return None

    def compare_models(self, models=("llama3.2:3b", "gemma3:4b"), n=30, reuse=False):
        """같은 시험을 세 가지 방식으로 보게 해서 비교한다.

          1) 학습하지 않은 범용 모델에게 지시문만 주고 시키기 (models에 적은 것 가운데 이 컴퓨터에 받아 둔 모델)
          2) 증류하기 전의 제자 (바탕 모델 그대로)
          3) 증류한 제자
        점수는 "스승과 같은 전문가 조합을 고른 비율"이다. 사람이 단 정답이 아니라 스승의 답이 기준이라는 점에 주의한다.
        시간을 아끼려고 유형마다 n문제만 뽑아 시험한다. reuse=True면 전에 재 둔 결과가 있을 때 그것을 보여 준다.
        """
        self._need_data()

        def show(rows):
            lines = ["| 방식 | 모델 | 파라미터 | 파일 크기 | 본 유형 | 처음 보는 유형 | 질문 하나에 |", "|---|---|---|---|---|---|---|"]
            for r in rows:
                lines.append(f"| {r['how']} | {r['model']} | {r['params']} | {r['file_gb']}GB | {r['valid']:.0%} | {r['test']:.0%} | {r['seconds']}초 |")
            lines += ["", f"점수는 스승({TEACHER_BASE})과 같은 전문가 조합을 고른 비율입니다(유형마다 {rows[0].get('n', '?')}문제). "
                          "스승의 답이 기준이므로, 다른 모델이 스승과 다르게 골랐다고 해서 꼭 틀린 것은 아닙니다."]
            md("\n".join(lines))
        saved = self._saved("compare_models", reuse, show)
        if saved:
            return saved
        read = lambda name: [(u["content"].split("질문: ", 1)[1], a["content"]) for u, a in
                             (json.loads(l)["messages"] for l in open(self.folder / "data" / f"{name}.jsonl", encoding="utf-8"))]
        pick = lambda items: random.Random(5).sample(items, min(n, len(items)))
        valid, test = pick(read("valid")), pick(read("test"))
        try:
            have = {m.model: m for m in ollama.list().models}
        except Exception:
            have = {}
        rows = []
        for name in models:
            if name not in have:
                print(f"   (건너뜀) {name}: 이 컴퓨터에 받아 둔 모델이 아닙니다. 해 보려면 터미널에서 ollama pull {name}")
                continue
            status, scores, start = Status(f"⏳ {name}에게 지시문만 주고 시험을 보게 합니다"), [], time.time()
            for kind, items in (("valid", valid), ("test", test)):
                ok = 0
                for i, (q, answer) in enumerate(items, 1):
                    try:
                        ok += ",".join(self.route_with_teacher(q, model=name)) == answer
                    except Exception:
                        pass                    # 형식에 맞지 않는 답은 틀린 것으로 센다
                    status.update(f"⏳ {name}: {'본 유형' if kind == 'valid' else '처음 보는 유형'} {i}/{len(items)}문제  ({status.seconds():.0f}초)")
                scores.append(ok / len(items))
            n = len(valid) + len(test)
            status.done(f"✅ {name}: 시험 끝  ({status.seconds():.0f}초)")
            try:
                params = ollama.show(name).details.parameter_size
            except Exception:
                params = "?"
            rows.append({"how": "지시문만 (학습 없음)", "model": name, "params": params, "file_gb": round(have[name].size / 1e9, 1), "n": len(test),
                         "valid": scores[0], "test": scores[1], "seconds": round((time.time() - start) / n, 2)})
            try:
                ollama.generate(model=name, prompt="", keep_alive=0)      # 다 쓴 모델은 메모리에서 내린다
            except Exception:
                pass
        z = self.sizes()["student"]
        score = lambda items, trained: sum(self._generate(q, trained) == a for q, a in items) / len(items)
        for how, trained in (("증류 전 (바탕 모델 그대로)", False), ("**증류한 제자**", True)):
            if trained and not self.has_student():
                continue
            t0 = time.time()
            v, t = score(valid, trained), score(test, trained)
            rows.append({"how": how, "model": STUDENT_BASE + (" + 내 Adapter" if trained else ""), "params": z["params"], "file_gb": z["file_gb"], "n": len(test),
                         "valid": v, "test": t, "seconds": round((time.time() - t0) / (len(valid) + len(test)), 2)})
        (self.folder / "compare_models.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        show(rows)
        return rows

    def checkpoints(self):
        """학습 도중에 저장된 중간 모델들의 반복 횟수 (예: [100, 200, 300])."""
        return sorted(int(f.name.split("_")[0]) for f in (self.folder / "adapter").glob("*_adapters.safetensors"))

    def _adapter_at(self, n):
        """n번 반복 시점의 중간 모델을 불러올 수 있게 폴더를 하나 만들어 준다."""
        folder = self.folder / "checkpoints" / str(n)
        if not (folder / "adapters.safetensors").exists():
            folder.mkdir(parents=True, exist_ok=True)
            shutil.copy(self.folder / "adapter" / "adapter_config.json", folder / "adapter_config.json")
            shutil.copy(self.folder / "adapter" / f"{n:07d}_adapters.safetensors", folder / "adapters.safetensors")
        return str(folder)

    def _generate(self, question, trained):
        """작은 모델에게 묻는다. trained=False면 학습 전의 모델, 숫자면 그만큼 반복한 시점의 중간 모델."""
        from mlx_lm import generate, load
        if trained not in self._student:
            path = None if trained is False else str(self.folder / "adapter") if trained is True else self._adapter_at(trained)
            self._student[trained] = load(STUDENT_MODEL, adapter_path=path)
        model, tokenizer = self._student[trained]
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": self.instruction + question}],
                                               add_generation_prompt=True, tokenize=False)
        text = generate(model, tokenizer, prompt=prompt, max_tokens=16, verbose=False)
        if trained is True:     # 증류한 제자가 쓴 token도 센다 (작은 모델이라 몇 개 안 된다)
            count_tokens("지휘자 (전문가 고르기, 증류한 제자)", tokens_in=len(tokenizer.encode(prompt)), tokens_out=len(tokenizer.encode(text)))
        return text.split("<")[0].strip().lower().replace(" ", "")

    def load_student(self):
        """증류한 작은 모델로 전문가를 고르는 함수를 돌려준다. team.ask(..., router=이 함수)로 쓴다."""
        if not self.has_student():
            raise SystemExit("❌ 학습된 Adapter가 없습니다. 먼저 train_student()를 실행해 주세요.")

        def route_with_student(question):
            picked = [e for e in self._generate(question, True).split(",") if e in self.keys]
            return self._sorted(picked) or self.keys         # 알 수 없는 답이면 전원 호출
        route_with_student("준비")                            # 모델을 미리 메모리에 올려 둔다
        return route_with_student

    def compare_checkpoints(self, reuse=False):
        """학습을 100번, 200번, ... 했을 때의 모델을 같은 시험으로 비교한다. 표와 그래프로 보여 준다.

        "학습을 얼마나 해야 충분한가"를 눈으로 보기 위한 것이다. 0번은 학습 전의 모델이다.
        reuse=True면 전에 재 둔 결과가 있을 때 그것을 보여 준다.
        """
        self._need_data()

        def show(rows):
            print("\n   반복 횟수    본 유형    처음 보는 유형")
            for r in rows:
                print(f"   {str(r['iters']) + '번':>8}    {r['valid']:>6.0%}    {r['test']:>10.0%}" + ("   (학습 전)" if r["iters"] == 0 else ""))
            try:
                import matplotlib.pyplot as plt
                plt.rcParams["font.family"], plt.rcParams["axes.unicode_minus"] = "AppleGothic", False
                fig, ax = plt.subplots(figsize=(7, 3.4))
                xs = [r["iters"] for r in rows]
                ax.plot(xs, [r["valid"] * 100 for r in rows], marker="o", color="#1F4ED8", label="본 유형")
                ax.plot(xs, [r["test"] * 100 for r in rows], marker="o", color="#0E8A74", label="처음 보는 유형")
                ax.set(xlabel="학습 반복 횟수", ylabel="스승과 같은 답 (%)", ylim=(0, 105), xticks=xs)
                ax.grid(alpha=0.3), ax.legend(loc="lower right"), ax.spines[["top", "right"]].set_visible(False)
                plt.show()
            except Exception:
                pass                             # 그래프를 못 그려도 표는 이미 보여 줬다
        saved = self._saved("compare_checkpoints", reuse, show)
        if saved:
            return saved
        points = self.checkpoints()
        if not points:
            raise SystemExit("❌ 저장된 중간 모델이 없습니다. 먼저 train_student()로 학습해 주세요.")
        rows, status = [], Status(f"⏳ 학습 전과 {', '.join(map(str, points))}번 시점의 모델에게 같은 시험을 보게 합니다")
        for n in [0] + points:
            status.update(f"⏳ {n}번 반복 시점의 모델이 시험 보는 중: 모델 {len(rows)}/{len(points) + 1}  ({status.seconds():.0f}초)")
            which = False if n == 0 else n
            rows.append({"iters": n, "valid": self.exam("valid", trained=which, quiet=True), "test": self.exam("test", trained=which, quiet=True)})
            if n:
                self._student.pop(n, None)      # 다 쓴 중간 모델은 메모리에서 내린다
        status.done(f"✅ 모델 {len(rows)}개의 시험이 끝났습니다  ({status.seconds():.0f}초)")
        (self.folder / "compare_checkpoints.json").write_text(json.dumps(rows), encoding="utf-8")
        show(rows)
        return rows

    def exam(self, split="test", trained=True, show=0, quiet=False):
        """제자가 스승과 똑같은 조합을 고른 비율. split은 "valid"(본 유형) 또는 "test"(처음 보는 유형).

        trained에 숫자를 넣으면 그만큼 반복한 시점의 중간 모델로 시험을 본다.
        """
        self._need_data()
        items = [json.loads(l)["messages"] for l in open(self.folder / "data" / f"{split}.jsonl", encoding="utf-8")]
        kind = "본 유형" if split == "valid" else "처음 보는 유형"
        if quiet:
            return sum(self._generate(u["content"].split("질문: ", 1)[1], trained) == a["content"] for u, a in items) / len(items)
        correct, status = 0, Status(f"⏳ 제자가 시험을 봅니다 ({kind} {len(items)}문제). 모델을 불러오는 중")
        for i, (user, assistant) in enumerate(items):
            q = user["content"].split("질문: ", 1)[1]
            out = self._generate(q, trained)
            correct += out == assistant["content"]
            if i < show:
                print(f"   {'⭕' if out == assistant['content'] else '❌'} {q[:34]:<36} 제자: {out[:30]:<32} 스승: {assistant['content']}")
            status.update(f"⏳ 제자가 시험 보는 중 ({kind}): {i + 1}/{len(items)}문제  ({status.seconds():.0f}초)")
        status.done(f"✅ 시험 끝 ({kind}): {len(items)}문제 가운데 {correct}문제가 스승과 같음  ({status.seconds():.0f}초)")
        return correct / len(items)
