"""내 주제로 Multi-Agent 팀을 만드는 틀.

전문가 목록(이름, 역할, 도구)만 적으면 아래 세 가지가 따라온다.
  1) 지휘자: 질문마다 필요한 전문가를 고르고, 동시에 일을 시키고, 답을 하나로 합친다   → team.ask()
  2) 증류: "전문가 고르기"를 큰 모델(스승)에게 시켜 자료를 만들고 작은 모델(제자)을 학습시킨다
           → team.build_distill_data(), team.train_student(), team.load_student()
  3) 서비스: team_service.py가 이 팀을 웹 화면이나 MCP 서버로 띄운다

완성 예시는 xray_team.py, 설명은 example/10_my_multi_agent.ipynb 에 있다.
교육·연구용이며 진료나 임상 판단을 대신하지 않는다.
"""
import itertools
import json
import os
import random
import re
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")      # 모델을 불러올 때 진행 막대를 숨긴다

import ollama
from strands import Agent, tool
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

    def route_with_teacher(self, question):
        """큰 모델이 필요한 전문가를 고른다. 답은 정해진 형식(JSON)으로만 받는다."""
        schema = {"type": "object", "required": ["experts"], "properties": {"experts": {
            "type": "array", "minItems": 1, "maxItems": len(self.keys), "items": {"type": "string", "enum": self.keys}}}}
        kw = dict(model=EXPERT_MODEL, format=schema, options={"temperature": 0},
                  messages=[{"role": "system", "content": self.router_prompt}, {"role": "user", "content": question}])
        try:
            r = ollama.chat(think=False, **kw)     # 생각 과정을 끄면 빨라진다
        except Exception:
            r = ollama.chat(**kw)
        return self._sorted(json.loads(r["message"]["content"])["experts"])

    def ask_expert(self, key, question, context=""):
        """전문가 한 명에게 묻는다. 매번 새 Agent를 만들어 다른 질문의 기억이 섞이지 않게 한다."""
        spec = self.experts[key]
        agent = Agent(model=OllamaModel(host=OLLAMA_HOST, model_id=EXPERT_MODEL, temperature=0.2),
                      system_prompt=spec["prompt"] + self.rules, tools=spec["tools"], callback_handler=None)
        used = lambda: [b["toolUse"]["name"] for m in agent.messages for b in m.get("content", []) if "toolUse" in b]
        start = time.time()
        try:
            text = str(agent((f"[기초 자료]\n{context}\n\n" if context else "") + f"[질문]\n{question}")).strip()
            if not used():      # 도구 없이 기억만으로 답했으면 한 번 더 시킨다
                text = str(agent("도구를 한 번도 쓰지 않았다. 먼저 도구를 불러 근거를 찾고, 그 결과만으로 다시 답하라.")).strip()
        except Exception as e:
            text = f"(분석 실패: {e})"
        return {"expert": key, "title": spec["title"], "answer": text, "tools": used(), "seconds": round(time.time() - start, 1)}

    def synthesize(self, question, context, results):
        """지휘자가 전문가들의 답을 모아 하나로 쓴다."""
        parts = "\n\n".join(f"### {r['title']}의 분석\n{r['answer']}" for r in results)
        prompt = (f"너는 {self.goal}의 지휘자다. 아래 전문가 분석만 근거로 한국어 통합 답변을 쓴다.\n"
                  f"형식: {self.report}. 전문가 분석에 없는 내용은 쓰지 않는다. 전문가끼리 말이 다르면 다르다고 적는다. 전체 12줄 이내.\n\n"
                  + (f"[기초 자료]\n{context}\n\n" if context else "") + f"[질문]\n{question}\n\n{parts}")
        r = ollama.chat(model=EXPERT_MODEL, messages=[{"role": "user", "content": prompt}], options={"temperature": 0.2})
        return r["message"]["content"].strip()

    def ask(self, question, case="", router=None, quiet=False):
        """질문 하나를 처음부터 끝까지 처리한다.

        case   : 검토 대상 (예: 사진 파일 이름). prepare 함수가 있을 때만 쓴다
        router : 질문을 받아 전문가 키 목록을 돌려주는 함수. 비워 두면 큰 모델이 고른다.
                 증류한 작은 모델로 바꾸려면 router=team.load_student()
        """
        say = (lambda *a: None) if quiet else print
        start = time.time()
        context = self.prepare(case) if self.prepare else ""
        say(f"👤 질문: {question}" + (f"   (대상: {case})" if case else ""))
        t0 = time.time()
        keys = [k for k in (router or self.route_with_teacher)(question) if k in self.keys] or self.keys
        route_seconds = round(time.time() - t0, 2)
        say(f"🧭 고른 전문가: {', '.join(self.experts[k]['title'] for k in keys)}  (고르는 데 {route_seconds}초)")
        with ThreadPoolExecutor(max_workers=len(keys)) as pool:      # 고른 전문가들이 동시에 일한다
            results = list(pool.map(lambda k: self.ask_expert(k, question, context), keys))
        for r in results:
            say(f"\n🧑‍🔬 {r['title']}  ({r['seconds']}초, 도구: {' → '.join(r['tools']) or '안 씀'})\n{r['answer']}")
        answer = self.synthesize(question, context, results) if len(results) > 1 else results[0]["answer"]
        answer += "\n\n" + self.disclaimer
        seconds = round(time.time() - start)
        say("\n📋 통합 답변\n" + answer)
        say(f"\n⏱️ {seconds}초, 전문가 {len(keys)}명")
        return {"question": question, "case": case, "context": context, "experts": keys, "route_seconds": route_seconds,
                "results": results, "answer": answer, "seconds": seconds}

    # ---------- 2) 증류: "전문가 고르기"를 작은 모델에게 가르치기 ----------
    @property
    def instruction(self):
        """제자에게 주는 짧은 지시문. 긴 역할 설명 없이도 답하도록 가르치는 것이 목표다."""
        return f"질문에 답하는 데 필요한 전문가를 {', '.join(self.keys)} 중에서 골라 쉼표로만 답하라.\n질문: "

    def build_distill_data(self, train_templates, test_templates, subjects=("",), n_valid=40, seed=7):
        """질문 틀과 낱말을 조합해 질문을 만들고, 스승(큰 모델)이 고른 전문가를 정답으로 붙여 저장한다.

        질문 틀의 {d} 자리에 subjects의 낱말이 하나씩 들어간다. {d}가 없는 틀은 그대로 한 번 쓴다.
        test_templates는 학습에 쓰지 않는다. 처음 보는 유형에도 맞는지 시험하는 데 쓴다.
        """
        def expand(templates):
            qs = [fill(t, s) if "{d}" in t else t for t, s in itertools.product(templates, subjects)]
            return list(dict.fromkeys(q.strip() for q in qs))        # 같은 질문은 한 번만

        def label(questions, what):
            rows, start = [], time.time()
            for i, q in enumerate(questions, 1):
                rows.append({"question": q, "experts": ",".join(self.route_with_teacher(q))})
                if i % 60 == 0 or i == len(questions):
                    print(f"   {what}: {i}/{len(questions)}  ({time.time() - start:.0f}초)")
            return rows

        rng = random.Random(seed)
        train_q, test_q = expand(train_templates), expand(test_templates)
        rng.shuffle(train_q), rng.shuffle(test_q)
        print(f"학습용 질문 {len(train_q)}개, 처음 보는 유형의 시험용 질문 {len(test_q)}개에 스승({EXPERT_MODEL})이 답을 답니다.")
        train_rows, test_rows = label(train_q, "학습용"), label(test_q, "시험용")
        n_valid = min(n_valid, len(train_rows) // 5)
        splits = {"train": train_rows[n_valid:], "valid": train_rows[:n_valid], "test": test_rows}
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

    def _need_data(self):
        if not (self.folder / "data" / "train.jsonl").exists():
            raise SystemExit(f"❌ 학습 자료가 없습니다. 먼저 build_distill_data()를 실행해 주세요. ({self.folder})")

    def train_student(self, iters=600, learning_rate=1e-4):
        """스승의 답으로 작은 모델을 학습시킨다 (MLX LoRA). 아이맥에서 5분쯤 걸린다."""
        self._need_data()
        self._student.clear()
        adapter = self.folder / "adapter"
        cmd = [sys.executable, "-m", "mlx_lm", "lora", "--model", STUDENT_MODEL, "--train", "--data", str(self.folder / "data"),
               "--iters", str(iters), "--batch-size", "8", "--num-layers", "8", "--learning-rate", str(learning_rate),
               "--adapter-path", str(adapter), "--steps-per-report", "100", "--steps-per-eval", "200", "--mask-prompt"]
        env = {**os.environ, "NO_COLOR": "1", "TERM": "dumb", "HF_HUB_DISABLE_PROGRESS_BARS": "1"}
        start = time.time()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        for line in proc.stdout:
            line = line.rstrip()
            if any(k in line for k in ["Trainable", "train_loss", "val", "save", "Error", "error"]) and "│" not in line:
                print(line)
        if proc.wait() != 0:
            raise SystemExit("❌ 학습이 실패했습니다. 위의 메시지를 확인해 주세요.")
        size_mb = (adapter / "adapters.safetensors").stat().st_size / 1e6
        print(f"\n✅ 학습 완료  {time.time() - start:.0f}초   Adapter 파일 {size_mb:.1f}MB  ({adapter})")

    def has_student(self):
        return (self.folder / "adapter" / "adapters.safetensors").exists()

    def _generate(self, question, trained):
        """작은 모델에게 묻는다. trained=False면 학습 전의 모델."""
        from mlx_lm import generate, load
        if trained not in self._student:
            self._student[trained] = load(STUDENT_MODEL, adapter_path=str(self.folder / "adapter") if trained else None)
        model, tokenizer = self._student[trained]
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": self.instruction + question}],
                                               add_generation_prompt=True, tokenize=False)
        return generate(model, tokenizer, prompt=prompt, max_tokens=16, verbose=False).split("<")[0].strip().lower().replace(" ", "")

    def load_student(self):
        """증류한 작은 모델로 전문가를 고르는 함수를 돌려준다. team.ask(..., router=이 함수)로 쓴다."""
        if not self.has_student():
            raise SystemExit("❌ 학습된 Adapter가 없습니다. 먼저 train_student()를 실행해 주세요.")

        def route_with_student(question):
            picked = [e for e in self._generate(question, True).split(",") if e in self.keys]
            return self._sorted(picked) or self.keys         # 알 수 없는 답이면 전원 호출
        route_with_student("준비")                            # 모델을 미리 메모리에 올려 둔다
        return route_with_student

    def exam(self, split="test", trained=True, show=0):
        """제자가 스승과 똑같은 조합을 고른 비율. split은 "valid"(본 유형) 또는 "test"(처음 보는 유형)."""
        self._need_data()
        items = [json.loads(l)["messages"] for l in open(self.folder / "data" / f"{split}.jsonl", encoding="utf-8")]
        correct = 0
        for i, (user, assistant) in enumerate(items):
            q = user["content"].split("질문: ", 1)[1]
            out = self._generate(q, trained)
            correct += out == assistant["content"]
            if i < show:
                print(f"   {'⭕' if out == assistant['content'] else '❌'} {q[:34]:<36} 제자: {out[:30]:<32} 스승: {assistant['content']}")
        return correct / len(items)
