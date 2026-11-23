"""완성 예시: 흉부 엑스레이 판독을 돕는 Multi-Agent 팀.

  영상 판독 보조 전문가 : 사진을 직접 본다 (Vision 모델)
  유사 사례 전문가      : 사람이 판독 요약을 붙여 둔 사례 모음에서 비슷한 사례를 찾는다 (Embedding 검색)
  문헌 전문가           : PubMed에서 논문 근거를 찾는다

1일차 image_mcp.py에서는 Vision 모델이 쓴 설명문으로 검색했다. 그때 모델이 없는 소견을 지어내는
문제가 있었기 때문에, 여기서는 사례 검색의 근거를 "사람이 쓴 판독 요약"(data/xray/cases.json)으로 바꿨다.
사진은 모두 공개 라이선스 자료다(data/xray/SOURCES.md). 실제 환자 사진은 넣지 않는다.
교육·연구용이며 판독이나 진단을 대신하지 않는다.
"""
import json
import os
from pathlib import Path

import numpy as np
import ollama

import drug_tools as dt
from agent_team import EXPERT_MODEL, Team

XRAY_DIR = Path(__file__).resolve().parent / "data" / "xray"
VISION_MODEL = os.getenv("VISION_MODEL", EXPERT_MODEL)      # gemma4:e4b는 사진도 볼 수 있다
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")
LIBRARY = json.loads((XRAY_DIR / "cases.json").read_text(encoding="utf-8"))
CURRENT = {"path": None}        # 지금 검토 중인 사진
_cache = {}


def _look(path, prompt):
    kw = dict(model=VISION_MODEL, options={"temperature": 0},
              messages=[{"role": "user", "content": prompt, "images": [str(path)]}])
    try:
        r = ollama.chat(think=False, **kw)
    except Exception:
        r = ollama.chat(**kw)
    return r["message"]["content"].strip()


def prepare(case):
    """검토할 사진을 정하고, 모든 전문가에게 줄 기초 자료(사진 설명)를 만든다."""
    path = Path(case).expanduser()
    path = path if path.is_file() else XRAY_DIR / case
    if not path.is_file():
        raise ValueError(f"사진을 찾을 수 없습니다: {case}  (data/xray 폴더의 파일 이름이나 전체 경로를 넣으세요)")
    CURRENT["path"] = path
    if path not in _cache:
        _cache[path] = _look(path, "이 흉부 엑스레이 사진에 보이는 것을 한국어 4문장 이내로 설명하라. "
                                   "촬영 방향, 좌우 폐의 밝기 차이, 눈에 띄는 음영의 위치를 적는다. 진단명은 쓰지 말고 보이는 것만 쓴다.")
    return f"사진 파일: {path.name}\n[AI가 쓴 사진 설명. 틀릴 수 있다]\n{_cache[path]}"


# ---------- 도구: 평범한 파이썬 함수다. 설명글(docstring)을 보고 Agent가 언제 쓸지 정한다 ----------
def look_at_image(question: str) -> str:
    """지금 검토 중인 엑스레이 사진을 다시 보고 질문에 답한다. question에는 확인할 것을 구체적으로 적는다 (예: "오른쪽 폐 아래쪽에 하얀 음영이 있는가?")."""
    if not CURRENT["path"]:
        return "검토 중인 사진이 없다."
    return _look(CURRENT["path"], question + "\n한국어 3문장 이내로, 사진에서 보이는 것만 답하라. 확실하지 않으면 확실하지 않다고 쓴다.")


def search_similar_cases(findings: str) -> list:
    """사람이 판독 요약을 붙여 둔 사례 모음에서 소견이 비슷한 사례 3건을 찾는다. findings에는 사진에서 보이는 소견을 한국어로 적는다 (예: "오른쪽 폐 중간에 경계가 뚜렷한 하얀 음영")."""
    if "library" not in _cache:
        texts = [f"{c['label']}. {c['summary']}" for c in LIBRARY]
        _cache["library"] = np.array(ollama.embed(model=EMBED_MODEL, input=texts)["embeddings"])
    vecs = _cache["library"]
    q = np.array(ollama.embed(model=EMBED_MODEL, input=findings)["embeddings"][0])
    sims = vecs @ q / (np.linalg.norm(vecs, axis=1) * np.linalg.norm(q))
    current = CURRENT["path"].name if CURRENT["path"] else ""
    ranked = [i for i in np.argsort(-sims) if LIBRARY[i]["file"] != current][:3]     # 지금 보는 사진 자신은 뺀다
    return [{"file": LIBRARY[i]["file"], "label": LIBRARY[i]["label"], "summary": LIBRARY[i]["summary"],
             "similarity": round(float(sims[i]), 3)} for i in ranked]


def pubmed_search(query: str) -> dict:
    """PubMed에서 논문을 검색한다. 검색어는 영문으로 쓴다 (예: "lobar pneumonia chest radiograph findings")."""
    return dt.pubmed_search(query, 5)


TEAM = Team(
    name="xray",
    goal="흉부 엑스레이 판독을 돕는 팀",
    experts={
        "reader": {
            "title": "영상 판독 보조 전문가",
            "role": "지금 사진을 직접 다시 봐야 답할 수 있는 것. 음영의 위치와 모양, 좌우 비교, 촬영 상태",
            "prompt": "너는 흉부 엑스레이를 보는 영상의학 보조 전문가다. 사진을 다시 확인해 보이는 것을 위치와 함께 설명한다. 진단을 내리지 않고, 보이지 않는 것은 지어내지 않는다.",
            "tools": [look_at_image],
        },
        "cases": {
            "title": "유사 사례 전문가",
            "role": "판독이 끝난 과거 사례를 찾아 지금 사진과 비교하는 것",
            "prompt": "너는 과거 판독 사례를 찾아 비교하는 전문가다. 기초 자료의 소견으로 비슷한 사례를 찾고, 어떤 점이 비슷하고 다른지 말한다. 사례의 라벨은 사람이 붙인 것이다. 유사도가 낮으면 낮다고 말한다.",
            "tools": [search_similar_cases],
        },
        "literature": {
            "title": "문헌 전문가",
            "role": "논문 근거가 필요한 것. 질환의 일반적인 영상 소견, 감별할 질환, 추가 검사, 검사의 정확도",
            "prompt": "너는 의학 문헌을 찾는 전문가다. PubMed에서 찾은 논문의 제목과 연도를 근거로 답한다. 논문에 없는 내용은 쓰지 않는다.",
            "tools": [pubmed_search],
        },
    },
    report="(1) 한 줄 요약 (2) 사진에서 보이는 것 (3) 비슷한 사례와 문헌 근거 (4) 사람이 꼭 확인할 것",
    prepare=prepare,
    cases=["lobar_pneumonia.jpg", "normal_pa.jpg", "heart_failure.jpg", "pneumothorax.png", "rib_fracture.jpg"],
    disclaimer="※ 교육·연구용 실습 결과입니다. AI가 쓴 사진 설명은 틀릴 수 있으며, 판독과 진단은 영상의학과 전문의가 합니다.",
)

# ---------- 증류에 쓸 질문 틀. {d} 자리에 SUBJECTS의 낱말이 들어간다 ----------
SUBJECTS = ["폐렴", "기흉", "흉수", "심비대", "폐결핵", "폐결절", "무기폐", "폐부종", "늑골 골절", "폐기종", "종격동 확장", "폐섬유화"]
TRAIN_TEMPLATES = [
    # 사진을 다시 봐야 하는 질문
    "이 사진에 {d} 소견이 보여?", "{d}로 볼 만한 음영이 어느 쪽 폐에 있어?", "사진에서 {d}가 의심되는 부위를 다시 확인해 줘",
    "{d}가 있다면 사진의 어디에 보여야 하는지 짚어 줘", "사진을 다시 보고 {d} 소견이 있는지만 알려 줘",
    "좌우 폐의 밝기가 달라 보이는데 다시 봐 줘", "이 사진은 제대로 찍힌 거야? 촬영 상태를 확인해 줘", "심장 크기가 커 보이는지 사진을 봐 줘",
    "갈비뼈 선이 끊긴 곳이 있는지 봐 줘", "횡격막 모양이 어떤지 설명해 줘", "이 사진에서 가장 눈에 띄는 곳이 어디야?",
    # 과거 사례를 찾는 질문
    "{d}로 판독된 비슷한 사례를 찾아 줘", "예전에 {d}로 판독된 사례가 사례 모음에 있어?", "{d} 사례 가운데 가장 가까운 것을 보여 줘",
    "이 사진과 비슷한 과거 사례가 있어?", "비슷한 사례들은 결국 무엇으로 판독됐어?", "사례 모음에서 가장 가까운 것 세 개만 보여 줘",
    # 논문 근거를 찾는 질문
    "{d}의 흉부 엑스레이 소견을 논문에서 찾아 줘", "{d}와 헷갈리기 쉬운 질환이 뭔지 문헌 근거를 줘", "{d}가 의심되면 어떤 추가 검사를 하는지 논문을 찾아 줘",
    "{d} 진단에서 엑스레이의 정확도는 어느 정도야?", "{d}에 CT가 꼭 필요한지 근거를 찾아 줘", "AI 판독 보조의 {d} 검출 성능을 다룬 연구가 있어?",
    # 두 명 이상이 필요한 질문
    "이 사진의 소견을 {d}로 판독된 과거 사례와 비교해 줘", "사진을 다시 보고, {d}의 논문 소견과 맞는지 확인해 줘", "{d}로 판독된 과거 사례와 논문 근거를 함께 정리해 줘",
    "사진 소견을 설명하고 비슷한 과거 사례도 찾아 줘", "사진을 다시 확인하고 관련 논문도 찾아 줘",
    "{d} 가능성을 사진, 과거 사례, 논문으로 종합해서 리포트를 써 줘", "이 사진을 종합적으로 검토해 줘", "판독 전에 참고할 내용을 전부 정리해 줘",
    "이 음영이 {d}인지 사진, 과거 사례, 문헌으로 교차 확인해 줘",
]
TEST_TEMPLATES = [      # 학습에 쓰지 않는 질문 틀
    "기관이 한쪽으로 밀려 있는지 사진을 확인해 줘", "{d}로 판독된 과거 사례 가운데 이것과 닮은 것이 있어?", "{d}의 초기 엑스레이 소견에 대한 최근 연구는?",
    "사진을 다시 보고 {d}로 판독된 과거 사례와 닮았는지 알려 줘", "{d}인지 아닌지 사진, 사례, 논문 근거를 모두 모아서 정리해 줘", "폐 꼭대기 부분에 이상이 있는지 사진을 봐 줘",
]
