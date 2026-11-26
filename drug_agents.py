"""노트북 2-2에서 만든 Multi-Agent(전문가 3명 + 지휘자)를 다른 곳에서 다시 쓰기 위해 모아 둔 파일.

노트북 2-6에서 "전문가 고르기"만 증류한 작은 모델로 바꿔 끼울 때 쓴다.
    from drug_agents import consult
    consult("신장이 나쁜 환자에게 써도 될까?", drug="ibuprofen", router=route_with_student)
교육·연구용이며 임상 판단을 대신하지 않는다.
"""
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import ollama
from strands import Agent, tool
from strands.models.ollama import OllamaModel

import drug_tools as dt

EXPERT_MODEL = os.getenv("EXPERT_MODEL", "gemma4:e4b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")


@tool
def compound_info(name: str) -> dict:
    """화합물의 영문 이름으로 PubChem에서 기본 정보와 SMILES 구조식을 찾는다."""
    return dt.compound_info(name)


@tool
def molecule_properties(smiles: str) -> dict:
    """SMILES 구조식으로 분자량, 지용성(logP), 수소결합 수, Lipinski 규칙 통과 여부를 계산한다."""
    return dt.molecule_properties(smiles)


@tool
def similar_compounds(smiles: str) -> dict:
    """SMILES 구조식과 구조가 비슷한 화합물을 PubChem에서 찾는다."""
    return dt.similar_compounds(smiles, 5)


@tool
def pubmed_search(query: str) -> dict:
    """PubMed에서 논문을 검색한다. 검색어는 영문으로 쓴다 (예: "ibuprofen cardiotoxicity")."""
    return dt.pubmed_search(query, 5)


@tool
def adverse_events(drug: str) -> dict:
    """openFDA에 가장 많이 보고된 부작용을 찾는다. 약의 영문 성분명을 넣는다."""
    return dt.adverse_events(drug, 8)


@tool
def clinical_trials(query: str) -> dict:
    """ClinicalTrials.gov에서 임상시험을 검색한다. 약 이름이나 질환을 영문으로 넣는다."""
    return dt.clinical_trials(query, 5)


COMMON = """
- 반드시 도구를 써서 근거를 찾는다. 도구 결과에 없는 숫자와 사실은 쓰지 않는다.
- 한국어로, 소제목 3개 이내, 전체 10줄 이내로 답한다.
- 마지막 줄에 확신 정도(높음, 보통, 낮음)를 적는다."""

EXPERTS = {
    "chemist": {
        "title": "의약화학 전문가",
        "prompt": "너는 15년 경력의 의약화학 전문가다. 화합물의 구조, 물리화학적 특성, 구조가 비슷한 화합물을 보고 구조를 어떻게 고치면 좋을지 판단한다." + COMMON,
        "tools": [compound_info, molecule_properties, similar_compounds],
    },
    "pharmacologist": {
        "title": "임상약리 전문가",
        "prompt": "너는 20년 경력의 임상약리 전문가다. 논문과 부작용 보고를 근거로 안전성, 독성, 약물 상호작용, 용량을 판단한다." + COMMON,
        "tools": [pubmed_search, adverse_events],
    },
    "regulatory": {
        "title": "임상·규제 전략 전문가",
        "prompt": "너는 신약 허가 전략 전문가다. 비슷한 약의 임상시험 현황을 근거로 임상 설계, 허가 경로, 경쟁 상황을 판단한다." + COMMON,
        "tools": [clinical_trials, pubmed_search],
    },
}


def base_context(drug: str) -> str:
    """모든 전문가에게 똑같이 주는 기초 자료."""
    info = dt.compound_info(drug)
    if not info.get("found"):
        raise ValueError(f"PubChem에서 '{drug}'를 찾지 못했습니다. 영문 이름을 확인하세요.")
    p = dt.molecule_properties(info["smiles"])
    return (f"이름: {drug}\nSMILES: {info['smiles']}\n분자식: {info['formula']}\n"
            f"분자량: {p['molecular_weight']}, logP: {p['logp']}, 수소결합 주개/받개: {p['h_bond_donors']}/{p['h_bond_acceptors']}, "
            f"TPSA: {p['tpsa']}, QED: {p['qed']}, Lipinski 통과: {p['lipinski_pass']}")


def ask_expert(key: str, question: str, context: str) -> dict:
    """전문가 한 명에게 묻는다. 매번 새 Agent를 만들어 다른 질문의 기억이 섞이지 않게 한다."""
    spec = EXPERTS[key]
    agent = Agent(model=OllamaModel(host=OLLAMA_HOST, model_id=EXPERT_MODEL, temperature=0.2),
                  system_prompt=spec["prompt"], tools=spec["tools"], callback_handler=None)
    start = time.time()
    used = lambda: [b["toolUse"]["name"] for m in agent.messages for b in m.get("content", []) if "toolUse" in b]
    try:
        text = str(agent(f"[화합물 기본 정보]\n{context}\n\n[질문]\n{question}")).strip()
        if not used():      # 도구 없이 기억만으로 답했으면 한 번 더 시킨다
            text = str(agent("도구를 한 번도 쓰지 않았다. 먼저 도구를 불러 근거를 찾고, 그 결과만으로 다시 답하라.")).strip()
    except Exception as e:
        text = f"(분석 실패: {e})"
    return {"expert": key, "title": spec["title"], "answer": text, "tools": used(), "seconds": round(time.time() - start, 1)}


def run_experts(keys: list, question: str, context: str, say=None) -> list:
    """고른 전문가들에게 동시에 일을 맡긴다. say에 print를 넘기면 한 명이 끝날 때마다 알려 준다."""
    say = say or (lambda *a: None)
    say(f"⏳ 전문가 {len(keys)}명이 동시에 일하는 중입니다. 한 명에 30초~1분쯤 걸립니다.")
    start = time.time()
    with ThreadPoolExecutor(max_workers=len(keys)) as pool:
        jobs = {pool.submit(ask_expert, k, question, context): k for k in keys}
        for n, job in enumerate(as_completed(jobs), 1):
            say(f"   ✅ {n}/{len(keys)} {EXPERTS[jobs[job]]['title']} 끝  ({time.time() - start:.0f}초)")
        return [job.result() for job in jobs]


def synthesize(question: str, context: str, results: list) -> str:
    """지휘자가 전문가들의 답을 모아 하나의 리포트로 쓴다."""
    parts = "\n\n".join(f"### {r['title']}의 분석\n{r['answer']}" for r in results)
    prompt = f"""너는 신약 개발팀의 지휘자다. 아래 전문가 분석만 근거로 한국어 통합 리포트를 쓴다.
형식: (1) 한 줄 결론 (2) 근거 3개 이내 (3) 위험 요소 (4) 다음에 할 일. 전문가 분석에 없는 내용은 쓰지 않는다. 전체 12줄 이내.

[화합물 기본 정보]
{context}

[질문]
{question}

{parts}"""
    r = ollama.chat(model=EXPERT_MODEL, messages=[{"role": "user", "content": prompt}], options={"temperature": 0.2})
    return r["message"]["content"].strip()


def consult(question: str, drug: str, router) -> str:
    """질문 하나를 처음부터 끝까지 처리한다. router는 질문을 받아 전문가 이름 목록을 돌려주는 함수."""
    start = time.time()
    context = base_context(drug)
    t0 = time.time()
    keys = router(question)
    print(f"👤 질문: {question}")
    print(f"🧭 고른 전문가: {', '.join(EXPERTS[k]['title'] for k in keys)}  (고르는 데 {time.time() - t0:.2f}초)")
    results = run_experts(keys, question, context, say=print)
    for r in results:
        print(f"\n🧑‍🔬 {r['title']}  ({r['seconds']}초, 도구: {' → '.join(r['tools']) or '안 씀'})\n{r['answer']}")
    if len(results) > 1:
        print("\n⏳ 지휘자가 전문가들의 답을 하나로 합치는 중입니다. 20~40초 걸립니다.")
    answer = synthesize(question, context, results) if len(results) > 1 else results[0]["answer"]
    print("\n📋 통합 답변\n" + answer)
    print(f"\n⏱️ {time.time() - start:.0f}초, 전문가 {len(keys)}명")
    return answer
