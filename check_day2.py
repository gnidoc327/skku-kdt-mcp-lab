"""2일차 설치 확인: 패키지, 큰 모델(사진 보기 포함), 증류용 작은 모델이 이 컴퓨터에서 도는지 본다. 1분쯤 걸린다."""
import os
import sys
import time
import warnings

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")
failed = []


def step(name, fn, hint):
    start = time.time()
    try:
        print(f"✅ {name}: {fn()}  ({time.time() - start:.0f}초)")
    except BaseException as e:
        failed.append(name)
        print(f"❌ {name}: {str(e)[:200]}\n   → {hint}")


def packages():
    import mlx_lm, rdkit, strands  # noqa: F401
    return "strands-agents, rdkit, mlx-lm"


def big_model():
    import xray_team
    text = xray_team.prepare("normal_pa.jpg").splitlines()[-1]
    return f"{xray_team.VISION_MODEL}가 사진을 보고 답함: {text[:40]}…"


def case_search():
    import xray_team
    xray_team.CURRENT["path"] = None
    top = xray_team.search_similar_cases("갈비뼈가 부러진 사진")[0]
    if top["label"] != "늑골 골절":
        raise RuntimeError(f"검색 결과가 예상과 다릅니다: {top['label']}")
    return f"'갈비뼈가 부러진 사진' → {top['label']}"


def small_model():
    from mlx_lm import generate, load
    model, tok = load(os.getenv("STUDENT_MODEL", "mlx-community/gemma-3-1b-it-4bit"))
    prompt = tok.apply_chat_template([{"role": "user", "content": "1+1은? 숫자만 답해."}], add_generation_prompt=True, tokenize=False)
    return "증류용 작은 모델이 답함: " + generate(model, tok, prompt=prompt, max_tokens=8, verbose=False).split("<")[0].strip()[:20]


step("2일차 패키지", packages, "'bash setup.sh'를 다시 실행하세요.")
step("큰 모델", big_model, "Ollama 앱이 켜져 있는지, 'ollama pull gemma4:e4b'가 끝났는지 확인하세요. 메모리가 부족하면 다른 앱을 닫습니다.")
step("사례 검색", case_search, "'ollama pull bge-m3'를 했는지 확인하세요.")
step("작은 모델", small_model, "인터넷 연결을 확인하고 'bash setup.sh'를 다시 실행하세요.")
sys.exit(1 if failed else 0)
