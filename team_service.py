"""내가 만든 Multi-Agent 팀을 서비스로 띄운다.

웹 화면:   .venv/bin/python team_service.py xray_team          → 브라우저에서 http://localhost:8000
MCP 서버:  .venv/bin/python team_service.py xray_team --mcp    → Kiro나 Claude가 ask_team 도구로 부른다

첫 번째 인자는 팀이 들어 있는 파일 이름(.py 빼고)이다. 그 파일에 TEAM = Team(...)이 있어야 한다.
증류한 작은 모델(distill/<팀이름>/adapter)이 있으면 "전문가 고르기"를 그 모델에게 맡길 수 있다.
내 컴퓨터 안에서만 열린다(127.0.0.1). 교육·연구용이며 진료나 임상 판단을 대신하지 않는다.
"""
import importlib
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
args = [a for a in sys.argv[1:] if not a.startswith("--")]
if not args:
    sys.exit("사용법: .venv/bin/python team_service.py <팀 파일 이름> [--mcp] [--port 8000]\n예:     .venv/bin/python team_service.py xray_team")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
sys.path.insert(0, str(ROOT))
TEAM = importlib.import_module(args[0].removesuffix(".py")).TEAM

# 모델을 다루는 일은 한 줄로 세워 한 번에 하나씩 처리한다 (작은 모델은 불러온 그 thread에서만 쓸 수 있다)
worker = ThreadPoolExecutor(max_workers=1)
_router = {}


def run(question: str, case: str = "", router: str = "teacher") -> dict:
    def job():
        fn = None
        if router == "student" and TEAM.has_student():
            fn = _router.get("student") or _router.setdefault("student", TEAM.load_student())
        result = TEAM.ask(question, case=case, router=fn, quiet=True)
        result["router"] = "student" if fn else "teacher"
        return result
    return worker.submit(job).result()


def image_path(case: str):
    """검토 대상이 이 폴더 안의 사진 파일이면 그 경로를 돌려준다. 폴더 밖의 파일은 내주지 않는다."""
    candidates = [Path(case).expanduser()] + [d / case for d in (ROOT / "data").iterdir() if d.is_dir()]
    for p in candidates:
        p = p.resolve()
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"} and ROOT in p.parents:
            return p
    return None


# ---------- MCP 서버 ----------
if "--mcp" in sys.argv:
    from mcp.server.mcpserver import MCPServer

    server = MCPServer(f"{TEAM.name}-team")
    experts = ", ".join(v["title"] for v in TEAM.experts.values())
    case_help = f" case에는 검토 대상을 넣는다 (예: {', '.join(TEAM.cases[:3])})." if TEAM.cases else ""

    def ask_team(question: str, case: str = "") -> dict:
        r = run(question, case, "student" if TEAM.has_student() else "teacher")
        return {"answer": r["answer"], "router": r["router"], "seconds": r["seconds"],
                "experts": [{"title": x["title"], "tools": x["tools"], "answer": x["answer"]} for x in r["results"]]}

    ask_team.__doc__ = (f"{TEAM.goal}에게 질문한다. 내 컴퓨터의 전문가 Agent들({experts})이 각자 도구로 근거를 찾고 "
                        f"지휘자가 하나의 답으로 합친다. 1~2분 걸린다.{case_help}")
    server.tool()(ask_team)
    server.run()
    sys.exit(0)

# ---------- 웹 화면 ----------
import uvicorn
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.responses import FileResponse, HTMLResponse, JSONResponse
from starlette.routing import Route

PAGE = """<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Multi-Agent 팀</title><style>
:root{--blue:#1f4fd8;--teal:#0f9d8a;--ink:#16203a;--sub:#5b6680;--line:#dfe4ee;--soft:#f4f7fc}
*{box-sizing:border-box}body{margin:0;font-family:-apple-system,"Apple SD Gothic Neo","Pretendard","Noto Sans KR",sans-serif;color:var(--ink);background:#fff;line-height:1.6}
main{max-width:1080px;margin:0 auto;padding:28px 20px 60px}h1{font-size:26px;margin:0 0 4px}h1 span{color:var(--blue)}
.sub{color:var(--sub);margin:0 0 20px}.grid{display:grid;grid-template-columns:320px 1fr;gap:24px}@media(max-width:800px){.grid{grid-template-columns:1fr}}
label{display:block;font-weight:600;font-size:14px;margin:14px 0 6px}select,textarea,button{width:100%;font:inherit;border:1px solid var(--line);border-radius:8px;padding:10px 12px;background:#fff;color:var(--ink)}
textarea{min-height:92px;resize:vertical}button{background:var(--blue);color:#fff;border:0;font-weight:700;margin-top:16px;cursor:pointer}button:disabled{background:#9db0e6;cursor:wait}
#preview{width:100%;border-radius:8px;border:1px solid var(--line);margin-top:10px;display:none;background:#000}
.who{list-style:none;padding:0;margin:0}.who li{border-left:3px solid var(--line);padding:2px 0 2px 10px;margin:8px 0;font-size:14px;color:var(--sub)}.who b{color:var(--ink)}
.who li.on{border-color:var(--teal)}.who li.on b{color:var(--teal)}
.card{border:1px solid var(--line);border-radius:10px;padding:16px 18px;margin-bottom:14px}.card h3{margin:0 0 6px;font-size:16px}.card.final{border-color:var(--blue);background:var(--soft)}
.meta{font-size:13px;color:var(--sub);margin-bottom:8px}.tag{display:inline-block;background:#e3f5f2;color:#0b6f62;border-radius:4px;padding:1px 7px;font-size:12px;margin-right:4px}
pre{white-space:pre-wrap;word-break:break-word;font:inherit;margin:0}.note{font-size:13px;color:var(--sub);margin-top:22px}
</style></head><body><main>
<h1><span id="goal"></span></h1><p class="sub">전문가 Agent 여러 명이 각자 도구로 근거를 찾고, 지휘자가 하나의 답으로 합칩니다. 모두 이 컴퓨터 안에서 돕니다.</p>
<div class="grid"><section>
<div id="casebox"><label for="case">검토 대상</label><select id="case"></select><img id="preview" alt="검토 대상 사진"></div>
<label for="q">질문</label><textarea id="q">종합적으로 검토해 줘</textarea>
<label for="router">전문가를 고르는 모델</label><select id="router"></select>
<button id="go">질문하기</button>
<label>전문가</label><ul class="who" id="who"></ul>
</section><section id="out"><div class="card"><div class="meta">질문을 넣고 "질문하기"를 누르세요. 전문가가 많이 필요한 질문은 1~2분 걸립니다.</div></div></section></div>
<p class="note">교육·연구용 실습입니다. 진료나 임상 판단을 대신하지 않습니다.</p>
</main><script>
const $ = id => document.getElementById(id), esc = s => String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const md = s => esc(s).replace(/[*][*](.+?)[*][*]/g, '<b>$1</b>').replace(/^#{1,4} *(.+)$/gm, '<b>$1</b>').replace(/^ *[*-] +/gm, '• ');
let info;
async function init(){
  info = await (await fetch('info')).json();
  $('goal').textContent = info.goal; document.title = info.goal;
  $('who').innerHTML = info.experts.map(e => `<li data-k="${e.key}"><b>${esc(e.title)}</b><br>${esc(e.role)}</li>`).join('');
  if(info.cases.length){ $('case').innerHTML = info.cases.map(c => `<option>${esc(c)}</option>`).join(''); $('case').onchange = preview; preview(); }
  else $('casebox').style.display = 'none';
  $('router').innerHTML = `<option value="teacher">큰 모델 (${esc(info.teacher)})</option>` + (info.has_student ? '<option value="student" selected>증류한 작은 모델</option>' : '<option disabled>증류한 작은 모델 (아직 학습 안 함)</option>');
}
function preview(){ const img = $('preview'); img.onerror = () => img.style.display = 'none'; img.onload = () => img.style.display = 'block'; img.src = 'image?case=' + encodeURIComponent($('case').value); }
$('go').onclick = async () => {
  const btn = $('go'); btn.disabled = true; const t0 = Date.now();
  const tick = setInterval(() => btn.textContent = `전문가들이 일하는 중… ${Math.round((Date.now()-t0)/1000)}초`, 500);
  document.querySelectorAll('#who li').forEach(li => li.classList.remove('on'));
  try{
    const res = await fetch('ask', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({question: $('q').value, case: info.cases.length ? $('case').value : '', router: $('router').value})});
    const r = await res.json();
    if(r.error) throw new Error(r.error);
    r.experts.forEach(k => document.querySelector(`#who li[data-k="${k}"]`).classList.add('on'));
    $('out').innerHTML = `<div class="card final"><h3>통합 답변</h3><div class="meta">${r.seconds}초 · 전문가 ${r.experts.length}명 · 고른 모델: ${r.router === 'student' ? '증류한 작은 모델' : '큰 모델'} (${r.route_seconds}초)</div><pre>${md(r.answer)}</pre></div>`
      + r.results.map(x => `<div class="card"><h3>${esc(x.title)}</h3><div class="meta">${x.seconds}초 · 쓴 도구 ${x.tools.map(t => `<span class="tag">${esc(t)}</span>`).join('') || '없음'}</div><pre>${md(x.answer)}</pre></div>`).join('')
      + (r.context ? `<div class="card"><h3>모든 전문가에게 준 기초 자료</h3><pre>${esc(r.context)}</pre></div>` : '');
  }catch(e){ $('out').innerHTML = `<div class="card"><h3>실패했습니다</h3><pre>${esc(e.message)}\n\nOllama 앱이 켜져 있는지 확인하고 다시 눌러 주세요.</pre></div>`; }
  clearInterval(tick); btn.disabled = false; btn.textContent = '질문하기';
};
init();
</script></body></html>"""


async def home(request):
    return HTMLResponse(PAGE)


async def info(request):
    from agent_team import EXPERT_MODEL
    return JSONResponse({"name": TEAM.name, "goal": TEAM.goal, "cases": TEAM.cases, "teacher": EXPERT_MODEL, "has_student": TEAM.has_student(),
                         "experts": [{"key": k, "title": v["title"], "role": v["role"]} for k, v in TEAM.experts.items()]})


async def image(request):
    p = image_path(request.query_params.get("case", ""))
    return FileResponse(p) if p else JSONResponse({"error": "사진이 아닙니다"}, status_code=404)


async def ask(request):
    body = await request.json()
    question = str(body.get("question", "")).strip()
    if not question:
        return JSONResponse({"error": "질문이 비어 있습니다."})
    try:
        return JSONResponse(await run_in_threadpool(run, question, str(body.get("case", "")), body.get("router", "teacher")))
    except BaseException as e:      # 팀 코드의 SystemExit도 화면에 보여 준다
        return JSONResponse({"error": str(e)})


if __name__ == "__main__":
    port = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 8000
    print(f"✅ {TEAM.goal}: 브라우저에서 http://localhost:{port} 을 여세요. 끝낼 때는 Ctrl+C")
    app = Starlette(routes=[Route("/", home), Route("/info", info), Route("/image", image), Route("/ask", ask, methods=["POST"])])
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
