"""내가 만든 Multi-Agent 팀을 서비스로 띄운다.

웹 화면:   .venv/bin/python team_service.py xray_team          → 브라우저가 열린다 (http://127.0.0.1:8000, 이미 쓰는 중이면 다음 번호)
MCP 서버:  .venv/bin/python team_service.py xray_team --mcp    → Kiro나 Claude가 ask_team 도구로 부른다

첫 번째 인자는 팀이 들어 있는 파일 이름(.py 빼고)이다. 그 파일에 TEAM = Team(...)이 있어야 한다.
증류한 작은 모델(distill/<팀이름>/adapter)이 있으면 "전문가 고르기"를 그 모델에게 맡길 수 있다.
내 컴퓨터 안에서만 열린다(127.0.0.1). 교육·연구용이며 진료나 임상 판단을 대신하지 않는다.
"""
import importlib
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
args = [a for a in sys.argv[1:] if not a.startswith("--")]
if not args:
    sys.exit("사용법: .venv/bin/python team_service.py <팀 파일 이름> [--mcp] [--port 8000] [--no-open]\n예:     .venv/bin/python team_service.py xray_team")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
sys.path.insert(0, str(ROOT))
TEAM = importlib.import_module(args[0].removesuffix(".py")).TEAM

# 모델을 다루는 일은 한 줄로 세워 한 번에 하나씩 처리한다 (작은 모델은 불러온 그 thread에서만 쓸 수 있다)
worker = ThreadPoolExecutor(max_workers=1)
_router = {}
STARTED = int(time.time())     # 서버를 켠 시각. 사진 주소에 붙여서 브라우저가 예전에 받아 둔 사진을 다시 쓰지 않게 한다
progress = {"events": [], "draft": "", "start": 0.0}      # 지금 처리 중인 질문의 진행 소식. 웹 화면이 1초마다 읽어 간다


def on_event(event):
    if event["type"] == "draft":
        progress["draft"] = event["text"]       # 쓰고 있는 통합 답변은 가장 최근 것만 둔다
    else:
        progress["events"].append({**event, "at": round(time.time() - progress["start"])})


def run(question: str, case: str = "", router: str = "teacher", watch: bool = False) -> dict:
    def job():
        if watch:
            progress.update(events=[], draft="", start=time.time())
        fn = None
        if router == "student" and TEAM.has_student():
            fn = _router.get("student") or _router.setdefault("student", TEAM.load_student())
        result = TEAM.ask(question, case=case, router=fn, quiet=True, on_event=on_event if watch else None)
        result["router"] = "student" if fn else "teacher"
        return result
    return worker.submit(job).result()


def image_path(case: str, marked: bool = False):
    """검토 대상이 이 폴더 안의 사진 파일이면 그 경로를 돌려준다. 폴더 밖의 파일은 내주지 않는다.

    marked=True면 사람에게 보여 주는 설명용 사진(화살표를 그려 넣은 것, data/<폴더>/marked/)을 찾는다.
    """
    name = Path(case).name
    candidates = ([d / "marked" / name for d in (ROOT / "data").iterdir() if d.is_dir()] if marked else
                  [Path(case).expanduser()] + [d / case for d in (ROOT / "data").iterdir() if d.is_dir()])
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
#preview,#marked{width:100%;border-radius:8px;border:1px solid var(--line);margin-top:10px;display:none;background:#000}#marked{display:block}
.cap{margin:4px 0 2px}[hidden]{display:none!important}
.who{list-style:none;padding:0;margin:0}.who li{border-left:3px solid var(--line);padding:2px 0 2px 10px;margin:8px 0;font-size:14px;color:var(--sub)}.who b{color:var(--ink)}
.who li.on{border-color:var(--teal)}.who li.on b{color:var(--teal)}
.steps{list-style:none;margin:0;padding:0}.steps>li{padding:7px 0 7px 12px;border-left:3px solid var(--line);margin:6px 0;color:var(--sub)}
.steps>li.run{border-color:var(--blue)}.steps>li.run b{color:var(--blue)}.steps>li.done{border-color:var(--teal)}.steps>li.done b{color:var(--ink)}
.steps .n{display:inline-block;min-width:34px;font-weight:700;font-size:13px}.t{font-size:13px;color:var(--sub);font-weight:400}.steps .log{margin:4px 0 0 38px}
#compare{margin-top:8px;background:#fff;color:var(--blue);border:1px solid var(--blue)}#compare:disabled{color:#9db0e6;border-color:#9db0e6;background:#fff}
.help{font-size:13px;color:var(--sub);margin:6px 0 0}
.sizes{width:100%;border-collapse:collapse;font-size:12.5px;margin-top:10px}.sizes th,.sizes td{text-align:left;padding:4px 6px 4px 0;border-bottom:1px solid var(--line);vertical-align:top}.sizes th{color:var(--sub);font-weight:600}
.log{font-size:14px;color:var(--sub);margin:0;padding:0;list-style:none}.log li{padding:2px 0}.log li b{color:var(--ink)}
.who li.work{border-color:var(--blue)}.who li.work b{color:var(--blue)}
.card{border:1px solid var(--line);border-radius:10px;padding:16px 18px;margin-bottom:14px}.card h3{margin:0 0 6px;font-size:16px}.card.final{border-color:var(--blue);background:var(--soft)}
.meta{font-size:13px;color:var(--sub);margin-bottom:8px}.tag{display:inline-block;background:#e3f5f2;color:#0b6f62;border-radius:4px;padding:1px 7px;font-size:12px;margin-right:4px}
pre{white-space:pre-wrap;word-break:break-word;font:inherit;margin:0}.note{font-size:13px;color:var(--sub);margin-top:22px}
</style></head><body><main>
<h1><span id="goal"></span></h1><p class="sub">전문가 Agent 여러 명이 각자 도구로 근거를 찾고, 지휘자가 하나의 답으로 합칩니다. 모두 이 컴퓨터 안에서 돕니다.</p>
<div class="grid"><section>
<div id="casebox"><label for="case">검토 대상</label><select id="case"></select><div id="markedbox" hidden><img id="marked" alt="설명용 사진"><p class="help cap"><b>사람에게 보여 주는 사진.</b> 병변 위치에 화살표를 그려 넣은 것입니다.</p></div>
<img id="preview" alt="검토 대상 사진"><p class="help cap" id="previewcap"></p></div>
<label for="q">질문</label><textarea id="q">종합적으로 검토해 줘</textarea>
<label for="router">전문가를 고르는 모델</label><select id="router"></select>
<p class="help">질문을 보고 <b>어느 전문가를 부를지 정하는 일</b>(2단계)만 이 모델이 합니다. 전문가의 분석과 통합 답변은 어느 쪽을 골라도 큰 모델이 씁니다. 그래서 두 모델이 같은 전문가를 고르면 답의 질은 같고, 다르게 고르면 불리는 전문가가 달라집니다.</p>
<table class="sizes" id="sizes"></table>
<p class="help">GPT와 Claude는 크기를 공개하지 않아, 성능이 가깝다고 알려진 공개 모델로 대신 가늠했습니다. 전문가 고르기처럼 <b>답의 형태가 정해진 일</b>은 그런 큰 모델을 부르지 않아도, 이렇게 작은 모델을 직접 가르쳐 내 컴퓨터에서 돌릴 수 있습니다. 작은 모델은 배운 이 일 하나만 합니다.</p>
<button id="go">질문하기</button>
<button id="compare">두 모델이 고르는 전문가만 비교하기</button>
<label>전문가</label><ul class="who" id="who"></ul>
</section><section id="out"><div class="card"><div class="meta">질문을 넣고 "질문하기"를 누르세요. 전문가가 많이 필요한 질문은 1~2분 걸립니다. 일하는 동안 진행 상황이 여기에 나옵니다.</div></div></section></div>
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
  const z = info.sizes, n = p => p ? '약 ' + Math.round(parseFloat(p) * (p.endsWith('B') ? 10 : 100)) + (p.endsWith('B') ? '억 개' : '만 개') : '?', g = v => v == null ? '?' : v + 'GB';
  $('sizes').innerHTML = `<tr><th></th><th>파라미터</th><th>파일 크기</th><th>돌리는 곳</th></tr>`
    + `<tr><td><b>증류한 작은 모델</b><br>${esc(z.student.base)} + 내가 학습시킨 Adapter</td><td>${n(z.student.params)}</td><td>${g(z.student.file_gb)}${z.student.adapter_mb ? ' + ' + z.student.adapter_mb + 'MB' : ''}</td><td>내 컴퓨터 (메모리 1GB 안팎)</td></tr>`
    + `<tr><td><b>큰 모델</b><br>${esc(z.teacher.base)}</td><td>${n(z.teacher.params)}</td><td>${g(z.teacher.file_gb)}</td><td>내 컴퓨터 (메모리 약 10GB)</td></tr>`
    + `<tr><td><b>GPT·Claude급 공개 모델</b> (${esc(z.big.name)})</td><td>${esc(z.big.params)}</td><td>약 ${z.big.file_gb.toLocaleString()}GB</td><td>데이터센터의 GPU 서버 묶음 (80GB GPU 20장쯤)</td></tr>`;
  $('router').innerHTML = `<option value="teacher">큰 모델 (${esc(z.teacher.base)})</option>` + (info.has_student ? `<option value="student" selected>증류한 작은 모델 (${esc(z.student.base)} + Adapter)</option>` : '<option disabled>증류한 작은 모델 (아직 학습 안 함)</option>');
}
function preview(){ const c = $('case').value, img = $('preview'), has = info.marked.includes(c);
  img.onerror = () => { img.style.display = 'none'; $('previewcap').textContent = ''; };
  img.onload = () => { img.style.display = 'block'; $('previewcap').innerHTML = has ? '<b>모델이 보는 사진.</b> 화살표가 없는 원본입니다. 모델은 이것만 봅니다.' : '<b>모델이 보는 사진.</b> 이 사진 그대로 봅니다.'; };
  img.src = 'image?v=' + info.started + '&case=' + encodeURIComponent(c);
  $('markedbox').hidden = !has; if(has) $('marked').src = 'image?v=' + info.started + '&marked=1&case=' + encodeURIComponent(c); }
const card = x => `<div class="card"><h3>${esc(x.title)}</h3><div class="meta">${x.seconds}초 · 쓴 도구 ${x.tools.map(t => `<span class="tag">${esc(t)}</span>`).join('') || '없음'}</div><pre>${md(x.answer)}</pre></div>`;
const title = k => (info.experts.find(e => e.key === k) || {}).title || k;
function live(events, draft, now){      // 일하는 동안 보여 주는 화면: 단계별 진행, 끝난 전문가의 답, 쓰고 있는 통합 답변
  const names = (events.find(e => e.type === 'steps') || {names: []}).names, steps = names.map((name, i) => ({n: i + 1, name, state: 'wait', sub: []}));
  const done = []; let cur = null;
  for(const e of events){
    if(e.type === 'step'){ cur = steps[e.n - 1]; cur.state = 'run'; cur.from = e.at; }
    if(e.type === 'step_done'){ const st = steps[e.n - 1]; st.state = 'done'; st.seconds = e.seconds; st.note = e.note; }
    if(e.type === 'route') e.experts.forEach(k => document.querySelector(`#who li[data-k="${k}"]`).classList.add('work'));
    if(e.type === 'tool' && cur) cur.sub.push(`🔧 ${esc(e.title)}: ${esc(e.name)}(${esc(e.args)})  (${e.at}초)`);
    if(e.type === 'expert'){ if(cur) cur.sub.push(`✅ <b>${esc(e.result.title)}</b> 끝  (${e.result.seconds}초)`); done.push(e.result);
      const li = document.querySelector(`#who li[data-k="${e.result.expert}"]`); li.classList.remove('work'); li.classList.add('on'); }
  }
  const row = st => `<li class="${st.state}"><span class="n">${st.n}/${steps.length}</span> <b>${esc(st.name)}</b> `
    + (st.state === 'done' ? `<span class="t">끝 · ${st.seconds}초${st.note ? ' · ' + esc(st.note) : ''}</span>`
       : st.state === 'run' ? `<span class="t">진행 중 · ${Math.max(0, now - st.from)}초째</span>` : '<span class="t">기다리는 중</span>')
    + (st.sub.length ? `<ul class="log">${st.sub.map(l => `<li>${l}</li>`).join('')}</ul>` : '') + '</li>';
  $('out').innerHTML = `<div class="card"><h3>진행 상황 <span class="t">전체 ${now}초째</span></h3><ol class="steps">${steps.map(row).join('') || '<li class="run">시작하는 중…</li>'}</ol></div>`
    + (draft ? `<div class="card final"><h3>통합 답변 (쓰는 중)</h3><pre>${md(draft)} ▌</pre></div>` : '') + done.map(card).join('');
}
$('go').onclick = async () => {
  const btn = $('go'); btn.disabled = true; const t0 = Date.now();
  const tick = setInterval(() => btn.textContent = `전문가들이 일하는 중… ${Math.round((Date.now()-t0)/1000)}초`, 500);
  document.querySelectorAll('#who li').forEach(li => li.classList.remove('on', 'work'));
  const events = []; let watching = true, draft = ''; const sec = () => Math.round((Date.now() - t0) / 1000); live(events, draft, 0);
  const watch = (async () => { while(watching){ await new Promise(r => setTimeout(r, 800)); if(!watching) break;
    try{ const p = await (await fetch('progress?since=' + events.length)).json(); events.push(...p.events); draft = p.draft; if(watching) live(events, draft, sec()); }catch(e){} } })();
  try{
    const res = await fetch('ask', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({question: $('q').value, case: info.cases.length ? $('case').value : '', router: $('router').value})});
    const r = await res.json(); watching = false;
    if(r.error) throw new Error(r.error);
    document.querySelectorAll('#who li').forEach(li => li.classList.remove('work'));
    r.experts.forEach(k => document.querySelector(`#who li[data-k="${k}"]`).classList.add('on'));
    $('out').innerHTML = `<div class="card final"><h3>통합 답변</h3><div class="meta">${r.seconds}초 · 전문가 ${r.experts.length}명 · 고른 모델: ${r.router === 'student' ? '증류한 작은 모델' : '큰 모델'} (${r.route_seconds}초)</div><pre>${md(r.answer)}</pre></div>`
      + `<div class="card"><h3>단계별로 걸린 시간</h3><ol class="steps">${r.steps.map(st => `<li class="done"><span class="n">${st.n}/${r.steps.length}</span> <b>${esc(st.name)}</b> <span class="t">${st.seconds}초${st.note ? ' · ' + esc(st.note) : ''}</span></li>`).join('')}</ol></div>`
      + r.results.map(card).join('')
      + (r.context ? `<div class="card"><h3>모든 전문가에게 준 기초 자료</h3><pre>${esc(r.context)}</pre></div>` : '');
  }catch(e){ $('out').innerHTML = `<div class="card"><h3>실패했습니다</h3><pre>${esc(e.message)}\n\nOllama 앱이 켜져 있는지 확인하고 다시 눌러 주세요.</pre></div>`; }
  watching = false; clearInterval(tick); btn.disabled = false; btn.textContent = '질문하기';
};
$('compare').onclick = async () => {     // 전문가는 부르지 않고, 두 모델이 누구를 고르는지와 걸린 시간만 본다
  const btn = $('compare'); btn.disabled = true; btn.textContent = '두 모델에게 물어보는 중…';
  try{
    const r = await (await fetch('route', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({question: $('q').value})})).json();
    if(r.error) throw new Error(r.error);
    const line = (name, x) => x ? `<li class="done"><b>${name}</b> <span class="t">${x.seconds}초</span><ul class="log"><li>${x.experts.map(k => esc(title(k))).join(', ')}</li></ul></li>` : `<li><b>${name}</b> <span class="t">아직 학습하지 않았습니다 (노트북 2-4)</span></li>`;
    $('out').innerHTML = `<div class="card final"><h3>"${esc(r.question)}"에 누구를 부를까</h3><ol class="steps">${line('큰 모델 (' + esc(info.teacher) + ')', r.teacher)}${line('증류한 작은 모델', r.student)}</ol>`
      + (r.student ? `<p class="help">${r.same ? '⭕ 두 모델이 같은 전문가를 골랐습니다. 이 질문은 어느 모델로 골라도 같은 전문가가 일합니다.' : '❌ 두 모델이 다르게 골랐습니다. 작은 모델이 배우지 못한 말투이거나 스승도 헷갈리는 질문입니다. 말투를 바꿔 가며 어디서 달라지는지 찾아보세요.'}</p>` : '') + '</div>';
  }catch(e){ $('out').innerHTML = `<div class="card"><h3>실패했습니다</h3><pre>${esc(e.message)}</pre></div>`; }
  btn.disabled = false; btn.textContent = '두 모델이 고르는 전문가만 비교하기';
};
init();
</script></body></html>"""


async def home(request):
    return HTMLResponse(PAGE)


async def info(request):
    from agent_team import EXPERT_MODEL
    return JSONResponse({"name": TEAM.name, "goal": TEAM.goal, "cases": TEAM.cases, "teacher": EXPERT_MODEL, "has_student": TEAM.has_student(), "started": STARTED, "sizes": TEAM.sizes(),
                         "marked": [c for c in TEAM.cases if image_path(c, True)],      # 설명용(화살표) 사진이 따로 있는 검토 대상
                         "experts": [{"key": k, "title": v["title"], "role": v["role"]} for k, v in TEAM.experts.items()]})


async def image(request):
    p = image_path(request.query_params.get("case", ""), request.query_params.get("marked") == "1")
    # no-cache: 사진 파일을 바꿨는데 브라우저가 예전 사진을 계속 보여 주는 일을 막는다
    return FileResponse(p, headers={"Cache-Control": "no-cache"}) if p else JSONResponse({"error": "사진이 아닙니다"}, status_code=404)


def compare_routers(question: str) -> dict:
    """전문가를 부르지 않고, 큰 모델과 증류한 작은 모델이 각각 누구를 고르는지와 걸린 시간만 잰다."""
    def job():
        TEAM.route_with_teacher(question)       # 큰 모델을 미리 메모리에 올려 두고 잰다
        t0 = time.time(); a = TEAM.route_with_teacher(question); t1 = time.time()
        out = {"question": question, "teacher": {"experts": a, "seconds": round(t1 - t0, 2)}, "student": None, "same": None}
        if TEAM.has_student():
            fn = _router.get("student") or _router.setdefault("student", TEAM.load_student())
            t2 = time.time(); b = fn(question); t3 = time.time()
            out.update(student={"experts": b, "seconds": round(t3 - t2, 2)}, same=a == b)
        return out
    return worker.submit(job).result()


async def route_view(request):
    question = str((await request.json()).get("question", "")).strip()
    if not question:
        return JSONResponse({"error": "질문이 비어 있습니다."})
    try:
        return JSONResponse(await run_in_threadpool(compare_routers, question))
    except BaseException as e:
        return JSONResponse({"error": str(e)})


async def progress_view(request):
    """지금 처리 중인 질문의 진행 소식. since번째 뒤의 소식과, 쓰고 있는 통합 답변을 돌려준다."""
    since = int(request.query_params.get("since", 0))
    return JSONResponse({"events": progress["events"][since:], "draft": progress["draft"]})


async def ask(request):
    body = await request.json()
    question = str(body.get("question", "")).strip()
    if not question:
        return JSONResponse({"error": "질문이 비어 있습니다."})
    try:
        return JSONResponse(await run_in_threadpool(run, question, str(body.get("case", "")), body.get("router", "teacher"), True))
    except BaseException as e:      # 팀 코드의 SystemExit도 화면에 보여 준다
        return JSONResponse({"error": str(e)})


if __name__ == "__main__":
    port = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 8000
    from lab_ui import in_use
    first = port
    while in_use(port):         # 다른 프로그램이 쓰는 번호면 다음 번호로 옮긴다 (그대로 열면 그 프로그램의 화면이 뜬다)
        port += 1
    if port != first:
        print(f"⚠️  {first}번은 다른 프로그램이 쓰고 있어서 {port}번으로 엽니다.")
    url = f"http://127.0.0.1:{port}"
    print(f"✅ {TEAM.goal}: 브라우저에서 {url} 을 여세요. 끝낼 때는 Ctrl+C")
    if "--no-open" not in sys.argv:
        import threading, webbrowser
        threading.Timer(1.0, webbrowser.open, [url]).start()     # 서버가 뜬 뒤 브라우저를 자동으로 연다
    app = Starlette(routes=[Route("/", home), Route("/info", info), Route("/image", image), Route("/ask", ask, methods=["POST"]),
                            Route("/progress", progress_view), Route("/route", route_view, methods=["POST"])])
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
