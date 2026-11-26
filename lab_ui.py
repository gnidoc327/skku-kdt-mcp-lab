"""실습 화면 도우미: 오래 걸리는 일이 도는 동안 진행 상황을 보여 주고, 답을 보기 좋게 그린다.

Agent나 증류의 원리와는 상관없는, 화면에 보여 주는 일만 모아 둔 파일이다.
    Status        같은 줄을 고쳐 쓰는 진행 표시 ("⏳ 37초째 일하는 중")
    tool_log      Agent가 도구를 부를 때마다 한 줄 알려 주게 한다
    wait_for      오래 걸리는 일 하나를 돌리면서 몇 초째인지 보여 준다
    run_together  여러 일을 동시에 돌리면서 누가 끝났는지 보여 준다
    stream_answer 모델의 답을 쓰는 대로 화면에 그린다
    md            마크다운을 그려서 보여 준다 (굵은 글씨, 목록)
노트북(JupyterLab)에서는 줄을 고쳐 쓰고 마크다운을 그린다. 터미널에서는 평범하게 한 줄씩 찍는다.
"""
import logging
import queue
import time
from concurrent.futures import ThreadPoolExecutor

logging.getLogger("strands").setLevel(logging.ERROR)     # "reasoningContent is not supported" 같은 알림 줄을 숨긴다
USAGE = []      # 이번 질문을 처리하며 쓴 token을 적어 둔다: {"who", "in", "out", "calls", "live"}
                # live는 모델이 지금 쓰고 있는 글의 token을 어림으로 센 값이다. 그 호출이 끝나면 정확한 값으로 바뀐다


def count_tokens(who, response=None, tokens_in=None, tokens_out=None):
    """모델을 한 번 부른 뒤 쓴 token을 적어 둔다. who는 누가 불렀는지(전문가 이름, 지휘자 등).

    response에는 ollama.chat의 응답을 그대로 넘기면 된다.
    """
    try:
        if tokens_in is None and response is not None:
            tokens_in, tokens_out = response.prompt_eval_count or 0, response.eval_count or 0
        USAGE.append({"who": who, "in": int(tokens_in or 0), "out": int(tokens_out or 0), "calls": 1, "live": 0})
    except Exception:
        pass                                            # 세지 못해도 하던 일은 계속한다


def watch_tokens(agent, who):
    """Strands Agent가 쓰는 token을 실시간으로 센다.

    모델이 글이나 도구 호출을 쓰는 동안에는 조각 수로 어림하고, 모델 호출 하나가 끝날 때마다 그 호출의 정확한 값을 더한다.
    일이 다 끝난 뒤 settle_tokens(agent)를 부르면 Agent가 집계한 값으로 한 번 더 맞춘다.
    """
    row = {"who": who, "in": 0, "out": 0, "calls": 0, "live": 0}
    USAGE.append(row)

    def on_stream(**kw):
        event = kw.get("event")
        usage = event.get("metadata", {}).get("usage") if isinstance(event, dict) and "metadata" in event else None
        if usage:                                       # 모델 호출 하나가 끝났다: 그 호출이 쓴 정확한 값
            row["in"] += usage.get("inputTokens", 0)
            row["out"] += usage.get("outputTokens", 0)
            row["calls"] += 1
            row["live"] = 0
        elif kw.get("data") or (isinstance(event, dict) and "contentBlockDelta" in event):
            row["live"] += 1                            # 글이나 도구 호출의 조각 하나가 대략 token 하나다
    agent.callback_handler = on_stream
    agent._token_row = row
    return agent


def settle_tokens(agent):
    """Agent가 일을 마친 뒤, Agent가 스스로 집계한 token 수로 맞춘다 (실시간으로 센 값이 빠졌을 때를 대비한다)."""
    row, usage = getattr(agent, "_token_row", None), agent.event_loop_metrics.accumulated_usage
    if row is not None and usage.get("inputTokens", 0) >= row["in"]:
        row.update({"in": usage.get("inputTokens", 0), "out": usage.get("outputTokens", 0), "live": 0})
        row["calls"] = max(row["calls"], 1)


def chat_counted(who, think=None, **kw):
    """ollama.chat을 부르되, 글을 쓰는 동안 token을 실시간으로 센다. 다 쓴 글을 돌려준다."""
    import ollama
    row = {"who": who, "in": 0, "out": 0, "calls": 1, "live": 0}
    USAGE.append(row)
    text = ""
    for chunk in (ollama.chat(stream=True, think=think, **kw) if think is not None else ollama.chat(stream=True, **kw)):
        text += chunk.message.content or ""
        row["live"] += 1
        if chunk.done:
            row.update({"in": chunk.prompt_eval_count or 0, "out": chunk.eval_count or 0, "live": 0})
    return text.strip()


def usage_summary():
    """지금까지 쓴 token을 누가 썼는지별로 더한다. 쓰고 있는 중인 글도 어림으로 넣는다."""
    by = {}
    for u in list(USAGE):
        row = by.setdefault(u["who"], {"who": u["who"], "in": 0, "out": 0, "calls": 0})
        row["in"] += u["in"]; row["out"] += u["out"] + u.get("live", 0); row["calls"] += u.get("calls", 1)
    rows = list(by.values())
    return {"by": rows, "total": {"in": sum(r["in"] for r in rows), "out": sum(r["out"] for r in rows), "calls": sum(r["calls"] for r in rows)}}


_events = queue.Queue()      # 다른 thread에서 생긴 소식을 모아 두었다가, 화면에는 한곳에서만 찍는다


def in_notebook():
    try:
        from IPython import get_ipython
        return get_ipython() is not None and "IPKernelApp" in get_ipython().config
    except Exception:
        return False


def _clean(text):
    """마크다운으로 그릴 때 물결표(취소선)와 달러 기호(수식)가 글을 망치지 않게 한다."""
    return text.replace("\\~", "~").replace("~", "\\~").replace("$", "\\$")


def md(text):
    """마크다운을 그려서 보여 준다. 터미널에서는 그대로 찍는다."""
    if in_notebook():
        from IPython.display import Markdown, display
        display(Markdown(_clean(text)))
    else:
        print(text)


class Status:
    """한 줄짜리 진행 표시. 노트북에서는 같은 줄을 고쳐 쓴다. 터미널에서는 끝났을 때만 한 줄 찍는다.

    글 안에 "12/420"처럼 "한 것/전체"가 들어 있으면 진행 막대와 남은 시간을 함께 그린다.
    """

    def __init__(self, text="⏳ 시작합니다"):
        self.start, self.handle = time.time(), None
        if in_notebook():
            from IPython.display import display
            self.handle = display(self._bundle(text), raw=True, display_id=True)
        else:
            print(text)

    def seconds(self):
        return time.time() - self.start

    def _bundle(self, text, finished=False):
        """화면에 그릴 것을 만든다: 진행 막대가 있는 HTML과, HTML을 못 그리는 곳을 위한 글."""
        import html
        import re
        m = None if finished else re.search(r"(\d[\d,]*)/(\d[\d,]*)", text)
        bar = ""
        if m:
            done, total = int(m[1].replace(",", "")), int(m[2].replace(",", ""))
            if 0 <= done <= total and total > 0:
                left = f" · 남은 시간 약 {self.seconds() * (total - done) / done:.0f}초" if 0 < done < total and self.seconds() > 2 else ""
                bar = (f'<progress value="{done}" max="{total}" style="width:240px;height:14px;vertical-align:middle"></progress> '
                       f'<b>{done / total:.0%}</b>{left}<br>')
        return {"text/plain": text, "text/html": f'<div style="font-family:var(--jp-code-font-family,monospace);font-size:13px;line-height:1.7">{bar}{html.escape(text)}</div>'}

    def update(self, text):
        if self.handle:
            self.handle.update(self._bundle(text), raw=True)

    def done(self, text):
        self.handle.update(self._bundle(text, finished=True), raw=True) if self.handle else print(text)


def note(text):
    """어느 thread에서든 부를 수 있다. 소식 한 줄을 남기면 기다리는 쪽이 화면에 찍는다."""
    _events.put(text)


def _print_notes():
    while not _events.empty():
        print(_events.get())


def tool_log(agent, who):
    """Agent가 도구를 부를 때마다 "누가 어떤 도구를 무엇으로 불렀는지" 한 줄 남기게 한다."""
    from strands.hooks import BeforeToolCallEvent

    def on_tool(event):
        args = ", ".join(f"{k}={str(v)[:50]!r}" for k, v in (event.tool_use.get("input") or {}).items())
        note(f"   🔧 {who}: {event.tool_use.get('name')}({args})")
    agent.hooks.add_callback(BeforeToolCallEvent, on_tool)
    return agent


def run_together(jobs, on_done=None):
    """여러 일을 동시에 돌린다. jobs는 {이름: 함수}. 결과를 맡긴 순서대로 돌려준다.

    도는 동안 몇 초째인지, 누가 아직 일하는지 보여 주고, 도구를 부르거나 한 명이 끝나면 한 줄씩 찍는다.
    on_done(이름, 결과)를 주면 한 명이 끝날 때마다 불러 준다 (끝난 사람의 답을 바로 보여 줄 때 쓴다).
    """
    names = list(jobs)
    status = Status(f"⏳ {', '.join(names)}: 일을 시작합니다")
    with ThreadPoolExecutor(max_workers=len(names)) as pool:
        futures = {n: pool.submit(f) for n, f in jobs.items()}
        left = list(names)
        while left:
            _print_notes()
            for n in [n for n in left if futures[n].done()]:
                left.remove(n)
                _print_notes()
                if len(names) > 1:
                    print(f"   ✅ {len(names) - len(left)}/{len(names)} {n} 끝  ({status.seconds():.0f}초)")
                if on_done and futures[n].exception() is None:
                    on_done(n, futures[n].result())
            if left:
                status.update(f"⏳ {status.seconds():.0f}초째 일하는 중: {', '.join(left)}" + (f"  (끝난 사람 {len(names) - len(left)}/{len(names)})" if len(names) > 1 else ""))
                time.sleep(0.5)
    _print_notes()
    status.done(f"✅ {', '.join(names)}: 끝  ({status.seconds():.0f}초)")
    return [futures[n].result() for n in names]


def wait_for(name, job):
    """오래 걸리는 일 하나를 돌리면서 몇 초째인지 보여 준다. 그 일의 결과를 돌려준다."""
    return run_together({name: job})[0]


def stream_answer(model, messages, who="지휘자", temperature=0.2):
    """모델에게 묻고, 답을 쓰는 대로 화면에 그린다. 다 쓴 답(글)을 돌려준다."""
    import ollama
    status = Status(f"⏳ {who}가 답을 준비하는 중입니다")
    if not in_notebook():
        r = ollama.chat(model=model, messages=messages, options={"temperature": temperature})
        count_tokens(f"{who} (답 합치기)", r)
        status.done(f"✅ {who}가 답을 썼습니다  ({status.seconds():.0f}초)")
        return r["message"]["content"].strip()
    from IPython.display import Markdown, display
    box, text, shown = None, "", 0.0
    for chunk in ollama.chat(model=model, messages=messages, options={"temperature": temperature}, stream=True):
        text += chunk.message.content or ""
        if chunk.done:
            count_tokens(f"{who} (답 합치기)", chunk)
        if not text.strip():        # 답을 쓰기 전에 속으로 생각하는 모델도 있다
            status.update(f"🤔 {who}가 생각하는 중입니다  ({status.seconds():.0f}초)")
            continue
        status.update(f"✍️ {who}가 답을 쓰는 중입니다  ({status.seconds():.0f}초)")
        if time.time() - shown > 0.25:
            box = box or display(Markdown(""), display_id=True)
            box.update(Markdown(_clean(text) + " ▌"))
            shown = time.time()
    box = box or display(Markdown(""), display_id=True)
    box.update(Markdown(_clean(text.strip())))
    status.done(f"✅ {who}가 답을 썼습니다  ({status.seconds():.0f}초)")
    return text.strip()


# ---------- 팀을 웹 화면으로 띄우기, Kiro에 연결하기 (노트북의 셀에서 부른다) ----------
import os
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PID_FILE, LOG_FILE = ROOT / ".service.pid", ROOT / ".service.log"


def in_use(port):
    """이 번호를 다른 프로그램이 이미 쓰고 있는지 본다 (localhost로 열리는 두 주소 모두)."""
    import socket
    for family, host in ((socket.AF_INET, "127.0.0.1"), (socket.AF_INET6, "::1")):
        with socket.socket(family, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.3)
            if sock.connect_ex((host, port)) == 0:
                return True
    return False


def stop_service(quiet=False):
    """start_service로 띄운 웹 화면을 끈다."""
    try:
        pid = int(PID_FILE.read_text())
        command = subprocess.run(["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True).stdout
        if "team_service.py" in command:        # 우리가 띄운 것이 맞을 때만 끈다
            os.kill(pid, signal.SIGTERM)
            for _ in range(20):
                if subprocess.run(["ps", "-p", str(pid)], capture_output=True).returncode != 0:
                    break
                time.sleep(0.1)
            if not quiet:
                print("✅ 웹 화면을 껐습니다.")
        elif not quiet:
            print("켜져 있는 웹 화면이 없습니다.")
    except (FileNotFoundError, ValueError, ProcessLookupError):
        if not quiet:
            print("켜져 있는 웹 화면이 없습니다.")
    PID_FILE.unlink(missing_ok=True)


def start_service(team_file="xray_team", port=8000):
    """팀을 웹 화면으로 띄우고 브라우저를 연다. 다시 부르면 전에 띄운 것을 끄고 새로 띄운다.

    team_file은 팀이 들어 있는 파일 이름이다 (xray_team, my_team). 터미널에서
    .venv/bin/python team_service.py xray_team 을 실행하는 것과 같은 일을 한다.
    """
    import urllib.request
    import webbrowser
    stop_service(quiet=True)
    while in_use(port):         # 다른 프로그램이 쓰는 번호면 다음 번호로 옮긴다
        port += 1
    url = f"http://127.0.0.1:{port}"
    status = Status(f"⏳ {team_file} 팀의 웹 화면을 띄우는 중입니다")
    with open(LOG_FILE, "w", encoding="utf-8") as log:
        proc = subprocess.Popen([sys.executable, "-u", str(ROOT / "team_service.py"), team_file, "--port", str(port), "--no-open"],
                                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    PID_FILE.write_text(str(proc.pid))
    while status.seconds() < 90:
        if proc.poll() is not None:
            status.done("❌ 웹 화면을 띄우지 못했습니다. 이유:")
            print(LOG_FILE.read_text(encoding="utf-8")[-1500:])
            return None
        try:
            urllib.request.urlopen(url + "/info", timeout=1)
            break
        except Exception:
            status.update(f"⏳ {team_file} 팀의 웹 화면을 띄우는 중입니다  ({status.seconds():.0f}초)")
            time.sleep(0.5)
    status.done(f"✅ 웹 화면이 켜졌습니다: {url}   (끄려면 stop_service())")
    webbrowser.open(url)
    if in_notebook():
        from IPython.display import HTML, display
        display(HTML(f'브라우저가 열리지 않으면 여기를 누르세요: <a href="{url}" target="_blank">{url}</a>'))
    return url


def connect_kiro(*args):
    """connect_kiro.py를 실행한다. 예: connect_kiro("--team", "xray_team"). 터미널에서 실행하는 것과 같다."""
    r = subprocess.run([sys.executable, str(ROOT / "connect_kiro.py"), *args], cwd=ROOT, capture_output=True, text=True)
    print((r.stdout + r.stderr).strip())


def open_kiro():
    """Kiro에서 이 실습 폴더를 연다. Kiro가 "이 폴더를 믿습니까?"라고 물으면 믿는다를 고른다."""
    r = subprocess.run(["open", "-a", "Kiro", str(ROOT)], capture_output=True, text=True)
    if r.returncode == 0:
        print(f"✅ Kiro에서 실습 폴더를 열었습니다: {ROOT}")
    else:
        print(f"❌ Kiro를 열지 못했습니다. Kiro를 직접 켜고 [File → Open Folder]에서 이 폴더를 여세요: {ROOT}")
