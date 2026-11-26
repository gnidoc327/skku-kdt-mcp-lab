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
    """한 줄짜리 진행 표시. 노트북에서는 같은 줄을 고쳐 쓴다. 터미널에서는 끝났을 때만 한 줄 찍는다."""

    def __init__(self, text="⏳ 시작합니다"):
        self.start, self.handle = time.time(), None
        if in_notebook():
            from IPython.display import display
            self.handle = display({"text/plain": text}, raw=True, display_id=True)
        else:
            print(text)

    def seconds(self):
        return time.time() - self.start

    def update(self, text):
        if self.handle:
            self.handle.update({"text/plain": text}, raw=True)

    def done(self, text):
        self.update(text) if self.handle else print(text)


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
                status.update(f"⏳ {status.seconds():.0f}초째 일하는 중: {', '.join(left)}")
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
        status.done(f"✅ {who}가 답을 썼습니다  ({status.seconds():.0f}초)")
        return r["message"]["content"].strip()
    from IPython.display import Markdown, display
    box, text, shown = None, "", 0.0
    for chunk in ollama.chat(model=model, messages=messages, options={"temperature": temperature}, stream=True):
        text += chunk.message.content or ""
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
