"""Kiro 설정 파일에 실습 MCP 서버를 추가한다. 기존 설정은 .bak으로 백업한다.

사용법: .venv/bin/python connect_kiro.py           (memory만)
        .venv/bin/python connect_kiro.py --image   (images도)

Kiro의 사용자 설정 ~/.kiro/settings/mcp.json 에 넣으므로 어느 폴더를 열어도 쓸 수 있다.
"""
import json
import os
import shutil
import sys
from pathlib import Path

lab = Path(__file__).resolve().parent
cfg = Path(os.getenv("KIRO_MCP_CONFIG", "~/.kiro/settings/mcp.json")).expanduser()
py = lab / ".venv/bin/python"   # setup.sh가 만든 가상환경의 파이썬
if not py.exists():
    sys.exit("❌ .venv가 없습니다. 먼저 'bash setup.sh'를 실행해 주세요.")
py = str(py)

data = json.loads(cfg.read_text(encoding="utf-8")) if cfg.exists() and cfg.read_text(encoding="utf-8").strip() else {}
if cfg.exists():
    shutil.copy(cfg, str(cfg) + ".bak")
servers = data.setdefault("mcpServers", {})
# 저장과 조회는 묻지 않고 실행, 삭제(forget)는 매번 확인받는다
servers["memory"] = {
    "command": py,
    "args": [str(lab / "memory_mcp.py")],
    "disabled": False,
    "autoApprove": ["remember", "recall", "list_memories"],
}
if "--image" in sys.argv:
    servers["images"] = {"command": py, "args": [str(lab / "image_mcp.py")], "disabled": False, "autoApprove": ["search_images", "show_image"]}

cfg.parent.mkdir(parents=True, exist_ok=True)
cfg.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"✅ 추가했습니다: {', '.join(k for k in servers if k in ('memory', 'images'))}")
print(f"   설정 파일: {cfg}")
print("   Kiro는 설정 파일이 바뀌면 자동으로 다시 연결합니다. Kiro 채팅창에 이렇게 입력해 보세요.")
print("   이거 기억해 줘: 다음 주 화요일까지 RAG 과제 제출")
