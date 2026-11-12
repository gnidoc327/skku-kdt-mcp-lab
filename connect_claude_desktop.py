"""Claude Desktop 설정 파일에 실습 MCP 서버를 추가한다. 기존 설정은 .bak으로 백업한다.

사용법: python3 connect_claude_desktop.py           (memory만)
        python3 connect_claude_desktop.py --image   (images도)
"""
import json
import os
import shutil
import sys
from pathlib import Path

lab = Path(__file__).resolve().parent
cfg = Path(os.getenv("CLAUDE_DESKTOP_CONFIG", "~/Library/Application Support/Claude/claude_desktop_config.json")).expanduser()
py = lab / ".venv/bin/python"   # setup.sh가 만든 가상환경의 파이썬
if not py.exists():
    sys.exit("❌ .venv가 없습니다. 먼저 'bash setup.sh'를 실행해 주세요.")
py = str(py)

data = json.loads(cfg.read_text(encoding="utf-8")) if cfg.exists() else {}
if cfg.exists():
    shutil.copy(cfg, str(cfg) + ".bak")
servers = data.setdefault("mcpServers", {})
servers["memory"] = {"command": py, "args": [str(lab / "memory_mcp.py")]}
if "--image" in sys.argv:
    servers["images"] = {"command": py, "args": [str(lab / "image_mcp.py")]}

cfg.parent.mkdir(parents=True, exist_ok=True)
cfg.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"✅ 추가했습니다: {', '.join(k for k in servers if k in ('memory', 'images'))}")
print(f"   설정 파일: {cfg}")
print("   Claude Desktop을 완전히 종료(⌘Q)했다가 다시 켜세요.")
