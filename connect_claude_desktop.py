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
uv = shutil.which("uv") or str(Path.home() / ".local/bin/uv")

data = json.loads(cfg.read_text(encoding="utf-8")) if cfg.exists() else {}
if cfg.exists():
    shutil.copy(cfg, str(cfg) + ".bak")
servers = data.setdefault("mcpServers", {})
servers["memory"] = {"command": uv, "args": ["run", str(lab / "memory_mcp.py")]}
if "--image" in sys.argv:
    servers["images"] = {"command": uv, "args": ["run", str(lab / "image_mcp.py")]}

cfg.parent.mkdir(parents=True, exist_ok=True)
cfg.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"✅ 추가했습니다: {', '.join(k for k in servers if k in ('memory', 'images'))}")
print(f"   설정 파일: {cfg}")
print("   Claude Desktop을 완전히 종료(⌘Q)했다가 다시 켜세요.")
