"""Kiro 설정 파일에 실습 MCP 서버를 추가한다. 기존 설정은 .bak으로 백업한다.

사용법: .venv/bin/python connect_kiro.py           (memory만)
        .venv/bin/python connect_kiro.py --image   (images도)
        .venv/bin/python connect_kiro.py --drug    (2일차: 신약 후보 검토 도구 drug-review도)
        .venv/bin/python connect_kiro.py --team xray_team   (2일차: 내가 만든 Multi-Agent 팀. my_team도 가능)

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
if "--drug" in sys.argv:
    servers["drug-review"] = {
        "command": py,
        "args": [str(lab / "drug_tools_mcp.py")],
        "disabled": False,
        "autoApprove": ["compound_info", "molecule_properties", "similar_compounds", "pubmed_search", "adverse_events", "clinical_trials"],
    }
team = None
if "--team" in sys.argv:
    team = sys.argv[sys.argv.index("--team") + 1].removesuffix(".py") if sys.argv.index("--team") + 1 < len(sys.argv) else ""
    if not (lab / f"{team}.py").exists():
        sys.exit(f"❌ {team}.py 파일이 없습니다. 예: connect_kiro.py --team xray_team")
    # 팀 전체를 도구 하나(ask_team)로 내놓는다. 전문가 고르기와 분석은 내 컴퓨터의 팀이 한다
    servers[f"team-{team}"] = {"command": py, "args": [str(lab / "team_service.py"), team, "--mcp"], "disabled": False, "autoApprove": ["ask_team"]}

cfg.parent.mkdir(parents=True, exist_ok=True)
cfg.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"✅ 추가했습니다: {', '.join(k for k in servers if k in ('memory', 'images', 'drug-review', f'team-{team}'))}")
print(f"   설정 파일: {cfg}")
print("   Kiro는 설정 파일이 바뀌면 자동으로 다시 연결합니다. Kiro 채팅창에 이렇게 입력해 보세요.")
if team:
    print("   (답이 오기까지 1~2분 걸립니다. 팀 파일을 고친 뒤에는 Kiro의 MCP 서버 목록에서 다시 연결하세요)")
    print("   lobar_pneumonia.jpg 사진을 xray 팀에게 종합 검토시켜 줘" if team == "xray_team" else f"   {team} 팀에게 물어봐 줘: 검진 결과를 종합해서 설명해 줘")
elif "--drug" in sys.argv:
    print("   (Kiro에서 이 실습 폴더를 열어야 검토 절차 .kiro/steering/drug-review.md 가 함께 적용됩니다)")
    print("   ibuprofen을 다음 개발 단계로 진행해도 되는지 종합 검토해서 리포트로 만들어 줘")
else:
    print("   이거 기억해 줘: 다음 주 화요일까지 RAG 과제 제출")
