"""Kiro 설정 파일에 실습 MCP 서버를 추가한다. 기존 설정은 .bak으로 백업한다.

사용법: .venv/bin/python connect_kiro.py           (memory만)
        .venv/bin/python connect_kiro.py --image   (images도)
        .venv/bin/python connect_kiro.py --drug    (2일차: 신약 후보 검토 도구 drug-review도)
        .venv/bin/python connect_kiro.py --team xray_team   (2일차: 내가 만든 Multi-Agent 팀. my_team도 가능)
        .venv/bin/python connect_kiro.py --dev       (Kiro로 이 폴더에서 개발할 때: 자주 쓰는 안전한 명령은 묻지 않고 실행)
        .venv/bin/python connect_kiro.py --dev-off   (--dev로 넣은 허용 규칙만 다시 뺀다)

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


def dev_rules():
    """Kiro가 이 실습 폴더에서 일할 때 묻지 않고 해도 되는 일. 여기에 없는 것은 전처럼 매번 묻는다.

    묻게 남겨 둔 것: 파일 삭제(rm), 패키지 설치(pip), 모델 받기(ollama pull), git push, 증류 학습(mlx_lm lora).
    Kiro는 && 나 | 로 이은 명령을 하나씩 따로 검사한다.
    echo는 일부러 뺐다. 허용한 명령 뒤에 '> 파일'을 붙이면 폴더 밖에도 쓸 수 있고, 이것은 규칙으로 막히지 않는다
    (Kiro CLI 3.5.0에서 확인). 그래서 이 규칙은 편의 기능이지 안전장치가 아니다.
    """
    venv_py = [f"{lab}/.venv/bin/python *", ".venv/bin/python *", "../.venv/bin/python *"]
    return [
        # 가상환경 파이썬으로 실습 코드 실행. 학습은 오래 걸리고 뜨거워지므로 매번 묻는다
        {"capability": "shell", "effect": "allow", "match": venv_py, "exclude": ["*mlx_lm*lora*", "*train_student*", "*pip install*", "*pip uninstall*"]},
        # 보기만 하는 명령
        {"capability": "shell", "effect": "allow", "match": [
            "cd *", "pwd", "ls", "ls *", "cat *", "head *", "tail *", "grep *", "wc *", "which *",
            "git status*", "git diff*", "git log*", "git add *", "git commit *",
            "ollama list", "ollama ps", "ollama show *", "ollama stop *",
            "curl http://localhost*", "curl -s http://localhost*", "curl http://127.0.0.1*", "curl -s http://127.0.0.1*"]},
        # 이 실습 폴더 안의 파일 고치기 (.git, .vscode, .kiro/agents 등은 Kiro가 언제나 묻는다)
        {"capability": "fs_write", "effect": "allow", "match": [f"{lab}/**"], "exclude": [f"{lab}/setup.sh", f"{lab}/.venv/**"]},
        # 실습 MCP 서버의 도구. 기억 삭제(forget)만 매번 확인
        {"capability": "mcp", "effect": "allow", "match": ["memory/*", "images/*", "drug-review/*", "team-*/*"], "exclude": ["memory/forget"]},
    ]


if "--dev" in sys.argv or "--dev-off" in sys.argv:
    import yaml
    perm = Path(os.getenv("KIRO_PERMISSIONS", "~/.kiro/settings/permissions.yaml")).expanduser()
    doc = (yaml.safe_load(perm.read_text(encoding="utf-8")) if perm.exists() else None) or {}
    if not isinstance(doc, dict) or not isinstance(doc.get("rules", []), list):
        sys.exit(f"❌ {perm} 의 형식을 읽을 수 없어 건드리지 않았습니다. 파일을 강사에게 보여 주세요.")
    if perm.exists():
        shutil.copy(perm, str(perm) + ".bak")
    mine = dev_rules()
    ours = lambda r: r in mine or (isinstance(r, dict) and any(str(lab) in str(m) for m in r.get("match", [])))
    doc["rules"] = [r for r in doc.get("rules", []) if not ours(r)]      # 전에 넣은 규칙은 빼고
    if "--dev" in sys.argv:
        doc["rules"] += mine
    perm.parent.mkdir(parents=True, exist_ok=True)
    perm.write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False, width=200), encoding="utf-8")
    if "--dev-off" in sys.argv:
        print(f"✅ 허용 규칙을 뺐습니다. 이제 Kiro가 전처럼 매번 묻습니다.\n   설정 파일: {perm}")
        sys.exit(0)
    print(f"✅ Kiro가 이 폴더({lab})에서 묻지 않고 하는 일을 정했습니다.")
    print("   묻지 않음: 가상환경 파이썬 실행, 파일 보기, 이 폴더 안의 파일 고치기, git 상태 보기와 commit, 실습 MCP 도구")
    print("   매번 물음: 파일 삭제, 패키지 설치, 모델 받기, git push, 증류 학습, 그 밖의 모든 명령")
    print(f"   설정 파일: {perm}   (되돌리기: .venv/bin/python connect_kiro.py --dev-off)")
    print("   이제 Kiro에서 File → Open Folder로 이 폴더를 열고, 채팅창에 이렇게 입력해 보세요.")
    print("   xray_team.py를 본보기로 my_team.py에 '복약 상담 팀'을 만들어 줘. 전문가는 두 명, 만든 뒤 질문 하나로 돌려서 확인해 줘")
    if len([a for a in sys.argv[1:] if a.startswith("--")]) == 1:
        sys.exit(0)

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
