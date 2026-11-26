"""노트북을 git에 담을 때 실행 결과를 떼어 낸다 (git의 clean 필터로 쓴다).

수강생이 받는 노트북에 남의 실행 결과가 들어 있으면 실습이 안 되고, 결과에 내 컴퓨터의 경로 같은 것이 섞여 올라갈 수도 있다.
쓰는 법 (저장소를 올리는 사람만 한 번):
    git config filter.nbstrip.clean ".venv/bin/python nb_strip.py"
그 뒤로는 노트북을 실행한 채로 git add 해도 실행 결과 없이 담긴다. 내 컴퓨터의 파일은 그대로 남는다.
"""
import json
import sys

raw = sys.stdin.read()
try:
    nb = json.loads(raw)
    for cell in nb.get("cells", []):
        cell.pop("id", None)
        if cell.get("cell_type") == "code":
            cell["outputs"], cell["execution_count"] = [], None
        meta = cell.get("metadata", {})
        for key in ("execution", "collapsed", "scrolled", "jupyter"):
            meta.pop(key, None)
        if cell.get("cell_type") == "code":      # 강사가 자기 컴퓨터에서 켜 둔 REUSE = True는 기본값(False)으로 되돌려 담는다
            src = "".join(cell["source"])
            if "# REUSE = False\nREUSE = True" in src:
                src = src.replace("# REUSE = False\nREUSE = True", "REUSE = False\n# REUSE = True")
                cell["source"] = src.splitlines(keepends=True)
    # 키를 가나다순으로 적는다. JupyterLab이 저장하는 방식과 같아서, 누가 저장해도 같은 모양이 된다
    sys.stdout.write(json.dumps(nb, ensure_ascii=False, indent=1, sort_keys=True) + "\n")
except Exception:
    sys.stdout.write(raw)       # 노트북이 아니거나 깨진 파일이면 건드리지 않는다
