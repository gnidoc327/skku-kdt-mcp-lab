#!/bin/bash
# 실습 준비를 한 번에: Ollama 확인 → 가상환경 → 패키지 설치 → 모델 내려받기 → 동작 확인 → JupyterLab 실행
# 사용법: bash setup.sh     (다음에 다시 열 때도 같은 명령)
# 문제가 생기면 같은 폴더의 setup.log 파일을 보여 주세요.
set -e
set -o pipefail
cd "$(dirname "$0")"

LOG=setup.log
{
  echo "=== $(date '+%Y-%m-%d %H:%M:%S') setup.sh 시작"
  echo "폴더: $(pwd)"
  echo "macOS: $(sw_vers -productVersion 2>/dev/null) $(uname -m)"
  echo "터미널의 python3: $(command -v python3 2>/dev/null) $(python3 --version 2>&1)"
  echo "shell: $BASH_VERSION"
} > "$LOG"

# 명령의 출력을 화면에 보여 주면서 setup.log에도 남긴다
logged() { "$@" 2>&1 | tee -a "$LOG"; }

STEP="시작"
on_exit() {
  code=$?
  if [ $code -ne 0 ]; then
    echo
    echo "❌ '$STEP' 단계에서 멈췄습니다 (종료 코드 $code)."
    echo "   위에 나온 메시지와 이 폴더의 setup.log 파일을 강사에게 보여 주세요."
    echo "=== 실패: $STEP (종료 코드 $code)" >> "$LOG"
  fi
}
trap on_exit EXIT

STEP="1/6 Ollama 확인"; echo "$STEP"
if ! command -v ollama >/dev/null; then
  echo "❌ Ollama가 없습니다. https://ollama.com/download 에서 앱을 설치하고 한 번 실행한 뒤 다시 해 주세요."
  exit 1
fi
if ! ollama list >/dev/null 2>&1; then
  echo "❌ Ollama 앱이 꺼져 있습니다. 응용 프로그램에서 Ollama를 실행한 뒤 다시 해 주세요."
  exit 1
fi
echo "ollama: $(ollama --version 2>&1 | head -1)" >> "$LOG"

STEP="2/6 Python 가상환경(.venv) 준비"; echo "$STEP"
py_ok() { "$1" -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null; }
# 예전에 만든 .venv가 오래된 Python이거나 깨져 있으면 새로 만든다
if [ -d .venv ] && ! { py_ok .venv/bin/python && .venv/bin/python -m pip --version >/dev/null 2>&1; }; then
  echo "   기존 .venv를 쓸 수 없어 새로 만듭니다."
  echo "기존 .venv 삭제: $(.venv/bin/python --version 2>&1)" >> "$LOG"
  rm -rf .venv
fi
if [ ! -x .venv/bin/python ]; then
  # 맥에 기본으로 들어 있는 python3(/usr/bin/python3)은 3.9라서 실습 패키지가 설치되지 않는다.
  # 그래서 이 폴더 안(.tools)에만 Python 3.13을 따로 받아 쓴다. 관리자 암호가 필요 없고,
  # PATH와 ~/.zshrc를 건드리지 않으므로 터미널의 python3은 그대로다. 폴더를 지우면 함께 지워진다.
  PY=""
  TOOLS="$(pwd)/.tools"
  export UV_PYTHON_INSTALL_DIR="$TOOLS/python" UV_CACHE_DIR="$TOOLS/cache" UV_NO_MODIFY_PATH=1 UV_PYTHON_PREFERENCE=only-managed
  if [ ! -x "$TOOLS/uv" ]; then
    case "$(uname -m)" in arm64) ARCH=aarch64 ;; *) ARCH=x86_64 ;; esac
    echo "   실습 전용 Python 3.13을 이 폴더 안에 내려받습니다 (약 40MB, 맥의 기존 Python은 그대로 둡니다)"
    mkdir -p "$TOOLS"
    if curl -fsSL --retry 2 "https://github.com/astral-sh/uv/releases/latest/download/uv-$ARCH-apple-darwin.tar.gz" 2>>"$LOG" | tar -xz -C "$TOOLS" --strip-components 1 2>>"$LOG"; then :; else
      echo "   (내려받기 실패. 맥에 설치된 Python을 찾아봅니다)"; rm -f "$TOOLS/uv"
    fi
  fi
  if [ -x "$TOOLS/uv" ] && logged "$TOOLS/uv" python install -q 3.13; then
    PY="$("$TOOLS/uv" python find 3.13 2>>"$LOG" || true)"
  fi
  if [ -z "$PY" ]; then      # 내려받지 못했으면 맥에 이미 있는 Python 3.10 이상을 쓴다
    for c in python3.13 python3.12 python3.11 python3.14 python3.10 python3; do
      if command -v "$c" >/dev/null && py_ok "$c"; then PY="$c"; break; fi
    done
  fi
  if [ -z "$PY" ]; then
    echo "❌ 실습용 Python을 내려받지 못했고, 맥에도 Python 3.10 이상이 없습니다."
    echo "   인터넷 연결을 확인하고 다시 실행해 주세요. 계속 안 되면 https://www.python.org/downloads/ 에서"
    echo "   Python 3.13을 설치한 뒤 다시 실행합니다."
    exit 1
  fi
  echo "venv를 만드는 Python: $PY" >> "$LOG"
  logged "$PY" -m venv .venv
fi
echo "python: $(.venv/bin/python --version 2>&1) ($(.venv/bin/python -c 'import sys; print(sys.base_prefix)'))" >> "$LOG"

STEP="3/6 JupyterLab과 실습 패키지 설치"; echo "$STEP (처음 한 번만, 2~3분)"
logged .venv/bin/python -m pip install -q --disable-pip-version-check \
  jupyterlab ipykernel "mcp>=2.3" "chromadb>=0.5" "ollama>=0.4" numpy matplotlib \
  "strands-agents[ollama]" rdkit requests mlx-lm

STEP="4/6 Ollama 모델 내려받기"; echo "$STEP (처음 한 번만, 합쳐서 약 16GB. Wi-Fi에 따라 20~60분)"
for m in bge-m3 gemma3:4b llama3.2:3b gemma4:e4b; do
  echo "   - $m"
  ollama pull "$m"
  echo "ollama pull $m 완료" >> "$LOG"
done

STEP="5/6 증류 실습용 작은 모델 내려받기"; echo "$STEP (약 0.7GB)"
logged .venv/bin/python -c "
import os
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['HF_HUB_VERBOSITY'] = 'error'
from huggingface_hub import snapshot_download
snapshot_download('mlx-community/gemma-3-1b-it-4bit')
print('   - mlx-community/gemma-3-1b-it-4bit 준비 완료')
"

STEP="6/6 동작 확인"; echo "$STEP (1~2분. 큰 모델을 처음 메모리에 올리느라 시간이 걸립니다)"
logged .venv/bin/python check.py
logged .venv/bin/python check_day2.py

echo "=== 성공" >> "$LOG"
echo
echo "준비 끝. 브라우저에 JupyterLab이 열리면 번호 순서대로 실행하세요. (1일차 1-1 ~ 1-6, 2일차 2-1 ~ 2-6)"
echo "끝낼 때는 이 터미널에서 Ctrl+C 를 두 번 누릅니다."
echo "Kiro로 이어서 개발하려면 새 터미널에서: .venv/bin/python connect_kiro.py --dev"
STEP="JupyterLab 실행"
# caffeinate: JupyterLab이 켜져 있는 동안 컴퓨터가 잠들지 않게 한다 (잠들면 학습과 노트북 실행이 멈춘다)
if [ -z "$NO_LAUNCH" ]; then exec caffeinate -i .venv/bin/python -m jupyter lab --notebook-dir=example; fi
