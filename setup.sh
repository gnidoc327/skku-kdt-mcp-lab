#!/bin/bash
# 실습 준비를 한 번에: Ollama 확인 → 가상환경 → JupyterLab 설치 → 모델 내려받기 → 동작 확인 → JupyterLab 실행
# 사용법: bash setup.sh     (다음에 다시 열 때도 같은 명령)
set -e
cd "$(dirname "$0")"

echo "1/5 Ollama 확인"
if ! command -v ollama >/dev/null; then
  echo "❌ Ollama가 없습니다. https://ollama.com/download 에서 앱을 설치하고 한 번 실행한 뒤 다시 해 주세요."
  exit 1
fi
if ! ollama list >/dev/null 2>&1; then
  echo "❌ Ollama 앱이 꺼져 있습니다. 응용 프로그램에서 Ollama를 실행한 뒤 다시 해 주세요."
  exit 1
fi

echo "2/5 Python 가상환경(.venv) 준비"
if [ ! -x .venv/bin/python ]; then
  PY=""
  for c in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$c" >/dev/null && "$c" -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null; then PY="$c"; break; fi
  done
  if [ -z "$PY" ]; then
    echo "❌ Python 3.10 이상이 필요합니다. https://www.python.org/downloads/ 에서 설치하거나"
    echo "   Homebrew가 있으면 'brew install python@3.13'을 실행한 뒤 다시 해 주세요."
    exit 1
  fi
  "$PY" -m venv .venv
fi

echo "3/5 JupyterLab과 실습 패키지 설치 (처음 한 번만, 1~2분)"
.venv/bin/python -m pip install -q --disable-pip-version-check jupyterlab ipykernel "mcp>=2.3" "chromadb>=0.5" "ollama>=0.4" numpy matplotlib

echo "4/5 모델 내려받기 (처음 한 번만, 합쳐서 약 6.5GB. Wi-Fi에 따라 10~30분)"
for m in bge-m3 gemma3:4b llama3.2:3b; do
  echo "   - $m"
  ollama pull "$m"
done

echo "5/5 동작 확인"
.venv/bin/python check.py

echo
echo "준비 끝. 브라우저에 JupyterLab이 열리면 example/step0 부터 순서대로 실행하세요."
echo "끝낼 때는 이 터미널에서 Ctrl+C 를 두 번 누릅니다."
if [ -z "$NO_LAUNCH" ]; then exec .venv/bin/jupyter lab --notebook-dir=example; fi
