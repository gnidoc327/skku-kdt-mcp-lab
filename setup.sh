#!/bin/bash
# 실습 준비를 한 번에: Ollama 확인 → 모델 내려받기 → uv 설치 → 동작 확인
# 사용법: bash setup.sh          (Memory 실습만, 약 1.2GB)
#         bash setup.sh --image  (이미지 실습까지, 약 3.3GB 추가)
set -e
cd "$(dirname "$0")"

echo "1/4 Ollama 확인"
if ! command -v ollama >/dev/null; then
  echo "❌ Ollama가 없습니다. https://ollama.com/download 에서 앱을 설치하고 한 번 실행한 뒤 다시 해 주세요."
  exit 1
fi
if ! ollama list >/dev/null 2>&1; then
  echo "❌ Ollama 앱이 꺼져 있습니다. 응용 프로그램에서 Ollama를 실행한 뒤 다시 해 주세요."
  exit 1
fi

echo "2/4 모델 내려받기 (처음 한 번만, 몇 분 걸립니다)"
ollama pull bge-m3
if [ "$1" = "--image" ]; then ollama pull gemma3:4b; fi

echo "3/4 uv 확인"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "4/4 동작 확인"
uv run --quiet check.py
echo
echo "준비 끝. README의 '2. Claude에 연결'로 넘어가세요."
