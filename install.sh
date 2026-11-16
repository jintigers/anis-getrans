#!/usr/bin/env bash
# ANIS GeTrans 설치 (Linux / macOS). 번들 폴더 안에 .venv 가상환경을 만든다.
#   ./install.sh            기본 (Linux: CUDA판 PyTorch가 함께 설치되어 NVIDIA GPU를 씀 · macOS: CPU)
#   ./install.sh --cpu      CPU판 PyTorch만 설치 (GPU가 없는 PC, 받는 크기가 작음)
# 처음 번역할 때 바탕 모델(Gemma 4 E2B, 약 9.6GB)을 HuggingFace에서 한 번 내려받는다.
set -euo pipefail
cd "$(dirname "$0")"
CPU=0
for a in "$@"; do
  case "$a" in
    --cpu) CPU=1 ;;
    -h|--help) sed -n 2,5p "$0"; exit 0 ;;
    *) echo "알 수 없는 옵션: $a" >&2; exit 2 ;;
  esac
done
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  for c in python3.13 python3.12 python3.11 python3.10 python3 python; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
      PY="$c"; break
    fi
  done
fi
if [ -z "$PY" ]; then
  echo "Python 3.10 이상을 찾지 못했습니다. https://www.python.org 에서 설치한 뒤 다시 실행하세요." >&2
  exit 1
fi
echo "Python: $(command -v "$PY") ($("$PY" -V))"
"$PY" -m venv .venv
.venv/bin/python -m pip install --upgrade pip >/dev/null
if [ "$CPU" = 1 ]; then
  .venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
fi
.venv/bin/python -m pip install -e .
chmod +x anis-getrans anis-getrans-gui anis-getrans-server "ANIS GeTrans GUI.command" 2>/dev/null || true
echo
echo "설치 완료. 처음 번역할 때 바탕 모델(약 9.6GB)을 내려받으므로 시간이 걸립니다. 이 폴더에서:"
echo "  ./anis-getrans \"今日は雨が降ってるね。\""
echo "  ./anis-getrans-gui"
echo "  ./anis-getrans-server      (Ollama·OpenAI 호환 API, 기본 http://127.0.0.1:11436)"
