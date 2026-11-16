@echo off
rem ANIS GeTrans 설치 (Windows). 번들 폴더 안에 .venv 가상환경을 만든다.
rem   install.cmd           기본 (NVIDIA GPU용 CUDA판 PyTorch, 약 3GB)
rem   install.cmd --cpu     CPU판 PyTorch (GPU가 없는 PC, 매우 느림)
rem 처음 번역할 때 바탕 모델(Gemma 4 E2B, 약 9.6GB)을 HuggingFace에서 한 번 내려받는다.
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set TORCH_INDEX=https://download.pytorch.org/whl/cu128
:args
if "%~1"=="" goto argsdone
if /i "%~1"=="--cpu" set TORCH_INDEX=https://download.pytorch.org/whl/cpu
shift
goto args
:argsdone
set PY=
where py >nul 2>nul && set PY=py -3
if not defined PY (where python >nul 2>nul && set PY=python)
if not defined PY (
  echo Python 3.10 이상을 찾지 못했습니다. https://www.python.org 에서 설치할 때 "Add python.exe to PATH"를 켜 주세요.
  pause
  exit /b 1
)
%PY% -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" || (echo Python 3.10 이상이 필요합니다. & pause & exit /b 1)
%PY% -m venv .venv || (pause & exit /b 1)
.venv\Scripts\python.exe -m pip install --upgrade pip >nul
.venv\Scripts\python.exe -m pip install torch --index-url %TORCH_INDEX% || (pause & exit /b 1)
.venv\Scripts\python.exe -m pip install -e . || (pause & exit /b 1)
echo.
echo 설치 완료. 처음 번역할 때 바탕 모델(약 9.6GB)을 내려받으므로 시간이 걸립니다. 이 폴더에서:
echo   anis-getrans "今日は雨が降ってるね。"
echo   anis-getrans-gui      (또는 anis-getrans-gui.cmd 두 번 클릭)
echo   anis-getrans-server   (Ollama·OpenAI 호환 API, 기본 http://127.0.0.1:11436)
pause
