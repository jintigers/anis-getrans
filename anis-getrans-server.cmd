@echo off
rem ANIS GeTrans API 서버 실행기 (Windows)
set PYTHONUTF8=1
"%~dp0.venv\Scripts\python.exe" -m anis_getrans.server %*
if errorlevel 1 pause
