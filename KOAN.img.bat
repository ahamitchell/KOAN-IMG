@echo off
setlocal

set APP=%~dp0
cd /d "%APP%"

"%APP%.venv\Scripts\python.exe" -m streamlit run "%APP%ui_app.py"

pause
