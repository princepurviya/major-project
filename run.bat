@echo off
cd /d "%~dp0"
echo Starting MCP-Powered Conversational Data Analytics and Intelligence Platform...
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m streamlit run app.py
) else (
    call .venv\Scripts\activate
    streamlit run app.py
)
pause
