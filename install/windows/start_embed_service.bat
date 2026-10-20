@echo off
REM Warm embedding service for the LODESTAR semantic memory.
REM
REM Without it every `embed.py search` / `embed.py save` starts a fresh Python
REM and pays ~7 s just to import sentence-transformers. LODESTAR makes two of
REM those per task, so a task carries ~16 s of pure process startup. With this
REM running the same call takes ~0.3 s.
REM
REM It is an optimisation, never a requirement: if this is not running, embed.py
REM loads the model in-process exactly as before. Nothing breaks, it is slower.
REM
REM CPU only. No GPU is used or assumed, so this behaves the same on any machine.
REM Launched as pythonw.exe (no console window, survives logout) — same reason
REM as start_proxy.bat, and the interpreter is detected the same way.

setlocal
for %%I in ("%~dp0..\..") do set "INSTALL_DIR=%%~fI"
set "LOG=%INSTALL_DIR%\embed_service.log"
if "%EMBED_SERVICE_PORT%"=="" set "EMBED_SERVICE_PORT=18001"

call :find_pythonw
if not defined PYTHONW (
  echo [%date% %time%] ERROR: no Python interpreter found. Install Python 3.12+ or set LOCAL_LLM_GATEWAY_PYTHONW. >> "%LOG%"
  exit /b 1
)

cd /d "%INSTALL_DIR%"
echo [%date% %time%] embedding service starting on port %EMBED_SERVICE_PORT% using "%PYTHONW%" >> "%LOG%"
start /B "" "%PYTHONW%" "%INSTALL_DIR%\.context\semantic-memory\embed.py" serve --port %EMBED_SERVICE_PORT% >> "%LOG%" 2>&1
exit /b 0

:find_pythonw
set "PYTHONW="
if defined LOCAL_LLM_GATEWAY_PYTHONW if exist "%LOCAL_LLM_GATEWAY_PYTHONW%" set "PYTHONW=%LOCAL_LLM_GATEWAY_PYTHONW%"
if defined PYTHONW goto :eof
for /f "usebackq delims=" %%I in (`py -3 -c "import os,sys;p=os.path.join(os.path.dirname(sys.executable),'pythonw.exe');print(p if os.path.exists(p) else '')" 2^>nul`) do set "PYTHONW=%%I"
if defined PYTHONW goto :eof
for /f "usebackq delims=" %%I in (`python -c "import os,sys;p=os.path.join(os.path.dirname(sys.executable),'pythonw.exe');print(p if os.path.exists(p) else '')" 2^>nul`) do set "PYTHONW=%%I"
if defined PYTHONW goto :eof
where pythonw.exe >nul 2>nul && set "PYTHONW=pythonw.exe"
goto :eof
