@echo off
REM Wrapper for the Windows Scheduled Task that launches local-llm-gateway
REM detached as pythonw.exe (no console window, immune to Ctrl+C signals).
REM
REM The interpreter is DETECTED, not hardcoded: installation layouts differ
REM (Programs\Python\Python312, Python\pythoncore-3.14-64, the Store alias),
REM and a wrong hardcoded path silently fell through to whatever `pythonw` the
REM PATH happened to resolve. Asking Python where it lives works everywhere.
REM Override with LOCAL_LLM_GATEWAY_PYTHONW if you need a specific one.

setlocal
for %%I in ("%~dp0..\..") do set "INSTALL_DIR=%%~fI"
set "LOG=%INSTALL_DIR%\proxy.log"
REM The upstream is NOT defaulted here. GatewayConfig.from_env() reads
REM settings.json, and an environment variable OVERRIDES it — so setting a
REM fallback in this file silently beat the address the user configured in
REM the desktop app. This wrapper was written when the model ran on the same
REM machine; once the upstream moved to another host, every gateway started
REM by the Scheduled Task pointed at a localhost port with nothing on it,
REM logged "Could not discover n_ctx" and fell back to prune_limit=120000.
REM Measured 2026-09-01 after a reboot: the tray light was red and the cause
REM was two lines in this file. Set
REM LOCAL_LLM_GATEWAY_UPSTREAM before calling this script if you want it to
REM win; leaving it unset lets settings.json decide.

call :find_pythonw
if not defined PYTHONW (
  echo [%date% %time%] ERROR: no Python interpreter found. Install Python 3.12+ or set LOCAL_LLM_GATEWAY_PYTHONW. >> "%LOG%"
  exit /b 1
)

cd /d "%INSTALL_DIR%"
echo [%date% %time%] gateway starting (pythonw, no console) using "%PYTHONW%" >> "%LOG%"
"%PYTHONW%" -m gateway >> "%LOG%" 2>&1
exit /b %ERRORLEVEL%

:find_pythonw
set "PYTHONW="
if defined LOCAL_LLM_GATEWAY_PYTHONW if exist "%LOCAL_LLM_GATEWAY_PYTHONW%" set "PYTHONW=%LOCAL_LLM_GATEWAY_PYTHONW%"
if defined PYTHONW goto :eof
REM The py launcher knows every installed version; ask the newest where it is.
for /f "usebackq delims=" %%I in (`py -3 -c "import os,sys;p=os.path.join(os.path.dirname(sys.executable),'pythonw.exe');print(p if os.path.exists(p) else '')" 2^>nul`) do set "PYTHONW=%%I"
if defined PYTHONW goto :eof
REM No py launcher: the Store alias on PATH still reports the real location.
for /f "usebackq delims=" %%I in (`python -c "import os,sys;p=os.path.join(os.path.dirname(sys.executable),'pythonw.exe');print(p if os.path.exists(p) else '')" 2^>nul`) do set "PYTHONW=%%I"
if defined PYTHONW goto :eof
where pythonw.exe >nul 2>nul && set "PYTHONW=pythonw.exe"
goto :eof
