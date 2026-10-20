@echo off
REM ==============================================================================
REM  Start the FastStrata Gateway desktop app (Dashboard + System Tray)
REM ==============================================================================
REM Runs under pythonw.exe so the tray app has no console window behind it. That
REM would hide a startup crash (a missing PySide6, say), so stdout and stderr go
REM to gui.log — check it if the tray icon never appears.
REM
REM The interpreter is detected rather than hardcoded; override with
REM LOCAL_LLM_GATEWAY_PYTHONW.

setlocal
set "INSTALL_DIR=%~dp0"
if "%INSTALL_DIR:~-1%"=="\" set "INSTALL_DIR=%INSTALL_DIR:~0,-1%"
set "LOG=%INSTALL_DIR%\gui.log"

call :find_pythonw
if not defined PYTHONW (
  echo No Python interpreter found. Install Python 3.12+ or set LOCAL_LLM_GATEWAY_PYTHONW.
  echo [%date% %time%] ERROR: no Python interpreter found. >> "%LOG%"
  exit /b 1
)

cd /d "%INSTALL_DIR%"
echo [%date% %time%] GUI starting using "%PYTHONW%" >> "%LOG%"
start /B "" "%PYTHONW%" -m gateway.gui >> "%LOG%" 2>&1
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
