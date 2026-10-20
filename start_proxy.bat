@echo off
REM Forwarder. The wrapper's logic lives in install\windows\start_proxy.bat and
REM NOWHERE ELSE. Do not copy it back here.
REM
REM This file exists at all because a Scheduled Task registered before
REM 2026-09-03 points at this path, and because install/windows/install.ps1
REM generates a copy here when the install directory differs from the checkout.
REM
REM It is one line because two copies that both carried logic is what
REM S-2026-01 was: the 2026-09-01 fix removing the LOCAL_LLM_GATEWAY_UPSTREAM
REM default landed in install\windows\, while the Scheduled Task ran THIS file,
REM which still set it. The environment overrides settings.json, so every
REM gateway started by the task reached 127.0.0.1 instead of the configured
REM host, failed to discover n_ctx, and fell back to prune_limit=120000 --
REM while AGENTS.md, README.md and INSTALL.md all said the default was gone.
REM gateway/tests/test_autostart_wrappers.py fails if logic returns here.
call "%~dp0install\windows\start_proxy.bat" %*
exit /b %ERRORLEVEL%
