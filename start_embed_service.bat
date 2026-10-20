@echo off
REM Forwarder. The wrapper's logic lives in install\windows\start_embed_service.bat
REM and NOWHERE ELSE. Do not copy it back here.
REM
REM This file exists because the `local-llm-embeddings` Scheduled Task is
REM registered against this path. Until 2026-09-03 it was a byte-identical
REM second copy -- which is not a safe state, only a lucky one: its twin
REM start_proxy.bat was byte-identical too, right up to the day one of them
REM was fixed and the other was not (S-2026-01).
REM gateway/tests/test_autostart_wrappers.py fails if logic returns here.
call "%~dp0install\windows\start_embed_service.bat" %*
exit /b %ERRORLEVEL%
