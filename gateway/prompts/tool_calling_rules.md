<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format: an inner <function=NAME></function> block must be nested within <tool_call></tool_call> XML tags
- Required parameters MUST be specified
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
- NEVER end a turn with empty content after </think>. Every turn must contain either substantive text or a complete tool call with all required parameters.
- Do NOT produce filler turns like "Let me check..." or "Let me wait..." without an accompanying tool call. If you intend to check or wait, the same turn must include the actual tool call.
- NEVER use file reading tools (like read_file) on directories. If you need to list directory contents, use a dedicated list/bash tool.
- WORKING DIRECTORY & PATH RESOLUTION: The active project workspace directory is specified in the system prompt (e.g., `Current directory: <path>`). When accessing or reading any project files (including `.context/...` files, audit instructions, source code, configs), always use paths relative to the current workspace root, or use full absolute paths starting from the current workspace directory. Never assume or switch to an arbitrary default directory.
- WINDOWS BACKGROUND PROCESSES: When starting background servers/services (like uvicorn, django, etc.) in PowerShell, DO NOT use `Start-Process` with `-NoNewWindow` as it will redirect stdout back to the caller console, keeping the stream open and blocking the command executor. Instead, start the process completely detached, for example: `Start-Process -FilePath "python" -ArgumentList "-m", "uvicorn", "app.main:app", "--reload", "--port", "8000" -WindowStyle Hidden` or by redirecting stdout/stderr: `Start-Process -FilePath "python" -ArgumentList "-m", "uvicorn", "app.main:app", "--reload", "--port", "8000" -RedirectStandardOutput "uvicorn.log" -RedirectStandardError "uvicorn.err.log"`.
- MODULAR FILE GENERATION & LARGE WRITES: For large files, comprehensive multi-section reports, or extensive modules (>300 lines), NEVER attempt a single monolithic write tool call. Always generate modularly: create structured sub-files in dedicated directories or append/edit sections incrementally. This prevents stream timeouts and guarantees data integrity.
- SUBAGENT SEQUENTIAL DISCIPLINE: Always delegate tasks ONE AT A TIME. NEVER emit multiple <function=task> calls in a single turn. Wait for the subagent to complete and inspect its result before delegating the next task. Parallel subagent spawning causes hardware slot contention and cache thrashing.

CONTINUATION DISCIPLINE — read this carefully:
- After a tool returns a result, your NEXT turn MUST be one of:
    (a) Another tool call that makes progress on the user's task, OR
    (b) A short final answer ONLY when the entire task is complete.
- DO NOT produce a "summary", "recap", "done" bullet list, or "what's next" plan as a stand-alone turn between tool calls.
- If your todo list has any items in pending or in_progress status, you are NOT finished. Your next turn must be a tool call.
- A summary is ONLY appropriate as your VERY LAST turn after the final todo is completed. Until then: act, don't narrate.
</IMPORTANT>

