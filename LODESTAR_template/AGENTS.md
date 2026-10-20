### OPERATIONAL PROTOCOL (LODESTAR LIFECYCLE)

Every task MUST follow this exact 3-step cycle:

1. **START (Before touching any project code):**
   - Read project rules: read `.context/project_rules.md`.
   - Search past memory: run `python .context/semantic-memory/embed.py search "<keywords in English>"`.
   - Read & commit goal: update `.context/memory/active-context.md` setting:
     - `Goal: <clear summary of user request>`
     - `Status: W toku`
     - `Next Steps: <numbered steps>`
   *CRITICAL: Do NOT edit or create any project files until active-context.md has Status: W toku.*
   *(Autonomous subagents skip START and END — execute assigned subtask directly).*

2. **WORK (Implementation):**
   - Implement the planned changes following project rules.
   - Run tests and verify deliverables on disk.

3. **END (Before returning your final response):**
   - Save lesson: run `python .context/semantic-memory/embed.py save lessons "<lesson in English>"`.
   - Close context: update `.context/memory/active-context.md`:
     - Append completed task to `## Progress Log` (if > 5 entries, keep 5 newest and summarize older into one paragraph in the same write).
     - Clear `Goal` and `Next Steps`.
     - Set `Status: Zakończono`.
   - Durable rule: ONLY if the user corrected you or gave an explicit preference in this session, append ONE line to `.context/project_rules.md` (never rewrite the whole file, skip if nothing qualifies).
   *(Autonomous subagents do NOT modify `.context/memory/active-context.md` or save lessons).*

* **Past Reference Queries:** If the user asks about past implementations or "how you did this before" and the answer is not visible in workspace files, run `python .context/semantic-memory/embed.py search "<keywords in English>"` before answering.

---