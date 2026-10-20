# Project Rules

Durable, project-specific rules. The gateway injects this file into every
request, so keep it short: one imperative line per rule, no prose, max 15
rules. Add a rule only when the user has stated a lasting preference or
corrected the same mistake twice — not for one-off details, and never for
your own guesses about the codebase. When the file is full, replace the
least useful rule instead of appending.

- Diagnostic scripts, smoke runners and debug dumps live only in `test_scripts/` — never in the repository root or in source directories.
- Run the project's test suite before declaring any task complete, and report the real result; never assume a suite passes.
- Verify each deliverable on disk (the file exists, the test actually runs) before reporting the task as done.
- Commit to git after finishing a task, and never commit logs, caches or diagnostic artifacts.
- Start a dedicated git branch for a large or multi-file task; do not commit straight to the default branch.
- Never leave a placeholder such as "rest of the code unchanged" in an edit — write the complete, working file.
