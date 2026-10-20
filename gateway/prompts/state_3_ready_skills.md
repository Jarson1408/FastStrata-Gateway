

[LODESTAR STATE 3 - USER REQUESTED SKILLS — LOAD BEFORE IMPLEMENTING]
The user explicitly asked to use skills in this task. Your first action in Phase C MUST be to load the relevant skill(s) via the skill tool:
  skill(skill_name="brainstorming")     for design / requirements gathering
  skill(skill_name="test-driven-development")  for TDD workflow
  skill(skill_name="systematic-debugging")    for bug investigation
  skill(skill_name="writing-plans")          for multi-step plans
Pick the skill that matches the task. After loading, follow its instructions.
Do NOT start writing code, running commands, or creating files until you have loaded the skill and followed its workflow.
Do NOT delegate the task to a subagent (task tool) before loading the skill — delegation is NOT a substitute for using the requested skill.
The gateway will ALLOW skill calls in state 3 (after active-context.md is committed), but skill calls in states 1 and 2 are silently dropped.
