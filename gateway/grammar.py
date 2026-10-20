"""GBNF for llama-server, and the much larger thing it deliberately does not do.

The eGPU ExLlamaV3 server constrains tool-call syntax at its sampler with
`LLGuidanceFilter` (see AGENTS.md, "The Closers Are Now Mandatory"). llama.cpp
accepts a `grammar` field on `/completion`, so the obvious next step is parity.

**It is not parity, and the same grammar would be a regression here.** GBNF has
no lazy lexeme and no negative lookahead -- both verified against llama.cpp's
own grammar README, which documents `*`, `+`, `?`, `{m,n}`, ranges and negated
ranges, and nothing else. The llguidance rule that makes the eGPU version work
is

    pbody[lazy]: TEXT "\\n</parameter>\\n"

and `lazy` is the entire trick: it PREFERS the shortest match without
forbidding a longer one. The nearest GBNF expressible thing is "a parameter
value may not contain a line `</parameter>`", which does forbid it -- and that
breaks this workload outright, because a `write` call's content legitimately
contains that line. This repository is the proof: `inject_tools_to_prompt`
emits an example containing `</parameter>`, and AGENTS.md documents it. A
grammar built that way would make the model unable to write this project's own
files, and the failure would arrive as "the model refuses to write that file",
which reads as a model problem for a week.

So on llama.cpp the tolerant parser stays the answer, exactly as AGENTS.md
already says for the other direction: a constraint at the sampler is not a
reason to remove tolerance at the parser, and the absence of one is not a
reason to invent a worse constraint.

What IS safe is pinning a single literal the model must open with, leaving
everything after it unconstrained. That is enough for `tool_choice` naming one
function, which a prefill cannot do -- an open `<tool_call>` makes a call the
only plausible continuation, but it cannot make the model pick `read` out of a
list of nine.

**Not verified against a live llama.cpp.** No llama-server was reachable when
this was written (2026-09-12: the rig was on the eGPU upstream, and the Mac
Studio was down), so the grammar's acceptance is argued from llama.cpp's
documented GBNF syntax rather than measured. The blast radius is nil until a
client sends `tool_choice: {"type": "function", ...}`, which nothing here does
today -- but before relying on it, point the gateway at a llama-server and run

    python test_scripts/check_forced_function.py

which is the check that turns this paragraph into a measurement.
"""
from __future__ import annotations

import re

#: What a tool name may contain. Narrower than the tool schema allows on
#: purpose: anything outside this cannot be escaped into a GBNF literal without
#: thinking about it, and a name that needs thinking about is a name worth
#: refusing.
_SAFE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]{0,63}$")

#: Everything after the opener, unconstrained. NUL is excluded because a
#: negated class needs something to negate; no sampler emits it.
_ANY_TAIL = 'tail ::= [^\\x00]*'


def is_pinnable(name: str) -> bool:
    return bool(_SAFE_NAME.match(name or ""))


def forced_function(name: str) -> str | None:
    """GBNF that forces the turn to open with `<function=NAME>`, or None.

    The prompt this accompanies already ends with an open `<tool_call>`, so
    generation starts exactly where this grammar does. Everything after the
    opener is `tail`: the parameters, their values and the closers are NOT
    constrained, for the reason in this module's docstring.
    """
    if not is_pinnable(name):
        return None
    return f'root ::= "<function={name}>" tail\n{_ANY_TAIL}\n'
