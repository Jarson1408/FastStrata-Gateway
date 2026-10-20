"""Quick test: does repair_json actually fix the args the bramka emits?

Reproducer: the OpenCode client errors with 'JSON Parse error: Expected }'
when the model emits a bash call with the embed.py search command. The
display shows args as something like:
  {"command": "python \"C:\\Users\\...\\embed.py search \"query\""}
The 5-times-repeated pattern in the user's log shows the issue is
specifically with the args the FSM produces.
"""
import json
import sys
sys.path.insert(0, ".")
from gateway.tool_schema import repair_json


def check(label, raw_args):
    print(f"--- {label} ---")
    print(f"  raw:   {raw_args[:120]}{'...' if len(raw_args) > 120 else ''}")
    try:
        strict = json.loads(raw_args)
        print(f"  strict json.loads: OK, parsed {len(strict)} keys")
    except Exception as e:
        print(f"  strict json.loads: FAIL — {e}")
    try:
        repaired = repair_json(raw_args)
        print(f"  repair_json:      OK, type={type(repaired).__name__}, value={str(repaired)[:120]}{'...' if len(str(repaired)) > 120 else ''}")
    except Exception as e:
        print(f"  repair_json:      FAIL — {e}")


# The actual args string as the bramka would emit it (with json.dumps escaping)
# raw_args before escaping is: {"command": "python \"C:\\path\\embed.py\" search \"query\""}
# after escaping the value: {"command": "python \\\"C:\\\\path\\\\embed.py\\\" search \\\"query\\\"\""}
# Actually let me just write the JSON string Python-style:
case_args = json.dumps({"command": 'python "C:\\path\\embed.py" search "query"'})
check("bash with embedded quotes in path AND query", case_args)

# Reproducer of the OpenCode error: missing closing brace
case_broken = '{"command": "python /path/script.py search foo"'
check("missing closing brace", case_broken)

# Trailing comma
case_trailing = '{"command": "ls",}'
check("trailing comma", case_trailing)

# Embedded JSON in command (the actual 2026-07-07 bug)
case_embedded = '{"command": "python \\\\path\\\\script.py embed.py search \\\\\\"query\\\\\\\""}'
check("embedded backslashes + nested quotes", case_embedded)
