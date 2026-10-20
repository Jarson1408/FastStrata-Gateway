import os
import sqlite3
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")

db = os.path.expanduser("~/.local/share/opencode/opencode.db")
conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
conn.row_factory = sqlite3.Row
c = conn.cursor()
target = "ses_ec67de302ffeLO4msYFQ2JJMJ3"
msgs = c.execute("SELECT * FROM session_message WHERE session_id = ? ORDER BY seq", (target,)).fetchall()
for m in msgs:
    print(f"=== SEQ {m['seq']} | TYPE {m['type']} | ID {m['id']} ===")
    data = json.loads(m["data"])
    if "content" in data:
        for idx, part in enumerate(data["content"]):
            print(f"  Part {idx}: type={part.get('type')} keys={list(part.keys())}")
            if part.get("type") == "tool-call":
                print(f"    tool-call: {part}")
            elif part.get("type") == "text":
                print(f"    text: {part.get('text')}")
            elif part.get("type") == "reasoning":
                text = part.get("text", "")
                print(f"    reasoning tail (last 300 chars): {text[-300:]}")
    else:
        print("  no content in data:", data)
