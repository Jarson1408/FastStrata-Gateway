import os
import sqlite3
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")

db = os.path.expanduser("~/.local/share/opencode/opencode.db")
c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
c.row_factory = sqlite3.Row

target = "ses_ec67de302ffeLO4msYFQ2JJMJ3"
s = c.execute("SELECT * FROM session_v2 WHERE id = ?", (target,)).fetchone()
print("SESSION_V2:", dict(s) if s else None)

msgs = c.execute("SELECT * FROM session_message WHERE session_id = ? ORDER BY seq", (target,)).fetchall()
print(f"Total session_messages: {len(msgs)}")
for m in msgs:
    print("==================================================")
    print(f"MSG: {m['id']} | seq: {m['seq']} | type: {m['type']} | created: {m['time_created']}")
    try:
        data = json.loads(m["data"])
        # print concise summary
        if isinstance(data, dict):
            for k, v in data.items():
                if k == "content" and isinstance(v, list):
                    print("  content parts:", len(v))
                    for idx, part in enumerate(v):
                        ptype = part.get("type")
                        print(f"    [{idx}] type={ptype}")
                        if ptype == "text":
                            txt = part.get("text", "")
                            print(f"        text: {txt[:200]}... (len {len(txt)})")
                        elif ptype == "tool-call":
                            print(f"        tool-call: {part.get('toolName')} args={part.get('args')}")
                        elif ptype == "tool-result":
                            print(f"        tool-result: {part.get('toolName')} result={str(part.get('result'))[:200]}")
                        else:
                            print(f"        {part}")
                elif k in ("role", "finishReason", "error"):
                    print(f"  {k}: {v}")
                else:
                    val_str = str(v)
                    if len(val_str) > 150:
                        val_str = val_str[:150] + "..."
                    print(f"  {k}: {val_str}")
        else:
            print("  data:", data)
    except Exception as e:
        print("  Error decoding JSON:", e)
        print("  raw:", m["data"][:500])
