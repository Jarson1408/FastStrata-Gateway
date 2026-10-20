
path = r"C:\Users\user\AppData\Roaming\npm\node_modules\@opencode\cli\node_modules\@opencode\cli-windows-x64\bin\opencode.exe"
with open(path, "rb") as f:
    data = f.read()

for t in ["question", "patch", "edit", "glob", "grep", "read", "shell", "skill", "subagent"]:
    target = f'id:"opencode.tool.{t}"'.encode("utf-8")
    idx = data.find(target)
    if idx != -1:
        # Search backward 500 bytes and forward 500 bytes for description and input schema
        sub = data[max(0, idx - 800):min(len(data), idx + 200)].decode("utf-8", errors="ignore")
        print(f"=== {t.upper()} ===")
        # Look for description: or input:
        print(sub)
        print("-" * 50)
