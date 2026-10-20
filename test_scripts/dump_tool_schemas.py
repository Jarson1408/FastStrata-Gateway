
path = r"C:\Users\user\AppData\Roaming\npm\node_modules\@opencode\cli\node_modules\@opencode\cli-windows-x64\bin\opencode.exe"
with open(path, "rb") as f:
    data = f.read()

tools = ["question", "patch", "edit", "glob", "grep", "read", "shell", "skill", "subagent", "webfetch", "websearch", "write"]

for t in tools:
    target = f'id:"opencode.tool.{t}"'.encode("utf-8")
    idx = data.find(target)
    if idx != -1:
        snippet = data[max(0, idx - 400):min(len(data), idx + 1000)].decode("utf-8", errors="ignore")
        print(f"==================== TOOL: {t} ====================")
        print(snippet)
