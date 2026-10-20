import re

path = r"C:\Users\user\AppData\Roaming\npm\node_modules\@opencode\cli\node_modules\@opencode\cli-windows-x64\bin\opencode.exe"
with open(path, "rb") as f:
    data = f.read()

# 1. Search for "opencode.tool."
print("=== TOOLS MATCHING 'opencode.tool.' ===")
for m in re.finditer(rb'opencode\.tool\.[a-zA-Z0-9_.-]+', data):
    print(m.group(0).decode("utf-8", errors="ignore"))

# 2. Extract surrounding code for each tool to see name, input, description
pos = 0
tools_info = []
for m in re.finditer(rb'id:"opencode\.tool\.([a-zA-Z0-9_.-]+)"', data):
    tool_id = m.group(1).decode("utf-8")
    idx = m.start()
    snippet = data[max(0, idx - 200):min(len(data), idx + 800)].decode("utf-8", errors="ignore")
    tools_info.append((tool_id, snippet))

print(f"\nFound {len(tools_info)} tool implementations:")
for tid, snip in tools_info:
    print(f"\n--- TOOL: {tid} ---")
    print(snip[:500])
