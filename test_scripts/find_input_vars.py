import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

path = r"C:\Users\user\AppData\Roaming\npm\node_modules\@opencode\cli\node_modules\@opencode\cli-windows-x64\bin\opencode.exe"
with open(path, "rb") as f:
    data = f.read()

# Let's search for each tool's name string, e.g. "opencode.tool.edit"
for name in ["edit", "subagent", "patch", "shell", "read", "write", "skill"]:
    pattern = f'id:"opencode.tool.{name}"'.encode("utf-8")
    idx = data.find(pattern)
    if idx != -1:
        snip = data[idx:idx+500].decode("utf-8", errors="ignore")
        print(f"=== TOOL: {name} ===")
        # search for input:
        m = re.search(r'input:([a-zA-Z0-9_$]+)', snip)
        if m:
            var_name = m.group(1)
            print("input var:", var_name)
            # find where var_name is defined
            vpos = data.rfind(f"{var_name}=o(".encode("utf-8"), 0, idx)
            if vpos != -1:
                vsnip = data[vpos:vpos+300].decode("utf-8", errors="ignore")
                print("definition:", vsnip)
            else:
                vpos = data.find(f"{var_name}=o(".encode("utf-8"))
                if vpos != -1:
                    vsnip = data[vpos:vpos+300].decode("utf-8", errors="ignore")
                    print("definition (forward):", vsnip)
