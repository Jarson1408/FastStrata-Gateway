import os
import re

base = r"C:\Users\user\AppData\Roaming\npm\node_modules\@opencode\cli"
pattern = re.compile(r'name:\s*["\']([a-zA-Z0-9_-]+)["\']')

found = set()
for root, _dirs, files in os.walk(base):
    for f in files:
        if f.endswith(".js"):
            p = os.path.join(root, f)
            with open(p, "r", encoding="utf-8", errors="ignore") as fh:
                content = fh.read()
                # find tool registrations: tool({ ... name: "..." ... }) or similar
                for m in re.finditer(r'description:\s*["\'].*?["\'],\s*name:\s*["\']([a-zA-Z0-9_-]+)["\']', content):
                    found.add(m.group(1))
                for m in re.finditer(r'name:\s*["\']([a-zA-Z0-9_-]+)["\'],\s*description:', content):
                    found.add(m.group(1))

print("Tools found:", sorted(list(found)))
