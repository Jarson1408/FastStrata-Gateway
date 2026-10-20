
path = r"C:\Users\user\AppData\Roaming\npm\node_modules\@opencode\cli\node_modules\@opencode\cli-windows-x64\bin\opencode.exe"
with open(path, "rb") as f:
    data = f.read()

pos = 0
found = 0
while True:
    idx = data.find(b'"skill"', pos)
    if idx == -1 or found > 15:
        break
    pos = idx + 7
    snippet = data[max(0, idx - 100):min(len(data), idx + 400)]
    if b"properties" in snippet or b"description" in snippet:
        found += 1
        print("MATCH", found)
        print(snippet.decode("utf-8", errors="ignore"))
        print("=" * 50)
