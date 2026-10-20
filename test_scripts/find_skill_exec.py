
path = r"C:\Users\user\AppData\Roaming\npm\node_modules\@opencode\cli\node_modules\@opencode\cli-windows-x64\bin\opencode.exe"
with open(path, "rb") as f:
    data = f.read()

pos = 0
found = 0
while True:
    idx = data.find(b'WL=o({id:', pos)
    if idx == -1:
        break
    pos = idx + 10
    snippet = data[max(0, idx - 100):min(len(data), idx + 800)]
    print(snippet.decode("utf-8", errors="ignore"))
    print("=" * 60)
