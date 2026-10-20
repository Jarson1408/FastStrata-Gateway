import sys
sys.stdout.reconfigure(encoding='utf-8')
with open(r'C:\Users\user\AppData\Roaming\npm\node_modules\@opencode\cli\node_modules\@opencode\cli-windows-x64\bin\opencode.exe', 'rb') as f:
    data = f.read()

idx = data.find(b'id:"opencode.tool.read"')
print(data[idx:idx+1000].decode('utf-8', errors='ignore'))
