const fs = require('fs');
const path = require('path');
const base = 'C:\\Users\\user\\AppData\\Roaming\\npm\\node_modules\\@opencode';

function search(dir) {
    for (const f of fs.readdirSync(dir)) {
        const full = path.join(dir, f);
        if (fs.statSync(full).isDirectory()) search(full);
        else if (f.endsWith('.js')) {
            const txt = fs.readFileSync(full, 'utf8');
            if (txt.includes('skill')) {
                const lines = txt.split('\n');
                for (let i = 0; i < lines.length; i++) {
                    const l = lines[i];
                    if (l.includes('tool') && l.includes('skill')) {
                        console.log(`${f}:${i+1}: ${l.trim().slice(0, 120)}`);
                    }
                }
            }
        }
    }
}
search(base);
