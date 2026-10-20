import os
import sys
import re
import subprocess

def sdd_workspace():
    res = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True)
    root = res.stdout.strip()
    dir_path = os.path.join(root, ".superpowers", "sdd")
    os.makedirs(dir_path, exist_ok=True)
    
    with open(os.path.join(dir_path, ".gitignore"), "w", encoding="utf-8") as f:
        f.write("*\n")
        
    print(dir_path)
    return dir_path

def task_brief(plan_file, task_number, out_file=None):
    if not os.path.isfile(plan_file):
        print(f"no such plan file: {plan_file}", file=sys.stderr)
        sys.exit(2)
        
    if not out_file:
        workspace_dir = sdd_workspace()
        out_file = os.path.join(workspace_dir, f"task-{task_number}-brief.md")
        
    with open(plan_file, "r", encoding="utf-8") as f:
        content = f.read()
        
    lines = content.split("\n")
    out_lines = []
    intask = False
    infence = False
    
    header_pat = re.compile(rf"^#+[ \t]+[Tt]ask[ \t]+{task_number}(?:\b|$|[^0-9])")
    next_task_pat = re.compile(r"^#+[ \t]+[Tt]ask[ \t]+[0-9]+")
    
    for line in lines:
        if line.startswith("```"):
            infence = not infence
            
        if not infence:
            if header_pat.match(line):
                intask = True
            elif intask and next_task_pat.match(line):
                intask = False
                
        if intask:
            out_lines.append(line)
            
    if not out_lines:
        print(f"task {task_number} not found in {plan_file}", file=sys.stderr)
        sys.exit(3)
        
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + "\n")
        
    print(f"wrote {out_file}: {len(out_lines)} lines")

def review_package(base, head, out_file=None):
    for ref in (base, head):
        res = subprocess.run(["git", "rev-parse", "--verify", "--quiet", ref], capture_output=True)
        if res.returncode != 0:
            print(f"bad ref: {ref}", file=sys.stderr)
            sys.exit(2)
            
    if not out_file:
        workspace_dir = sdd_workspace()
        b_short = subprocess.run(["git", "rev-parse", "--short", base], capture_output=True, text=True, check=True).stdout.strip()
        h_short = subprocess.run(["git", "rev-parse", "--short", head], capture_output=True, text=True, check=True).stdout.strip()
        out_file = os.path.join(workspace_dir, f"review-{b_short}..{h_short}.diff")
        
    commits = subprocess.run(["git", "log", "--oneline", f"{base}..{head}"], capture_output=True, text=True, check=True).stdout
    stat = subprocess.run(["git", "diff", "--stat", f"{base}..{head}"], capture_output=True, text=True, check=True).stdout
    diff = subprocess.run(["git", "diff", "-U10", f"{base}..{head}"], capture_output=True, text=True, check=True).stdout
    
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(f"# Review package: {base}..{head}\n\n")
        f.write("## Commits\n\n")
        f.write(commits + "\n")
        f.write("## Files changed\n\n")
        f.write(stat + "\n")
        f.write("## Diff\n\n")
        f.write(diff + "\n")
        
    commit_count = subprocess.run(["git", "rev-list", "--count", f"{base}..{head}"], capture_output=True, text=True, check=True).stdout.strip()
    size = os.path.getsize(out_file)
    print(f"wrote {out_file}: {commit_count} commit(s), {size} bytes")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python sdd_helper.py <command> [args...]", file=sys.stderr)
        sys.exit(1)
        
    cmd = sys.argv[1]
    if cmd == "sdd-workspace":
        sdd_workspace()
    elif cmd == "task-brief":
        if len(sys.argv) < 4:
            print("Usage: python sdd_helper.py task-brief PLAN_FILE TASK_NUMBER [OUTFILE]", file=sys.stderr)
            sys.exit(1)
        plan_file = sys.argv[2]
        task_num = sys.argv[3]
        outfile = sys.argv[4] if len(sys.argv) > 4 else None
        task_brief(plan_file, task_num, outfile)
    elif cmd == "review-package":
        if len(sys.argv) < 4:
            print("Usage: python sdd_helper.py review-package BASE HEAD [OUTFILE]", file=sys.stderr)
            sys.exit(1)
        base = sys.argv[2]
        head = sys.argv[3]
        outfile = sys.argv[4] if len(sys.argv) > 4 else None
        review_package(base, head, outfile)
    else:
        print(f"Unknown command: {cmd}", file=sys.stderr)
        sys.exit(1)
