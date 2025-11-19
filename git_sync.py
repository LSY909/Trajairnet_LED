import subprocess
import datetime
import sys
import os

def run_git_command(command):
    try:
        result = subprocess.run(command, check=True, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr.strip()}")
        return None

def git_sync(msg=None):
    if not os.path.exists(".git"): return print("Not a git repo.")
    if not run_git_command("git status --porcelain"): return print("No changes.")
    print("Adding files...")
    run_git_command("git add .")
    msg = msg or f"Update: {datetime.datetime.now()}"
    print(f"Committing: {msg}")
    run_git_command(f'git commit -m "{msg}"')
    branch = run_git_command("git branch --show-current") or "main"
    print(f"Pushing to {branch}...")
    if run_git_command(f"git push origin {branch}") is not None: print("Done!")
    else: print("Push failed.")

if __name__ == "__main__":
    git_sync(sys.argv[1] if len(sys.argv) > 1 else None)