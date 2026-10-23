"""Install repository-local public identity and hooks; never change global Git settings."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
NAME = "Myeongjuuu"
EMAIL = "155465796+Myeongjuuu@users.noreply.github.com"


def run(*args):
    subprocess.run(["git", "-C", str(ROOT), *args], check=True)


if __name__ == "__main__":
    run("config", "--local", "user.name", NAME)
    run("config", "--local", "user.email", EMAIL)
    run("config", "--local", "user.useConfigOnly", "true")
    run("config", "--local", "core.hooksPath", ".githooks")
    subprocess.run([sys.executable, str(ROOT / "scripts/check_git_identity.py"), "--history"],
                   cwd=ROOT, check=True)
