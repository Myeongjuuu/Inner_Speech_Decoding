"""Block commits/pushes that could attribute this repository to another identity.

This is a local guard, not a substitute for checking GitHub's account mapping.
Hooks can be bypassed, so the publishing process also audits the complete history.
"""
from pathlib import Path
import subprocess
import sys

NAME = "Myeongjuuu"
EMAIL = "155465796+Myeongjuuu@users.noreply.github.com"


def git(*args):
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True,
                          encoding="utf-8", errors="replace").stdout.strip()


def check_current():
    expected = f"{NAME} <{EMAIL}> "
    for field in ("GIT_AUTHOR_IDENT", "GIT_COMMITTER_IDENT"):
        if not git("var", field).startswith(expected):
            raise RuntimeError("Identity mismatch. Commit blocked; no private identity is printed.")


def check_history():
    for sha in git("rev-list", "--all").splitlines():
        fields = git("show", "-s", "--format=%an%x00%ae%x00%cn%x00%ce%x00%P", sha).split("\0")
        author, email, committer, committer_email, parents = fields
        if (author, email) != (NAME, EMAIL):
            raise RuntimeError("Unexpected author in reachable history. Push blocked.")
        # The target's original root commit was created through GitHub's web UI.
        github_root = not parents and committer == "GitHub" and committer_email == "noreply@github.com"
        if not github_root and (committer, committer_email) != (NAME, EMAIL):
            raise RuntimeError("Unexpected committer in reachable history. Push blocked.")


if __name__ == "__main__":
    try:
        check_current()
        if "--history" in sys.argv:
            check_history()
        print("Public repository identity verified.")
    except (RuntimeError, subprocess.CalledProcessError) as error:
        print("Identity validation failed. No commit/push was authorized.", file=sys.stderr)
        sys.exit(1)
