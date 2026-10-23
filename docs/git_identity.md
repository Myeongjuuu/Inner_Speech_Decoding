# Repository author identity

All new commits in this repository must use:

- Author and committer name: `Myeongjuuu`
- Author and committer email: `155465796+Myeongjuuu@users.noreply.github.com`

This repository was prepared from file snapshots, without importing the Git objects, branches or commit history of the earlier working repository. The target's existing GitHub-created root commit is retained.

After cloning, run this once from the repository root:

```bash
python scripts/install_git_identity.py
```

It changes only this repository's local Git configuration and enables the tracked `pre-commit` and `pre-push` hooks. Global Git settings are untouched. The guards reject an unexpected current author/committer; the push guard checks all reachable history too. The original GitHub-generated root may have GitHub itself as committer, which does not attribute authorship to another personal account.

Clones do not automatically activate hooks. Hooks can be bypassed with Git options or by changing configuration, so they are safeguards, not an absolute guarantee. Before publication, verify that every new commit's author **and** committer use the intended noreply email and that GitHub maps authorship to the intended account. Do not add private email addresses, unrelated account names or `Co-authored-by` trailers to messages.
