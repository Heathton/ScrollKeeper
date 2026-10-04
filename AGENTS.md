# AGENTS.md

Scope: the whole repository. These rules apply to every coding agent (lead sessions and delegated subagents alike).

ScrollKeeper is a Discord bot for tabletop campaigns: it records speaker-separated voice, transcribes it, writes session summaries, maintains campaign notes, and answers lore questions. The `README.md` describes how it works today; the open GitHub issues describe where it is going.

## Before You Start

1. Read the issue you are working on, in full, including its comments. Issues are the task list and the design record; there is no separate roadmap file.
2. List open issues (`gh issue list --state open`) to see what neighbours your change. Read another issue's body only when your work depends on it.
3. Read the code you are about to change. The codebase is small (`src/scrollkeeper/`, ~2.3k lines), so read whole modules rather than guessing from names.

## Target Environment

Design for where the bot will run, not for the developer's machine:

- **Kubernetes (k3s) deployed by Argo CD with auto-sync.** Every merge to `main` that changes a released image can restart the bot, so state must survive restarts (persist it in SQLite, never only in memory).
- **CPU-only nodes**: 4-core Intel Skylake (AVX2, no AVX-512/VNNI), 16 GB RAM, shared with other workloads. Do not require CUDA or a GPU in the bot or the speech-to-text service.
- **LLM via an OpenAI-compatible endpoint** (`/v1/chat/completions`) that may **cold-start for several minutes** (the GPU host wakes on demand). Use long timeouts and tell the Discord channel when the bot is waiting.
- **One bot replica** per Discord token. No Docker socket and no sibling containers: external services are reached by URL from configuration.

## Workflow

- **One worktree and branch per issue.** Update `main` first, then branch from it:

  ```bash
  git checkout main && git pull --ff-only origin main
  git worktree add .worktrees/<issue>-<slug> -b issue-<issue>/<slug>
  ```

  Do not edit files in the main checkout.
- **Commit** on the task branch when the work is complete, with a concise message that describes the outcome.
- **Open a draft pull request** (`gh pr create --draft`) after checking none exists for the branch (`gh pr list --head <branch> --state all`). The body says what changed, how it was tested, and what the operator still needs to verify by hand (for example a live Discord voice test).
- **Closing keywords:** use `Closes #N` when the PR fully delivers the issue's recommended action, and `Refs #N` when it delivers part of it. If work remains, file a follow-up issue and link it before the PR merges.
- **Never merge, and never push to `main`.** The operator reviews and merges every PR.
- Comment on issues to record decisions or findings. Append rather than rewriting issue bodies.

### Delegated agents (subagents)

A subagent edits **the worktree the lead names, by absolute path**. It does not create its own worktree or branch, and does not commit, push, or open PRs unless the lead's brief says so. It reports what it changed and what it could not verify.

## Public Repository

This repository is **public**. Never commit or post (in code, docs, issues, or PRs):

- Discord tokens, API keys, `.env` files, or recorded session data (`data/` is git-ignored; keep it that way).
- Details of the operator's private infrastructure: internal IP addresses, hostnames, or links to private repositories. Refer to deployment details generically ("an OpenAI-compatible endpoint", "the cluster").

## Code Conventions

- Python 3.11+, `from __future__ import annotations`, type hints, `dataclasses` for plain data, `logging.getLogger(__name__)` for logs. Match the surrounding code's style.
- **Never block the event loop.** File I/O, SQLite, audio decoding, and HTTP calls run in `asyncio.to_thread` or a worker thread. A stalled loop drops the Discord voice connection.
- **Configuration** comes from environment variables prefixed `SCROLLKEEPER_` (read in `config.py`). When you add or rename a setting, update `.env.example` and the README in the same change.
- **Data compatibility:** existing users have recorded sessions and notes. Version SQLite schema changes (`PRAGMA user_version`) with a forward migration, and keep old session audio reprocessable (or convert it in the migration).
- `discord-ext-voice-recv` (voice receive) is the most fragile dependency. Pin it to a commit, and change it deliberately in its own PR.

## Testing

Tests use the standard library `unittest`:

```bash
pip install -e .            # in a virtual environment
PYTHONPATH=src python -m unittest discover -s tests
```

- Add or update tests for every behaviour change. Tests must not need the network, a Discord connection, a GPU, or downloaded models: fake HTTP responses and audio inputs.
- Run the full suite before committing, and state the result in the PR. If a test cannot run in your environment (for example, missing dependencies), say so in the PR rather than skipping silently.
- Voice capture can't be tested end to end by an agent. List the manual checks the operator should run (a short test session on a test Discord server) in the PR body.
