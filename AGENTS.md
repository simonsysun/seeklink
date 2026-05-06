# Project conventions for AI coding agents

## Core model

The source of truth is the repository and git history.

Sessions, chat context, local notes, and worktrees are convenience layers only.
Important state must be recoverable from repo files or git history.

Keep this file short. Add rules only when they affect agent behavior or prevent
repeated mistakes.

## Start of session

Before editing:

1. Read `docs/current_focus.md` if it exists.
2. Check branch and git status.
3. Identify relevant verification commands.
4. For non-trivial changes, state the approach before writing code.

Do not rely on chat history as project memory.

## Current focus

Use `docs/current_focus.md` for current work state only. Keep it under 30 lines.

It should contain:
- current task;
- branch;
- next concrete step;
- open questions;
- verification commands;
- at most the latest two handoff notes.

Rewrite stale content instead of appending indefinitely.

Handoff format:

## Handoff (YYYY-MM-DD, Agent, branch)
Done: <completed work>
Open: <unresolved items>
Next: <next concrete step>
Status: <in progress / blocked / ready for review>

Do not create changelog, backlog, task tracker, memory, or per-agent state files
unless explicitly requested.

## History and decisions

Commit messages are the primary history layer.

Use commit bodies to explain why non-trivial changes were made.

Use `docs/decisions.md` only for decisions that remain active constraints on
future work. Do not create it by default.

Decision docs are appropriate when a decision:
- spans multiple commits or subsystems;
- involves external context not visible in code;
- prevents a recurring wrong choice;
- affects public API, data model, deployment, security, or product behavior.

Do not mirror every commit into decision docs.

Agents may propose decision entries, but must not create or update long-lived
decision records unless the human approves the decision text.

If `docs/decisions.md` grows beyond roughly 10 active decisions, propose moving
to `docs/decisions/YYYY-MM-DD-<slug>.md` ADR files.

## History lookup

When you need to understand why code is shaped a certain way:

1. Read `docs/current_focus.md`.
2. Read `docs/decisions.md` if present.
3. Use git history:
   - `git log --oneline -- <path>`
   - `git show <commit>`
   - `git blame <path>`

Do not infer historical intent from code alone when relevant git history exists.

## Commits

Use one logical change per commit when practical.

Small obvious commits may use only a subject line.

For non-trivial commits, include a body with:
- Why: reason this change exists
- Decision: design choice made, if any
- Rejected: alternatives considered, if useful
- Verification: checks run

Do not use commit bodies as file-by-file summaries.

Example:

refactor(auth): move token parsing into middleware

Why: upcoming API routes need consistent auth behavior, and handler-level
parsing was causing inconsistent error responses.

Decision: middleware is the single enforcement point for request auth.

Rejected: a shared helper would reduce duplication but still rely on each route
remembering to call it.

Verification: pnpm test auth, pnpm typecheck

## Branches, sessions, and worktrees

Default:
- one task = one branch;
- use the current session unless the task truly branches.

Use a worktree only when both are true:
- files will be modified; and
- another branch, task, or agent may need to modify files at the same time.

When parallel work is needed:
- one task = one branch = one worktree = one primary session;
- use consistent names for branch, worktree, and main session when practical.

Agents must not silently create new worktrees or assume a new session exists.

If the current task should be isolated in a new worktree, stop before editing and
tell the human:
- why a worktree is needed;
- proposed branch name;
- proposed worktree path;
- command to create it.

Do not create a generic worktree for a role such as `codex-review`. Worktrees are
task/branch containers, not reviewer containers.

## Second review

A second-review agent is read-only by default.

For Claude-triggered Codex reviews, prefer a fresh ephemeral non-interactive run
per branch, commit, or diff:

codex exec --ephemeral --sandbox read-only "<review prompt>"

Do not reuse a long-running reviewer session across unrelated tasks.

A reviewer may inspect files and report findings, but must not modify files,
commit, push, or update `docs/current_focus.md` unless the human explicitly asks
for a patch.

If review fixes require editing, use the primary implementation session or create
a task-specific branch/worktree.

## Merge protocol

Never merge to main unless `docs/current_focus.md` contains a human-written
marker:

## APPROVED FOR MERGE: <branch-name>

The marker is single-use and applies only to the named branch.

When feature branch work is complete:

1. Run required verification.
2. Commit and push the feature branch.
3. Update `docs/current_focus.md` with a handoff marked `ready for review`.
4. Stop.

When the approval marker is present, use the standard merge sequence unless repo
state makes it unsafe:

git switch main && git pull && git merge --no-ff <branch> && git push
git worktree remove <path> && git branch -d <branch>

Do not rebase, squash, or choose a different merge strategy unless the human
explicitly says so.

If there are merge conflicts, stop, list conflict files in `docs/current_focus.md`,
and wait for the human.

Do not reuse a stale approval marker for a later merge.

## Boundaries

Do not modify:
- `.env`, `.env.*`
- secrets, credentials, private keys
- generated output directories such as `dist/`, `build/`, `.next/`, `coverage/`

Do not modify unless explicitly required:
- lockfiles
- applied migrations
- vendored dependencies
- generated client code

If a protected file must change, explain why before changing it.

## Verification

Before claiming work is done, run the relevant project checks.

If verification commands are missing or unclear, discover them before asking the
human:
- read README and project docs;
- inspect package scripts and build files;
- identify the smallest useful smoke check;
- identify full test, lint, typecheck, and build commands when available.

Do not ask the human to name test commands before inspecting the project.

When commands are discovered and proven useful, propose updating this section
with:

Smoke:
- <command>

Full:
- <command>

Lint/typecheck/build:
- <command>

If checks cannot run, report the exact reason and what was verified instead.

Done means verified, not merely implemented.

## Working style

Prefer small diffs.
Do not make broad refactors while fixing a narrow bug.
Ask before adding production dependencies.
If corrected twice on the same issue in one session, stop and reassess the approach.
