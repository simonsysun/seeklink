## Claude Code only

- Use plan mode before non-trivial code changes.
- After compacting, clearing, resuming, or switching worktrees, re-read `docs/current_focus.md` and check git status.
- For large or ambiguous features, interview first, write a short plan/spec, then implement from clean context when appropriate.
- Do not use Claude auto-memory, skills, or hidden local notes as project truth. Durable project state belongs in repo files or git history.
- Use skills for repeatable procedures, not for long-lived project state.

## Claude-triggered Codex second review

When asking Codex for a second review, prefer a fresh ephemeral read-only run:

codex exec --ephemeral --sandbox read-only "Review the current diff against main. Focus on correctness, security, behavior regressions, and missing test coverage. Do not modify files."

Do not create or reuse a long-running Codex reviewer session.

Do not create a generic `codex-review` worktree.

Apply review fixes in the primary implementation session unless the human explicitly asks Codex to patch.
