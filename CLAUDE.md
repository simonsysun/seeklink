# SeekLink — development and release discipline

Guidance for contributors — including AI assistants — working on this
repository. These rules distil lessons learned from prior release
iterations.

SeekLink is a public OSS tool shipped to PyPI. What lives in this
repository is visible to every user who lands on the GitHub page or
installs the package. Treat the repo as a product surface, not a work
log.

## Core principle: ship product, not process

The repo should read as a shipped tool, not as a dev journal. A visitor
should learn what SeekLink does, how to use it, and why to trust it —
not how we iterated, who reviewed what, or which approaches were tried
and rejected.

Three tests before anything lands on `main`:

1. **Does this help a user?** If no, it belongs in a git comment or a
   local note, not in a committed file.
2. **Would I be comfortable showing this to someone evaluating whether
   SeekLink is professionally maintained?** If no, rewrite or remove.
3. **Does this describe what shipped, or how we got there?** Shipped =
   keep. Got-there = exclude.

## Release discipline

### Pre-tag checklist

Never tag a release until every item below is satisfied. Tag premature
and you end up shipping visibly broken docs; the fix is another tag
the same day and a messy public history.

- [ ] All tests pass on the target Python versions declared in `pyproject.toml`.
- [ ] `CHANGELOG.md` has an entry for the new version. Describe user-visible
      changes, not internal iteration. One section per category (Added,
      Changed, Fixed, Removed, Dev). No "Deferred" in a Fixed list.
- [ ] `README.md` reviewed cold: open it in a fresh browser tab, read top
      to bottom as if you've never seen SeekLink, and flag anything that
      looks wrong or outdated.
- [ ] Every CLI command example in `README.md` and `llms.txt` has been
      manually executed against the bundled `tests/corpus/` fixture vault
      and verified to produce output matching the documented shape.
- [ ] Version in `pyproject.toml` matches the tag you're about to push.
- [ ] No committed files under `docs/` describe abandoned designs or
      internal scratch work. If a planning doc was useful during
      development, either delete it post-release or rewrite it as
      user-facing documentation.
- [ ] `tests/blind/results/` contains only measurements a user would want
      to read: one baseline, one shipping, and any upper-bound reference.
      Intermediate iteration outputs do not ship.

### Versioning

Follow [SemVer](https://semver.org/). A docs-only or cleanup release is
a patch bump; never ship docs-only changes by overwriting or tagging
the same version number twice. PyPI forbids version reuse anyway.

If a release ships with visible issues that get caught shortly after,
**yank the PyPI release** via its project page before cutting a
replacement patch, so the simple index no longer advertises the broken
docs snapshot.

## CHANGELOG conventions

- Sections per [Keep a Changelog](https://keepachangelog.com/): Added,
  Changed, Deprecated, Removed, Fixed, Security. Internal-only notes
  live under a `Dev` subsection and stay terse.
- Each entry describes one shipped change in plain English. A reader
  who has never used SeekLink should still learn what the change means
  for them.
- No process detail: no "after review", no "X caught this", no "Options
  A/B/C", no model or tool names of the internal review pipeline.
- Metric claims always include sample size and scope. Say "22-query
  pilot", not just "+4.5 pp MRR".
- Link to code only when the link helps a user (e.g. a file path for a
  new public helper). Don't link internal design docs.

## README conventions

The README is the single most important artifact in the repo — most
visitors read it and nothing else.

- Lead with what SeekLink does for a specific user, not what techniques
  it uses internally.
- Install + first-successful-search path is ≤ 30 seconds and correct.
  Specifically: `index` must come before `search` in any end-to-end
  example.
- Claims that can be measured include a measurement (sample size, date,
  environment). Don't advertise benchmark numbers without their caveats.
- Every command shown in examples runs against a fresh install. Re-test
  before every release.
- `For agents` section is first-class content. Agents are a real
  consumer; keep the output contract and exit-code table accurate.
- Link to `llms.txt` once at the bottom; don't repeat the agent contract
  in multiple places.

## Commit message conventions

- Describe user-observable change, not internal process. Good:
  `fix(cli): seeklink get <file>:LINE now reports beyond-EOF correctly`.
  Bad: `address review findings`.
- Scope prefix (`feat`, `fix`, `chore`, `docs`, `test`, `ci`) matches
  Conventional Commits so the log is auto-summarizable later.
- One logical change per commit. Don't bundle a feature, a refactor,
  and a docs fix in one commit.
- No `Co-Authored-By: <AI>` lines unless explicitly requested by the
  maintainer. The AI tooling we use internally is not public credit-worthy.
- Commit message body is optional. Use it for the *why* when the *what*
  isn't self-explanatory. Keep it under ~10 lines; longer belongs in a
  design doc or an issue.

## Documentation discipline

- `docs/` contains user-facing documentation: explainers, references,
  specifications. If a document describes internal iteration (Options
  A/B/C, abandoned approaches, blow-by-blow review notes), it does
  not belong in `docs/` — it belongs in a git commit body, a closed
  issue, or nowhere.
- `tests/` is fine to commit in its entirety, including fixture data.
  Intermediate measurement outputs (A/B/C of iteration runs) should be
  `.gitignore`d or pruned before the release commit.
- `CLAUDE.md` (this file) is the one exception to "no internal docs in
  the repo". It exists to instruct future AI collaborators, which is a
  legitimate developer need.

## Testing discipline

- Every shipped CLI command has at least one test covering the happy
  path and one edge case (missing file, empty input, malformed argument).
- Every public helper function in `seeklink/` has a unit test.
- Search-quality changes land alongside a blind-test result update in
  `tests/blind/results/`. Don't ship a search-quality claim without the
  numbers to back it.
- The blind-test framework at `tests/blind/` is the gate for any change
  that might affect retrieval ranking. If a change could move rankings,
  run config A (baseline) before and after, and commit both JSONs.
- Run `pytest` clean locally before pushing. The CI gate is the
  backstop, not the primary check.

## Code-comment discipline

Comments in `seeklink/*.py` describe the current design: what the code
does and why it's the right shape for the problem. They don't narrate
iteration ("originally we tried X, then Y, now Z"). That history is in
git; comments are for current state.

Exception: a design decision that a future maintainer might naively
reverse is worth documenting as "we tried the obvious alternative; it
broke because Y, so we do X instead". Keep it to 2-3 lines.

## Dependencies

- Runtime dependencies (`[project.dependencies]`) stay minimal. Every
  added dependency is a supply-chain cost and a support burden.
- Dev dependencies (`[dependency-groups.dev]`) can be more liberal,
  but any dependency added for a single test file or one-off task
  should be questioned.
- Pin minimum versions, not exact versions. Users need upgrade
  flexibility.

## Agent-specific notes

AI assistants working on this repo should:

- Read this file before starting any task.
- Read `README.md` and the last 3 entries of `CHANGELOG.md` to calibrate
  on current state.
- Never auto-merge to `main` or auto-tag a release without explicit
  human confirmation.
- When a task involves changing search behavior, run the blind-test
  framework locally and attach the before/after numbers to whatever
  handoff summary they produce.
- If uncertain whether something should land on `main` vs stay as a
  local note, default to staying local.
