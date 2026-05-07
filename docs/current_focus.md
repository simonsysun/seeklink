# Current Focus

Task: Add optional read-only MCP stdio adapter for SeekLink v0.7.
Branch: codex/mcp-v07-adapter
Next: Run final release checks, publish v0.7.0, then verify PyPI and GitHub release surfaces.
Open: None
Verify: `uv run python -m pytest tests/test_mcp_services.py tests/test_mcp_integration.py -q`; `uv run python -m pytest tests/ -q`; MCP stdio smoke against copied `tests/corpus`; `uv build --out-dir /private/tmp/seeklink-v070-dist`; `uvx twine check /private/tmp/seeklink-v070-dist/*`; `git diff --check`.

## Handoff (2026-05-06, Codex, codex/mcp-v07-adapter)
Done: Implemented and hardened `seeklink[mcp]`, `seeklink mcp --vault PATH`, read-only `search/get/status/doctor` MCP tools, warm embedder/reranker reuse, compact search text summaries, docs, v0.7.0 changelog/version, schema disclosure, path safety, output caps, stdout guards, MCP tests, and release discoverability copy.
Open: None
Next: Run final release checks, publish v0.7.0, then verify PyPI and GitHub release surfaces.
Status: ready for release
