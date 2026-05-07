# Current Focus

Task: Add minimal Glama listing support.
Branch: main
Next: Push Dockerfile and Glama metadata, then submit SeekLink to Glama.
Open: None
Verify: `git diff --check`; `jq . glama.json`; `uv run python -m pytest tests/test_mcp_services.py tests/test_mcp_integration.py -q`; `docker build -t seeklink-mcp-glama .`; MCP `tools/list` smoke through Docker.

## Handoff (2026-05-06, Codex, main)
Done: Published v0.7.0 to PyPI and GitHub Release; added minimal Dockerfile and Glama metadata for MCP directory introspection checks.
Open: Submit `https://github.com/simonsysun/seeklink` to Glama and wait for the score badge path.
Next: Verify Dockerfile, push it to GitHub, then use the Glama score badge in the awesome-mcp-servers PR.
Status: in progress

## Handoff (2026-05-06, Codex, codex/mcp-v07-adapter)
Done: Implemented and hardened `seeklink[mcp]`, `seeklink mcp --vault PATH`, read-only `search/get/status/doctor` MCP tools, warm embedder/reranker reuse, compact search text summaries, docs, v0.7.0 changelog/version, schema disclosure, path safety, output caps, stdout guards, MCP tests, and release discoverability copy.
Open: None
Next: Run final release checks, publish v0.7.0, then verify PyPI and GitHub release surfaces.
Status: ready for release
