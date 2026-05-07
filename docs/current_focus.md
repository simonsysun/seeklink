# Current Focus

Task: Track Glama listing review.
Branch: main
Next: Wait for Glama to publish the SeekLink server page and score badge, then add the badge to the awesome-mcp-servers PR.
Open: Glama submission is under review; public page/API currently return 404 until Glama finishes review or scanning.
Verify: `git diff --check`; `jq . glama.json`; `uv run python -m pytest tests/test_mcp_services.py tests/test_mcp_integration.py -q`; `docker build -t seeklink-mcp-glama .`; MCP `tools/list` smoke through Docker.

## Handoff (2026-05-06, Codex, main)
Done: Published v0.7.0 to PyPI and GitHub Release; added and pushed minimal Dockerfile and Glama metadata; submitted `https://github.com/simonsysun/seeklink` to Glama.
Open: Glama review/scanning has not published the server page or score badge yet.
Next: Recheck `https://glama.ai/mcp/servers/simonsysun/seeklink`; once live, use the score badge in the awesome-mcp-servers PR.
Status: blocked

## Handoff (2026-05-06, Codex, codex/mcp-v07-adapter)
Done: Implemented and hardened `seeklink[mcp]`, `seeklink mcp --vault PATH`, read-only `search/get/status/doctor` MCP tools, warm embedder/reranker reuse, compact search text summaries, docs, v0.7.0 changelog/version, schema disclosure, path safety, output caps, stdout guards, MCP tests, and release discoverability copy.
Open: None
Next: Run final release checks, publish v0.7.0, then verify PyPI and GitHub release surfaces.
Status: ready for release
