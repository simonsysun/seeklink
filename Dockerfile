FROM python:3.12-slim

WORKDIR /app

# Build the MCP adapter from this repository. The canonical host install path
# remains `pip install "seeklink[mcp]"`; this image mainly exists so MCP server
# directories can start the stdio server and run introspection checks.
COPY pyproject.toml README.md LICENSE ./
COPY seeklink ./seeklink

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir ".[mcp]"

# Provide a minimal default vault so `seeklink mcp` starts without requiring a
# bind mount. Real use should mount a Markdown vault at /vault.
RUN mkdir -p /vault \
    && printf '# SeekLink stub vault\n' > /vault/README.md

ENV PYTHONUNBUFFERED=1
ENV SEEKLINK_VAULT=/vault

ENTRYPOINT ["seeklink", "mcp"]
