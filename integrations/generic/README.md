# Any other MCP client

`.mcp.json` in the project root is the format most MCP clients read, and the one omp, Cursor and
Claude Code all accept as a fallback. Copy [`.mcp.json`](.mcp.json) to the root of your project
and replace `/ABS/PATH`.

Nothing in the server is client-specific: it speaks MCP over stdio and is configured entirely
through three environment variables.

| Variable | Default | What it does |
|---|---|---|
| `ARBITER_URL` | `http://localhost:8010` | where the arbiter server is |
| `ARBITER_API_KEY` | unset | sent as `Authorization: Bearer …` when the server requires a key |
| `ARBITER_MODEL` | `auto` | pin a checkpoint: `laya-english`, `laya-multilingual`, `laya-typed-decisions` |
| `ARBITER_TIMEOUT` | `30` | seconds to wait for a response |

To check the server outside any client:

```bash
ARBITER_URL=http://localhost:8010 python3 integrations/mcp/arbiter_mcp.py
```

It will sit there waiting for JSON-RPC on stdin, which means it started cleanly.
