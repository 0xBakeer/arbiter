# Codex

One command, no file editing:

```bash
codex mcp add arbiter --env ARBITER_URL=http://localhost:8010 -- python3 /ABS/PATH/integrations/mcp/arbiter_mcp.py
```

Then `codex mcp list` to confirm it is there, and `/mcp` inside the TUI to see it connected.

To configure it by hand instead, copy the block in [`config.toml`](config.toml) into
`~/.codex/config.toml` (user-wide) or `.codex/config.toml` (this project only). Codex reads
`[mcp_servers.<name>]` tables with `command`, `args` and `env`.

If the server needs a key, add it to `env`:

```toml
env = { ARBITER_URL = "http://localhost:8010", ARBITER_API_KEY = "…" }
```

The five tools then show up as `arbiter_classify`, `arbiter_score`, `arbiter_check`, `arbiter_gate` and
`arbiter_decide`. Codex has no PreToolUse hook equivalent, so the `arbiter_gate` tool is the way to
get the same guard: tell the agent, in `AGENTS.md`, to call it before anything destructive.
