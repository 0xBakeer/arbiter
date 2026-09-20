# omp (oh-my-pi)

Copy [`.omp/mcp.json`](.omp/mcp.json) into your project as `.omp/mcp.json`, or into
`~/.omp/agent/mcp.json` to have it everywhere, and replace `/ABS/PATH` with your checkout.
`stdio` is the default transport, so `type` is optional.

You may not need to do even that. omp discovers MCP servers from other tools' configuration as
well: a root `.mcp.json` (see [`../generic/.mcp.json`](../generic/.mcp.json)), `~/.claude.json`
and project `.claude/.mcp.json`, Codex's `~/.codex/config.toml`, OpenCode's `opencode.json`,
Cursor, Windsurf and VS Code. If you have already set arbiter up for one of those, omp will pick it
up with no second config file.

Project entries are seen before the same-named user entry, so a project `"enabled": false`
switches off a user-level `arbiter` for this repository only.

Add the server interactively with `/mcp add`, list what was discovered with `/mcp`.
