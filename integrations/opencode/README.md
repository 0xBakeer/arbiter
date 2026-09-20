# OpenCode

Merge the `mcp` block from [`opencode.json`](opencode.json) into your `opencode.json` -- the one
in the project root, or `~/.config/opencode/opencode.json` for every project -- and replace
`/ABS/PATH` with your checkout.

OpenCode's local-server fields are `type: "local"`, `command` (an array, command and arguments
together), `environment`, and `enabled`. There is no separate `args` key.

```json
{
  "mcp": {
    "laya": {
      "type": "local",
      "command": ["laya-mcp"],
      "environment": { "LAYA_URL": "http://localhost:8010" }
    }
  }
}
```

is the shorter form once `pip install /ABS/PATH/integrations/mcp` has put `laya-mcp` on your
PATH.

Every MCP server costs context, so five small tools is about the right budget. If you only want
the guard, there is no way to expose a subset from the config -- run a second copy of the server
behind a wrapper that only registers `laya_gate`, or just tell the agent which tool to use.
