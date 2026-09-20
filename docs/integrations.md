# Use it from your coding agent

Everything here talks to one endpoint on an arbiter server you are already running. Start it first
and check it:

```bash
curl -s localhost:8010/readyz
# {"status":"ready","models":["english","multilingual","typed-decisions"],...}
```

The bridge is an MCP server in [`../integrations/mcp/arbiter_mcp.py`](../integrations/mcp/arbiter_mcp.py):
one file, the official Python SDK, stdio transport.

```bash
pip install "mcp>=2"
# or, to get a `arbiter-mcp` command on your PATH:
pip install ./integrations/mcp
```

It exposes five tools:

| Tool | Answers |
|---|---|
| `arbiter_check` | a probability that a statement about the state is true |
| `arbiter_classify` | one of up to 12 labelled options, with the distribution |
| `arbiter_score` | a position on a 2-10 level scale |
| `arbiter_gate` | `allow` / `confirm` / `block` for a proposed action, plus the five signals |
| `arbiter_decide` | any set of questions at once -- one forward pass for all of them |

Every tool returns MCP `structuredContent` (the numbers) plus one short line of text (what it
decided), and refuses input it cannot answer well: more than twelve options, a score with one
level, an empty state.

Configuration is three environment variables everywhere:

| Variable | Default | |
|---|---|---|
| `ARBITER_URL` | `http://localhost:8010` | where the server is |
| `ARBITER_API_KEY` | unset | sent as `Authorization: Bearer …` |
| `ARBITER_MODEL` | `auto` | or `laya-english`, `laya-multilingual`, `laya-typed-decisions` |

Below, `/ABS/PATH` is the absolute path to your checkout.

---

## Claude Code

One command for the tools alone:

```bash
claude mcp add arbiter --env ARBITER_URL=http://localhost:8010 -- python3 /ABS/PATH/integrations/mcp/arbiter_mcp.py
```

Or install the whole plugin in [`../integrations/claude-code/`](../integrations/claude-code),
which adds two things the bare MCP server does not have:

```bash
claude plugin validate /ABS/PATH/integrations/claude-code     # optional, checks the manifest
/plugin                                                       # then: install from a local path
```

* **A skill**, `skills/arbiter-decisions/SKILL.md`: when to call which tool, how to shape the state
  and the questions, the option budget, and why the thresholds belong in your code.
* **A `PreToolUse` hook on `Bash`**, `hooks/guard.py`: every shell command the agent proposes is
  sent to `/v1/systemone` with the five tool-call-guard questions, and the hook answers with
  `{"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "allow"|"ask"|"deny",
  "permissionDecisionReason": "…"}}`.

The hook's behaviour is deliberate and worth reading before you install it:

| Variable | Default | Effect |
|---|---|---|
| `ARBITER_GUARD_TIMEOUT` | `3` | seconds to wait for the server |
| `ARBITER_GUARD_FAIL_CLOSED` | unset | **fail-open by default**: if the server cannot be reached the hook logs to stderr and stays out of the way. Set to `1` to have it `ask` instead. A decision service that is down should not silently stop your session -- but it should not silently approve either, and that is the switch. |
| `ARBITER_GUARD_NO_AUTO_ALLOW` | unset | by default a clean verdict emits `"allow"`, which **skips your normal permission prompt**. Set to `1` and a clean verdict emits nothing instead, so the usual permission flow still runs and the hook can only ever add friction, never remove it. |
| `ARBITER_GUARD_LOG` | unset | append one line per decision to this file |

The plugin's [`.mcp.json`](../integrations/claude-code/.mcp.json) points at
`${CLAUDE_PLUGIN_ROOT}/../mcp/arbiter_mcp.py`, which resolves inside this checkout. If you copy the
plugin directory somewhere else, either copy `integrations/mcp/arbiter_mcp.py` along with it and
fix the path, or `pip install ./integrations/mcp` and change the command to `arbiter-mcp`.

## Codex

```bash
codex mcp add arbiter --env ARBITER_URL=http://localhost:8010 -- python3 /ABS/PATH/integrations/mcp/arbiter_mcp.py
codex mcp list
```

Or edit `~/.codex/config.toml` (user-wide) or `.codex/config.toml` (one project) with the block
in [`../integrations/codex/config.toml`](../integrations/codex/config.toml). Codex has no
`PreToolUse` equivalent, so the way to get the guard is to tell the agent in `AGENTS.md` to call
`arbiter_gate` before anything destructive.

## OpenCode

Merge the `mcp` block from
[`../integrations/opencode/opencode.json`](../integrations/opencode/opencode.json) into your
`opencode.json`, or into `~/.config/opencode/opencode.json` for every project. Local servers use
`type: "local"` and a `command` **array** that holds the command and its arguments together --
there is no separate `args` key.

## omp (oh-my-pi)

Copy [`../integrations/omp/.omp/mcp.json`](../integrations/omp/.omp/mcp.json) to `.omp/mcp.json`
in your project, or `~/.omp/agent/mcp.json` for all of them.

You may not need to. omp also imports MCP servers from other tools' configuration: a root
`.mcp.json`, `~/.claude.json` and `.claude/.mcp.json`, Codex's `~/.codex/config.toml`,
OpenCode's `opencode.json`, Cursor, Windsurf and VS Code. Set arbiter up for one of those and omp
finds it with no second file.

## Any other MCP client

[`../integrations/generic/.mcp.json`](../integrations/generic/.mcp.json) in your project root is
the format most clients read. Nothing in the server is client-specific.

To check the server starts at all, run it in a terminal:

```bash
ARBITER_URL=http://localhost:8010 python3 integrations/mcp/arbiter_mcp.py
```

It will sit there waiting for JSON-RPC on stdin. That is a healthy start.

---

## CI: gate pull requests on the diff

[`../integrations/ci/pr_risk_gate.yml`](../integrations/ci/pr_risk_gate.yml) runs
`examples/pr_risk_gate.py` over the pull request's diff, writes the table into a sticky comment,
and fails the job on `block`. Copy it to `.github/workflows/pr-risk-gate.yml`.

It needs a runner that can reach the arbiter server -- a self-hosted one by default. Set the
repository variable `ARBITER_URL` and, if the server wants a key, the secret `ARBITER_API_KEY`. The
exit codes are the whole contract:

| Exit | Meaning | What the job does |
|---|---|---|
| 0 | allow | comment, pass |
| 1 | review | comment, pass with a warning annotation |
| 2 | block | comment, **fail** |
| 3 | the server was unreachable | warn and pass; set the variable `ARBITER_FAIL_CLOSED=1` to fail instead |

The same script works as a local pre-push check with no CI at all:

```bash
git diff main | python examples/pr_risk_gate.py || echo "worth a second look"
```
