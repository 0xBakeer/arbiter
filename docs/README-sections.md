<!--
Drafted README sections for the orchestrator to merge into README.md.
Two sections: "Examples" and "Use it from your coding agent". 57 lines of body.
-->

## Examples

Nine runnable scripts in [`examples/`](examples), each one a real decision with the thresholds
in the caller and a review band in the middle. They need nothing but a Python 3 and a running
server: [`examples/laya_client.py`](examples/laya_client.py) is a single dependency-free file
whose API mirrors the hosted SDK, so code written against Jev ports by changing the import and
the base URL.

| Use case | Script | What it decides | Route |
|---|---|---|---|
| Support tickets | `support_triage.py` | department, urgency, frustration, refund, churn | `auto` · `review` · `escalate` |
| Inbox triage | `email_triage.py` | category, phishing, action needed, reply-by | `auto` · `review` · `escalate` · `block` |
| Agent shell commands | `tool_call_guard.py` | destructive, secrets, leaves repo, network, blast radius | `allow` · `ask` · `deny` |
| Pull requests | `pr_risk_gate.py` | credentials, migration, shared infra, rollback, blast radius | `allow` · `review` · `block` |
| Monitoring alerts | `alert_triage.py` | service, root cause, severity, duplicate of an open incident | `suppress` · `ticket` · `page` |
| Model selection | `model_router.py` | complexity, needs tools, needs long context | `fast` · `cascade` · `powerful` |
| RAG passages | `rag_relevance.py` | per-passage relevance, hierarchical past 12 | `keep` · `?` · `drop` |
| Message screening | `moderation.py` | jailbreak, harmful, PII, off-topic, severity | `allow` · `review` · `block` |
| Invoices (typed-decisions) | `invoice_fields.py` | currency, amount band, duplicate, approval, bank change | `pay` · `approve` · `review` |

```bash
python examples/support_triage.py                 # a built-in sample
python examples/email_triage.py --sample de       # watch the router pick multilingual
git diff main | python examples/pr_risk_gate.py   # exit code is the gate: 0/1/2
python examples/moderation.py --state msg.txt --json
```

Each script prints one row per question with a probability bar, the checkpoint that answered and
the latency, then the route its own thresholds chose. Captured output for every one of them is
in [`docs/use-cases.md`](docs/use-cases.md); the shapes they are built from -- fan-out,
confidence-gated routing, composite scoring, hierarchical intent, cascade -- are in
[`docs/patterns.md`](docs/patterns.md).

## Use it from your coding agent

[`integrations/mcp/laya_mcp.py`](integrations/mcp/laya_mcp.py) is an MCP server over the same
endpoint: `laya_check`, `laya_classify`, `laya_score`, `laya_gate` and `laya_decide`, each
returning structured content plus one line of text. Point any MCP client at it:

```bash
pip install "mcp>=2"
claude mcp add laya --env LAYA_URL=http://localhost:8010 -- python3 $PWD/integrations/mcp/laya_mcp.py
codex  mcp add laya --env LAYA_URL=http://localhost:8010 -- python3 $PWD/integrations/mcp/laya_mcp.py
```

[`integrations/claude-code/`](integrations/claude-code) is a plugin that adds a skill (when to
use which primitive, how to shape state and questions, why thresholds belong in your code) and a
`PreToolUse` hook that judges every `Bash` command before it runs -- allow, ask or deny, in tens
of milliseconds, failing open when the server is not there. Ready-made configuration for Codex,
OpenCode, omp and any generic `.mcp.json` client, plus a GitHub Actions job that gates pull
requests, is in [`integrations/`](integrations) and documented in
[`docs/integrations.md`](docs/integrations.md).
