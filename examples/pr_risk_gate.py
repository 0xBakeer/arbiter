#!/usr/bin/env python3
"""Gate a pull request on its diff: allow it, ask for a review, or block the merge.

Reads a unified diff on stdin, so it drops straight into a pre-push hook or a CI job:

    git diff main | python examples/pr_risk_gate.py
    git diff main | python examples/pr_risk_gate.py --description "$(git log -1 --pretty=%B)"

The exit code is the gate: 0 allow, 1 review, 2 block. `integrations/ci/pr_risk_gate.yml` is a
GitHub Actions job built on exactly that.

A diff is long and the model reads a fixed window of it, so the description and the file list
are put first in the state where they are certain to be read.
"""
import subprocess
import sys

import _cli
from arbiter_client import ArbiterClient, Noul, Score

QUESTIONS = {
    "touches_prod_credentials": Noul(
        "This change adds, moves or exposes production credentials: secrets, API keys, tokens, "
        "certificates, connection strings, or the configuration that holds them."),
    "touches_migration": Noul(
        "This change contains a database migration or otherwise alters a schema in place.",
        {"true": "a migration file, an ALTER/DROP, a column or index change",
         "false": "application code only"},
    ),
    "touches_shared_infra": Noul(
        "This change alters infrastructure other teams depend on: CI pipelines, Terraform, "
        "Kubernetes manifests, base images, shared libraries, build configuration."),
    "no_rollback_plan": Noul(
        "This change would be hard to roll back: it is irreversible, destroys data, or the "
        "description gives no way to undo it.",
        {"true": "no stated rollback, or one that cannot work after the fact",
         "false": "reversible by reverting the commit, or a rollback is described"},
    ),
    "blast_radius": Score(
        "If this change is wrong, how far does the damage reach?",
        ["one isolated file or test",
         "one service or module",
         "several services that call each other",
         "every user of the product"],
    ),
}

BLOCK_CREDENTIALS = 0.65     # credentials in a diff are never a judgement call
BLOCK_BLAST = 2.60
REVIEW_SIGNAL = 0.45
REVIEW_BLAST = 1.50

SAMPLE_DESCRIPTION = "Speed up the nightly rollup and drop the column it no longer needs"
SAMPLE_DIFF = """diff --git a/db/migrations/0042_drop_legacy_totals.sql b/db/migrations/0042_drop_legacy_totals.sql
new file mode 100644
--- /dev/null
+++ b/db/migrations/0042_drop_legacy_totals.sql
@@
+ALTER TABLE orders DROP COLUMN legacy_total_cents;
+DROP INDEX IF EXISTS idx_orders_legacy_total;
diff --git a/deploy/k8s/rollup-cron.yaml b/deploy/k8s/rollup-cron.yaml
--- a/deploy/k8s/rollup-cron.yaml
+++ b/deploy/k8s/rollup-cron.yaml
@@
-  schedule: "0 3 * * *"
+  schedule: "*/15 * * * *"
-      concurrencyPolicy: Forbid
+      concurrencyPolicy: Allow
diff --git a/services/rollup/query.py b/services/rollup/query.py
--- a/services/rollup/query.py
+++ b/services/rollup/query.py
@@
-    rows = db.fetch_all(SELECT_ORDERS, since=since)
+    rows = db.fetch_all(SELECT_ORDERS_FAST, since=since)
"""


def files_in(diff: str):
    return [line.split(" b/", 1)[1].strip()
            for line in diff.splitlines() if line.startswith("diff --git ") and " b/" in line]


def build_state(diff: str, description: str) -> dict:
    return {"description": description or "(no description given)",
            "files_changed": files_in(diff),
            "diff": diff}


def decide(r):
    """allow | review | block."""
    creds = r.noul("touches_prod_credentials")
    migration = r.noul("touches_migration")
    infra = r.noul("touches_shared_infra")
    no_rollback = r.noul("no_rollback_plan")
    blast = r.score("blast_radius")

    if creds >= BLOCK_CREDENTIALS:
        return "block", "touches_prod_credentials %.2f >= %.2f" % (creds, BLOCK_CREDENTIALS)
    if no_rollback >= 0.60 and blast >= 2.0:
        return "block", "no rollback plan (%.2f) with blast_radius %.2f" % (no_rollback, blast)
    if blast >= BLOCK_BLAST:
        return "block", "blast_radius %.2f >= %.2f" % (blast, BLOCK_BLAST)
    hot = {k: v for k, v in (("credentials", creds), ("migration", migration),
                             ("shared infra", infra), ("no rollback", no_rollback))
           if v >= REVIEW_SIGNAL}
    if hot or blast >= REVIEW_BLAST:
        detail = ", ".join("%s %.2f" % kv for kv in sorted(hot.items(), key=lambda kv: -kv[1]))
        return "review", (detail or "blast_radius %.2f >= %.2f" % (blast, REVIEW_BLAST))
    return "allow", "nothing above %.2f, blast_radius %.2f" % (REVIEW_SIGNAL, blast)


def read_diff(args) -> str:
    if args.state and args.state != "-":
        return open(args.state, encoding="utf-8").read()
    if args.state == "-" or not sys.stdin.isatty():
        piped = sys.stdin.read()
        if piped.strip():
            return piped
    if args.git_diff:
        return subprocess.run(["git", "diff", args.git_diff], capture_output=True,
                              text=True, check=True).stdout
    return SAMPLE_DIFF


def main() -> int:
    p = _cli.parser(__doc__)
    p.add_argument("--description", default=SAMPLE_DESCRIPTION,
                   help="the PR title and body, read before the diff")
    p.add_argument("--git-diff", metavar="REF",
                   help="run `git diff REF` instead of reading stdin")
    args = p.parse_args()
    diff = read_diff(args)
    state = build_state(diff, args.description)
    response = ArbiterClient(base_url=args.url).system_one(state, QUESTIONS, model=args.model)
    if args.json:
        return _cli.dump(response)
    action, reason = decide(response)
    _cli.render("PR risk gate", response, action, reason,
                extra=["%s%d files: %s%s" % (_cli.DIM, len(state["files_changed"]),
                                             ", ".join(state["files_changed"])[:60], _cli.RESET)])
    return {"allow": 0, "review": 1, "block": 2}[action]


if __name__ == "__main__":
    sys.exit(_cli.run(main))
