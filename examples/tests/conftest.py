import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

# The example scripts are run as scripts (`python examples/support_triage.py`), so they import
# `laya_client` and `_cli` as top-level modules. Tests import them the same way.
for path in (HERE, os.path.join(ROOT, "examples"), os.path.join(ROOT, "integrations", "mcp")):
    if path not in sys.path:
        sys.path.insert(0, path)
