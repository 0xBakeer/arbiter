# Engines

One package per model. An engine subclasses `server.engine.Engine` and implements four methods:

| method | does |
|---|---|
| `load(name)` | bring checkpoint `name` into memory, ready to answer |
| `build_rows(name, state, questions)` | one opaque row per question, in the order asked |
| `predict_rows(name, rows)` | run a batch: an answer and a token count per row, and the path taken |
| `route(state, questions, model, task, lang)` | which checkpoint answers, and why |

Rows are whatever that engine needs; nothing outside it looks inside one. In exchange the engine
gets the worker thread per checkpoint, cross-request batching, the queue cap, the warm-up and
`/v1/systemone` for free. Expose a module-level `from_env()` that builds it from the
environment, and `ARBITER_ENGINE=<package>` selects it (default `laya`).
