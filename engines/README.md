# Engines

One package per model. An engine subclasses `server.engine.Engine` and implements four methods:

| method | does |
|---|---|
| `load(name)` | bring checkpoint `name` into memory, ready to answer |
| `build_rows(name, state, questions)` | one opaque row per question, in the order asked |
| `predict_rows(name, rows)` | run a batch: an answer and a token count per row, and the path taken |
| `route(state, questions, model, task, lang)` | which checkpoint answers, and why |

What is here:

| package | runs on | selected with |
|---|---|---|
| [`laya`](laya) | torch: CUDA, MPS or CPU. The default, and the one every recipe measures | nothing, or `ARBITER_ENGINE=laya` |
| [`laya_mlx`](laya_mlx) | MLX on Apple silicon, over the community port's converted checkpoints | `ARBITER_ENGINE=laya_mlx`, opt-in |

Both serve the same three checkpoints under the same names and route identically; what differs
is the arithmetic underneath and, therefore, the equivalence table each one owes.

Rows are whatever that engine needs; nothing outside it looks inside one. In exchange the engine
gets the worker thread per checkpoint, cross-request batching, the queue cap, the warm-up and
`/v1/systemone` for free. Expose a module-level `from_env()` that builds it from the
environment, and `ARBITER_ENGINE=<package>` selects it (default `laya`).
