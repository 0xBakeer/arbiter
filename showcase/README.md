# arbiter plays

Six small games the Laya typed-decision model plays in real time, in the browser, against this
repository's own server. Every move is one `POST /v1/systemone`: the game writes the position down
in words, asks a typed question about it, and plays the answer. No sampling, no tokens generated,
no second call to check the first one.

Open `/showcase/` on a running server (`./run.sh serve`, then <http://localhost:8010/showcase/>),
or open a game folder from disk in Safari or Firefox to watch the recorded run with no server at
all.

## The games

<!-- index-table -->
| game | what the model decides | questions per tick | tick | decisions/s | p50 inference | model | random | reference |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [snake](snake/) | a direction, plus two nouls | 3 | 120 ms | 15.7 | 61.6 ms | **1.35** | 1.2 | 26.45 |
| [hopper](hopper/) | one noul: flap or glide | 1 | 100 ms | 32.5 | 29.1 ms | **13.4** | 0 | 26 |
| [crossing](crossing/) | one of five moves, plus a hit-risk score | 2 | 150 ms | 16.4 | 58.2 ms | **5.6** | 0.6 | 7.25 |
| [paddle](paddle/) | left, stay or right | 1 | 60 ms | 32.4 | 29.1 ms | **4.6** | 0.15 | 12 |
| [mines](mines/) | which of up to twelve cells is safe | 12 | 250 ms | 7.2 | 137.3 ms | **38.4** | 32.65 | 50.05 |
| [dungeon](dungeon/) | one of three to five actions, plus a danger score | 2 | 500 ms | 15.8 | 60.7 ms | **139.9** | 256.65 | 396.75 |

The last three columns are the mean score over 20 episodes for the best model policy, for the random
baseline and for the hand-written one, all behind the same shield where the game has one. They are not
comparable across games — a snake apple is not a dungeon florin — but they are comparable down a column.
`mines` makes only 7.2 decisions a second because each decision carries twelve questions: that is 87 typed
answers a second out of one forward pass.
Each game's README has the full table, the per-policy timings and the calibration against ground truth.

Open a game with `./run.sh serve` running and go to `/showcase/<game>/`; press `M` to switch between the
model and the hand-written policy, `S` to switch the shield off, space to pause, the arrow keys to change
speed. With no server the page plays the recorded run from `replay/<game>.json` instead.
<!-- index-table -->

## How the harness works

One loop, shared by the browser pages and by `measure.mjs`, in `_lib/harness.mjs`:

```
encode(state) -> questions(state) -> one POST /v1/systemone -> decide(state, answers)
              -> shield(state, action, extras) -> step(state, action) -> repeat
```

`encode` turns the position into a short piece of text. `questions` builds the typed questions, and
this is where the craft is: the options of a choice are described by what each one leads to, in the
same words a person would use ("food gets closer, 5 steps to the wall, open space beyond"). The
model answers all of them in a single forward pass. `decide` takes the argmax of the choice (or the
noul's side of 0.5) and hands back the probabilities in `extras`. `shield` is plain code: it may
veto an action that ends the episode when a survivable one exists, and when it does it takes the
model's *next* choice by probability, so the ranking is still the model's. Every intervention is
counted and reported. `step` is a pure function of `(state, action)`.

A policy is who answers: `model`, `model+shield`, `random+shield`, `heuristic+shield`. The two
baselines exist so that a score means something. They draw from a separate random stream inside the
state, so the same seed builds the same world whichever policy plays it.

`_lib/client.mjs` is the only code that touches the network, and it reports two timings that do not
mean the same thing: `inference_ms` is the server's own `latency_ms`, `roundtrip_ms` is what the
caller waited for. Everything called "decisions per second" in this directory is derived from the
round trip.

## Measuring

```
node showcase/measure.mjs --game snake --episodes 20 \
     --policies model+shield,random+shield,heuristic+shield \
     --base http://localhost:8010
```

With `--game all` it runs every game, writes `<game>/measurements.json`, replaces the table between
the `<!-- measurements -->` markers in each game README, and saves the best model episode to
`replay/<game>.json` so the page can play it without a server. `--render-only` rebuilds the tables
from the JSON already on disk. The whole sweep is about fifteen minutes on a Mac.

Where a question has ground truth in the game's own code — `dead_end` in snake, the twelve cell
nouls in mines, `hit_risk` in crossing, `danger` in dungeon — the measurement also reports AUROC
and the confusion table at the 0.5 threshold. Ground truth is computed by `truth(state, action)` in
`logic.mjs` and is never shown to the model.

## Adding a game

A game is a directory with `logic.mjs`, `index.html`, `README.md` and `measurements.json`.
`logic.mjs` is pure ESM that runs unchanged in node 22 and in the browser — no DOM, no `fetch`, no
imports beyond `_lib/prng.mjs` — and exports:

| export | what it does |
| --- | --- |
| `meta` | `{ id, name, tick_ms, max_steps, description, actions, shield }` |
| `init(seed)` | the starting state, with both random streams inside it |
| `encode(state)` | the position as text, short |
| `questions(state)` | the typed questions, criteria computed from this tick |
| `decide(state, answers)` | `{ action, extras }` — the argmax plus whatever the panel shows |
| `legal(state)` | the actions that may be played now |
| `shield(state, action, extras)` | `{ action, intervened, reason }` — code-owned safety, or a pass-through |
| `step(state, action)` | `{ state, reward, done, events }`, pure, never mutates its argument |
| `random(state)` / `heuristic(state)` | the two baselines |
| `summary(state)` | `{ score, steps, ... }` |
| `truth(state, action)` | optional: ground truth keyed by question id, for the calibration table |
| `baseline(state)` | optional: a number per question id to compare the model's ranking against |

Then register it in the `GAMES` table in `measure.mjs`, copy a page from another game and replace
the block at the bottom, and run `node --test showcase/tests/` — `contract.test.mjs` checks every
game against the list above, including that a choice never goes out with fewer than two options.

## The honest part

This is a showcase of speed and of typed decisions, not a claim about intelligence. The model is a
sentence encoder with typed heads; it reads criteria and returns calibrated numbers in tens of
milliseconds, and that is the whole of it. Where it plays at the level of a random baseline, the
tables say so with numbers. Where the shield does the surviving, the intervention count says so.
Where a question hands the model a number and it ranks by that number, the README says that too —
`mines` reports the AUROC of the naive estimate next to the model's own, and it is the better of
the two.

Two rules kept every measurement in this directory honest:

- the baselines and the model see the same worlds, from the same seeds;
- nothing reported here was produced by a mock. Every number comes from a run against a live
  server, and the server, the device and the date are recorded in each `measurements.json`.
