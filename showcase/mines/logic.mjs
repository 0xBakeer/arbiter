// Minesweeper, asked as a batch. Every tick sends up to twelve independent nouls - one per
// frontier cell, each carrying its own clues in its own instructions - and the server answers
// all of them in a single forward pass. The board then opens the cell with the highest "safe".
// Ground truth is exact here, so the calibration table in the README is the real thing.

import { seedStreams, rnd, prnd } from "../_lib/prng.mjs";

export const meta = {
  id: "mines",
  name: "Mines",
  tick_ms: 250,
  max_steps: 60,
  description: "Up to twelve safety nouls per tick, one forward pass, the safest cell is opened.",
  actions: ["open a cell"],
  shield: "none",
  // Every tick asks the same question about twelve different cells, so the calibration table
  // pools them all instead of reporting sixty tables of four answers each.
  calibration_group: "cell_is_safe",
};

const W = 8;
const H = 8;
const MINES = 10;
const MAX_QUESTIONS = 12;
const OPEN_R = 3;
const OPEN_C = 3;

const idx = (r, c) => r * W + c;
export const cellId = (r, c) => `r${r}c${c}`;
const parseCell = (id) => {
  const m = /^r(\d+)c(\d+)$/.exec(id);
  return { r: Number(m[1]), c: Number(m[2]) };
};

function neighbours(r, c) {
  const out = [];
  for (let dr = -1; dr <= 1; dr += 1) {
    for (let dc = -1; dc <= 1; dc += 1) {
      if (!dr && !dc) continue;
      const nr = r + dr;
      const nc = c + dc;
      if (nr >= 0 && nc >= 0 && nr < H && nc < W) out.push([nr, nc]);
    }
  }
  return out;
}

export function init(seed) {
  const state = {
    w: W,
    h: H,
    mines: new Array(W * H).fill(false),
    revealed: new Array(W * H).fill(false),
    counts: new Array(W * H).fill(0),
    total_mines: MINES,
    score: 0,
    steps: 0,
    alive: true,
    cleared: false,
    ...seedStreams(seed),
  };
  // The opening cell and its ring are always safe, the way a first click is in every
  // minesweeper: otherwise a third of the seeds would end on tick one.
  const banned = new Set([idx(OPEN_R, OPEN_C), ...neighbours(OPEN_R, OPEN_C).map(([r, c]) => idx(r, c))]);
  let placed = 0;
  while (placed < MINES) {
    const at = Math.floor(rnd(state) * W * H);
    if (banned.has(at) || state.mines[at]) continue;
    state.mines[at] = true;
    placed += 1;
  }
  for (let r = 0; r < H; r += 1) {
    for (let c = 0; c < W; c += 1) {
      state.counts[idx(r, c)] = neighbours(r, c).filter(([nr, nc]) => state.mines[idx(nr, nc)]).length;
    }
  }
  reveal(state, OPEN_R, OPEN_C);
  return state;
}

// Opening a zero opens its neighbours too, as in the original game.
function reveal(state, r, c) {
  const queue = [[r, c]];
  let opened = 0;
  while (queue.length) {
    const [cr, cc] = queue.pop();
    const at = idx(cr, cc);
    if (state.revealed[at]) continue;
    state.revealed[at] = true;
    opened += 1;
    if (state.counts[at] === 0) queue.push(...neighbours(cr, cc));
  }
  state.score += opened;
  return opened;
}

const unrevealed = (state) => {
  const out = [];
  for (let r = 0; r < H; r += 1) for (let c = 0; c < W; c += 1) if (!state.revealed[idx(r, c)]) out.push([r, c]);
  return out;
};

// Unrevealed cells that touch a number: the only cells there is any evidence about.
function frontier(state) {
  return unrevealed(state).filter(([r, c]) =>
    neighbours(r, c).some(([nr, nc]) => state.revealed[idx(nr, nc)]));
}

export function candidates(state) {
  const front = frontier(state);
  return (front.length ? front : unrevealed(state)).slice(0, MAX_QUESTIONS);
}

// The clues touching one cell, and the tightest of them: "1 mine among 3 unknown cells".
function evidence(state, r, c) {
  const clues = [];
  for (const [nr, nc] of neighbours(r, c)) {
    const at = idx(nr, nc);
    if (!state.revealed[at]) continue;
    const unknown = neighbours(nr, nc).filter(([ur, uc]) => !state.revealed[idx(ur, uc)]).length;
    clues.push({ n: state.counts[at], unknown, ratio: unknown ? state.counts[at] / unknown : 0 });
  }
  clues.sort((a, b) => b.ratio - a.ratio);
  return clues;
}

// The naive estimate every minesweeper player makes in their head: of the clues touching this
// cell, the one with the least room to spread its mines. With no clue at all, the density of
// mines over the unknown cells. This number is a fact about the board, not a solution: the
// single-point deductions that actually clear a board are never applied here.
export function naiveRisk(state, r, c) {
  const clues = evidence(state, r, c);
  if (!clues.length) return state.total_mines / Math.max(1, unrevealed(state).length);
  return Math.min(1, clues[0].ratio);
}

const percent = (p) => Math.round(p * 100);

// -- the contract ------------------------------------------------------------

// Any unrevealed cell can be opened. The questions only cover the twelve with evidence, but the
// heuristic is allowed to open a cell it deduced somewhere else on the board.
export function legal(state) {
  return unrevealed(state).map(([r, c]) => `open ${cellId(r, c)}`);
}

export function encode(state) {
  const unknown = unrevealed(state).length;
  return [
    `Minesweeper on an 8 by 8 board with ${MINES} mines.`,
    `${state.score} cells are open and ${unknown} are still unknown.`,
    "Each question below asks about one unknown cell and carries that cell's own clues.",
  ].join(" ");
}

export function questions(state) {
  const out = {};
  for (const [r, c] of candidates(state)) {
    const id = cellId(r, c);
    // Spelling the clue out as a percentage beats spelling it out as "a 3 with 4 unknown
    // neighbours": measured over 216 labelled cells, AUROC 0.79 against 0.46 for the long form,
    // and a third fewer tokens. See the honesty note in the README about what that means.
    out[id] = {
      type: "noul",
      instructions: `Cell ${id}: the clues around it put the chance of a mine at about `
        + `${percent(naiveRisk(state, r, c))} percent. This cell is safe to open.`,
      criteria: { true: "the chance of a mine is low", false: "the chance of a mine is high" },
    };
  }
  return out;
}

export function decide(state, answers) {
  let best = null;
  const scores = {};
  for (const [r, c] of candidates(state)) {
    const id = cellId(r, c);
    const p = answers[id] ? answers[id].noul : 0;
    scores[id] = p;
    if (!best || p > best.p) best = { id, p };
  }
  return { action: `open ${best.id}`, extras: { safe: scores, picked: best.id, confidence: best.p } };
}

// No shield: the board opens whatever the model calls safest, and the mine is the consequence.
export function shield(state, action) {
  return { action, intervened: false, reason: "" };
}

export function step(state, action) {
  const next = structuredClone(state);
  const events = [];
  const { r, c } = parseCell(action.replace("open ", ""));
  next.steps += 1;
  if (next.mines[idx(r, c)]) {
    next.revealed[idx(r, c)] = true;
    next.alive = false;
    events.push(`opened a mine at ${cellId(r, c)}`);
    return { state: next, reward: -1, done: true, events };
  }
  const opened = reveal(next, r, c);
  events.push(`opened ${opened} cell${opened === 1 ? "" : "s"} at ${cellId(r, c)}`);
  if (next.score >= W * H - MINES) {
    next.cleared = true;
    events.push("board cleared");
    return { state: next, reward: opened, done: true, events };
  }
  return { state: next, reward: opened, done: false, events };
}

export function random(state) {
  const cells = candidates(state);
  const [r, c] = cells[Math.floor(prnd(state) * cells.length)];
  return `open ${cellId(r, c)}`;
}

// Single-point solving: a clue whose number equals its unknown neighbours marks them all as
// mines, and a clue with no mines left marks its neighbours safe. Anything else takes the cell
// with the loosest clue.
export function heuristic(state) {
  const known = new Set();
  for (let r = 0; r < H; r += 1) {
    for (let c = 0; c < W; c += 1) {
      const at = idx(r, c);
      if (!state.revealed[at] || state.counts[at] === 0) continue;
      const unknown = neighbours(r, c).filter(([nr, nc]) => !state.revealed[idx(nr, nc)]);
      if (unknown.length === state.counts[at]) unknown.forEach(([nr, nc]) => known.add(idx(nr, nc)));
    }
  }
  for (let r = 0; r < H; r += 1) {
    for (let c = 0; c < W; c += 1) {
      const at = idx(r, c);
      if (!state.revealed[at]) continue;
      const unknown = neighbours(r, c).filter(([nr, nc]) => !state.revealed[idx(nr, nc)]);
      const flagged = unknown.filter(([nr, nc]) => known.has(idx(nr, nc))).length;
      if (flagged === state.counts[at]) {
        const safe = unknown.find(([nr, nc]) => !known.has(idx(nr, nc)));
        if (safe) return `open ${cellId(safe[0], safe[1])}`;
      }
    }
  }
  const cells = candidates(state).filter(([r, c]) => !known.has(idx(r, c)));
  const pool = cells.length ? cells : candidates(state);
  const ranked = pool.slice().sort((a, b) => naiveRisk(state, a[0], a[1]) - naiveRisk(state, b[0], b[1]));
  return `open ${cellId(ranked[0][0], ranked[0][1])}`;
}

// The number the questions were handed, turned back into a "safe" score, so measure.mjs can
// report whether the model's ranking beats simply trusting that number.
export function baseline(state) {
  const out = {};
  for (const [r, c] of candidates(state)) out[cellId(r, c)] = 1 - naiveRisk(state, r, c);
  return out;
}

// One label per question: whether that cell really is safe.
export function truth(state) {
  const out = {};
  for (const [r, c] of candidates(state)) out[cellId(r, c)] = !state.mines[idx(r, c)];
  return out;
}

export function summary(state) {
  return {
    score: state.score,
    steps: state.steps,
    alive: state.alive,
    cleared: state.cleared,
    unknown: unrevealed(state).length,
  };
}
