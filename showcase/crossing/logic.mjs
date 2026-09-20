// Frogger. Nine lanes of traffic between two banks, five moves to choose from, and the criteria
// say what is in the target cell and how soon a car gets there. The shield refuses to step into
// a cell a car reaches this tick, so `model+shield` and `random+shield` differ only in who picks
// among the survivable moves.

import { seedStreams, rnd, prnd } from "../_lib/prng.mjs";

export const meta = {
  id: "crossing",
  name: "Crossing",
  tick_ms: 150,
  max_steps: 250,
  description: "Cross seven lanes of traffic. One choice of five, plus a three-level hit risk.",
  actions: ["wait", "forward", "back", "left", "right"],
  shield: "vetoes a step into a cell a car reaches this tick",
};

const W = 9;
const H = 9;                              // row 8 is the start bank, row 0 the goal bank
const LANES = [1, 2, 3, 4, 5, 6, 7];
const MOVES = { wait: [0, 0], forward: [0, -1], back: [0, 1], left: [-1, 0], right: [1, 0] };

export function init(seed) {
  const state = {
    w: W,
    h: H,
    x: 4,
    y: H - 1,
    lanes: [],
    score: 0,
    steps: 0,
    alive: true,
    ...seedStreams(seed),
  };
  for (const row of LANES) {
    const dir = rnd(state) < 0.5 ? -1 : 1;
    const period = rnd(state) < 0.4 ? 2 : 1;       // every tick, or every other tick
    const count = 2 + Math.floor(rnd(state) * 2);  // two or three cars in the lane
    const cars = [];
    while (cars.length < count) {
      const x = Math.floor(rnd(state) * W);
      if (!cars.includes(x)) cars.push(x);
    }
    state.lanes.push({ row, dir, period, cars: cars.sort((a, b) => a - b) });
  }
  return state;
}

// -- traffic -----------------------------------------------------------------

const laneAt = (state, y) => state.lanes.find((l) => l.row === y) || null;
const occupiedNow = (state, x, y) => {
  const lane = laneAt(state, y);
  return !!lane && lane.cars.includes(x);
};

// The one fact the shield acts on: stepping there is death, either because a car is already
// standing on the cell or because one rolls onto it before the next decision.
function deadly(state, x, y) {
  if (x < 0 || y < 0 || x >= W || y >= H) return true;
  const ticks = arrivesIn(state, x, y);
  return ticks !== null && ticks <= 1;
}

// Ticks until a car stands on (x, y): 0 means one is there now, 1 means one rolls onto it
// before the next decision. Simulated rather than divided, because half the lanes only move on
// every other tick and the phase changes the answer.
const HORIZON = 12;

function arrivesIn(state, x, y) {
  const lane = laneAt(state, y);
  if (!lane) return null;
  let cars = lane.cars;
  for (let t = 0; t < HORIZON; t += 1) {
    if (cars.includes(x)) return t;
    if ((state.steps + t) % lane.period === 0) cars = cars.map((c) => (c + lane.dir + W) % W);
  }
  return null;
}

export function legal(state) {
  return Object.keys(MOVES).filter((m) => {
    const [dx, dy] = MOVES[m];
    const x = state.x + dx;
    const y = state.y + dy;
    return x >= 0 && y >= 0 && x < W && y < H;
  });
}

const target = (state, move) => ({ x: state.x + MOVES[move][0], y: state.y + MOVES[move][1] });

// -- the contract ------------------------------------------------------------

export function encode(state) {
  const row = state.y;
  const place = row === H - 1 ? "on the start bank" : row === 0 ? "on the far bank" : `in lane ${H - 1 - row} of 7`;
  return [
    `Crossing seven lanes of traffic. You are ${place}, ${row} rows from the far side.`,
    `You have crossed ${state.score} times.`,
  ].join(" ");
}

function moveFacts(state, move) {
  const { x, y } = target(state, move);
  // Verdict first, then the progress, then one number. The long version of this text ("into lane
  // 3, traffic runs right to left, a car reaches that cell in 2 ticks") made the model pick a
  // cell its own criteria called deadly in 7 of 16 sampled positions; this one picked none.
  const progress = move === "forward" ? "closer to the far bank"
    : move === "back" ? "backwards" : move === "wait" ? "no progress" : "sideways";
  if (y === 0) return "safe: the far bank, this completes the crossing";
  if (y === H - 1) return `safe: ${progress}, the start bank has no traffic`;
  if (occupiedNow(state, x, y)) return "deadly: a car is standing on that cell";
  const ticks = arrivesIn(state, x, y);
  if (ticks !== null && ticks <= 1) return "deadly: a car arrives on that cell this tick";
  return `safe: ${progress}, nearest car ${ticks === null ? HORIZON : ticks} ticks away`;
}

export function questions(state) {
  const criteria = {};
  for (const move of legal(state)) criteria[move] = moveFacts(state, move);
  return {
    move: {
      type: "choice",
      instructions: "Pick the move that gets closer to the far bank without being run over. Waiting is only worth it when every step forward is under a car.",
      criteria,
    },
    hit_risk: {
      type: "score",
      instructions: "How close is the traffic to the cell you are moving into?",
      criteria: ["the cell is clear", "a car is a couple of ticks away", "a car is on that cell or arrives now"],
    },
  };
}

export function decide(state, answers) {
  const options = legal(state);
  const probabilities = (answers.move && answers.move.probabilities) || {};
  let action = options[0];
  let best = -1;
  for (const move of options) {
    const p = probabilities[move] || 0;
    if (p > best) {
      best = p;
      action = move;
    }
  }
  return {
    action,
    extras: {
      probabilities,
      confidence: answers.move ? answers.move.confidence : null,
      hit_risk: answers.hit_risk ? answers.hit_risk.score : null,
      hit_risk_level: answers.hit_risk ? topLevel(answers.hit_risk.probabilities) : null,
    },
  };
}

function topLevel(probabilities) {
  if (!probabilities) return null;
  return Number(Object.entries(probabilities).sort((a, b) => b[1] - a[1])[0][0]);
}

export function shield(state, action, extras = {}) {
  const { x, y } = target(state, action);
  if (!deadly(state, x, y)) return { action, intervened: false, reason: "" };
  const safe = legal(state).filter((m) => {
    const t = target(state, m);
    return !deadly(state, t.x, t.y);
  });
  if (!safe.length) return { action, intervened: false, reason: "every move is under a car" };
  const probabilities = extras.probabilities || {};
  const ranked = safe.slice().sort((a, b) => (probabilities[b] || 0) - (probabilities[a] || 0));
  const pick = probabilities[ranked[0]] ? ranked[0] : safe.includes("wait") ? "wait" : ranked[0];
  return { action: pick, intervened: true, reason: `${action} steps under a car; took ${pick}` };
}

export function step(state, action) {
  const next = structuredClone(state);
  const events = [];
  const move = legal(state).includes(action) ? action : "wait";
  const t = target(state, move);
  next.x = t.x;
  next.y = t.y;
  next.steps += 1;

  // The frog moves into the traffic as it stands, then the traffic rolls one step.
  let hit = occupiedNow(state, next.x, next.y);
  for (const lane of next.lanes) {
    if (state.steps % lane.period === 0) lane.cars = lane.cars.map((x) => (x + lane.dir + W) % W);
  }
  if (!hit && next.y > 0 && next.y < H - 1) {
    const lane = laneAt(next, next.y);
    hit = lane.cars.includes(next.x);
  }
  if (hit) {
    next.alive = false;
    events.push("run over");
    return { state: next, reward: -1, done: true, events };
  }
  if (next.y === 0) {
    next.score += 1;
    next.y = H - 1;                       // back to the start bank for the next crossing
    events.push("reached the far bank");
    return { state: next, reward: 1, done: false, events };
  }
  return { state: next, reward: 0, done: false, events };
}

export function random(state) {
  const options = legal(state);
  return options[Math.floor(prnd(state) * options.length)];
}

// Walk forward when the cell ahead survives the tick, otherwise sidestep or wait.
export function heuristic(state) {
  const options = legal(state).filter((m) => {
    const t = target(state, m);
    return !deadly(state, t.x, t.y);
  });
  if (!options.length) return "wait";
  const rank = { forward: 0, left: 1, right: 1, wait: 2, back: 3 };
  return options.slice().sort((a, b) => rank[a] - rank[b] || horizon(state, b) - horizon(state, a))[0];
}

function horizon(state, move) {
  const t = target(state, move);
  const ticks = arrivesIn(state, t.x, t.y);
  return ticks === null ? HORIZON : ticks;
}

// Ground truth for the score question, on the cell the model actually chose.
export function truth(state, action) {
  const t = target(state, action);
  if (deadly(state, t.x, t.y)) return { hit_risk: 2, fatal: true };
  const ticks = horizon(state, action);
  return { hit_risk: ticks <= 3 ? 1 : 0, fatal: false };
}

export function summary(state) {
  return { score: state.score, steps: state.steps, row: state.y, alive: state.alive };
}
