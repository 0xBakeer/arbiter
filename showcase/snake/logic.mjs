// Snake on a 12x12 grid. The model is asked one choice question per tick, and the options are
// written as what each move leads to ("up: food 1 step closer, 5 cells of clear space"), because
// a text encoder can only weigh what the text says. Two nouls ride along for display, and both
// have exact ground truth in this file, so `measure.mjs` can score the model's calibration.

import { seedStreams, rnd, prnd } from "../_lib/prng.mjs";

export const meta = {
  id: "snake",
  name: "Snake",
  tick_ms: 120,
  max_steps: 250,
  description: "12x12 snake. One choice question per tick, two nouls, a shield that vetoes suicide.",
  actions: ["up", "down", "left", "right"],
  shield: "vetoes a move into a wall or into the snake",
};

const W = 12;
const H = 12;
const STARVE = 150;                       // ticks without food before the episode is called off
const DIRS = { up: [0, -1], down: [0, 1], left: [-1, 0], right: [1, 0] };
const OPPOSITE = { up: "down", down: "up", left: "right", right: "left" };
const ORDER = ["up", "down", "left", "right"];

export function init(seed) {
  const state = {
    w: W,
    h: H,
    snake: [{ x: 5, y: 6 }, { x: 4, y: 6 }, { x: 3, y: 6 }],
    dir: "right",
    food: { x: 8, y: 3 },
    alive: true,
    score: 0,
    steps: 0,
    sinceFood: 0,
    ...seedStreams(seed),
  };
  state.food = placeFood(state);
  return state;
}

// -- geometry ----------------------------------------------------------------

const key = (x, y) => `${x},${y}`;

function occupied(state, { skipTail = false } = {}) {
  const body = skipTail ? state.snake.slice(0, -1) : state.snake;
  return new Set(body.map((c) => key(c.x, c.y)));
}

function ahead(state, dir) {
  const head = state.snake[0];
  const [dx, dy] = DIRS[dir];
  return { x: head.x + dx, y: head.y + dy };
}

const inside = (c) => c.x >= 0 && c.y >= 0 && c.x < W && c.y < H;

// The tail tip vacates the cell on the same tick, unless the move eats and the snake grows.
function fatal(state, dir) {
  const target = ahead(state, dir);
  if (!inside(target)) return true;
  const eats = target.x === state.food.x && target.y === state.food.y;
  return occupied(state, { skipTail: !eats }).has(key(target.x, target.y));
}

function wallDistance(state, dir) {
  const head = state.snake[0];
  if (dir === "up") return head.y;
  if (dir === "down") return H - 1 - head.y;
  if (dir === "left") return head.x;
  return W - 1 - head.x;
}

// How many steps straight ahead before the snake's own body is in the way (null if none).
function bodyDistance(state, dir) {
  const [dx, dy] = DIRS[dir];
  const body = occupied(state);
  const head = state.snake[0];
  for (let i = 1; i < Math.max(W, H); i += 1) {
    const c = { x: head.x + dx * i, y: head.y + dy * i };
    if (!inside(c)) return null;
    if (body.has(key(c.x, c.y))) return i;
  }
  return null;
}

// Free cells reachable from `from`, walking around the body as it will stand after the move.
function reachable(from, blocked) {
  if (!inside(from) || blocked.has(key(from.x, from.y))) return 0;
  const seen = new Set([key(from.x, from.y)]);
  const queue = [from];
  while (queue.length) {
    const c = queue.pop();
    for (const [dx, dy] of Object.values(DIRS)) {
      const n = { x: c.x + dx, y: c.y + dy };
      const k = key(n.x, n.y);
      if (!inside(n) || seen.has(k) || blocked.has(k)) continue;
      seen.add(k);
      queue.push(n);
    }
  }
  return seen.size;
}

function headRoom(state) {
  const blocked = occupied(state, { skipTail: true });
  blocked.delete(key(state.snake[0].x, state.snake[0].y));
  return reachable(state.snake[0], blocked) - 1;
}

function spaceAfter(state, dir) {
  const target = ahead(state, dir);
  const eats = target.x === state.food.x && target.y === state.food.y;
  const blocked = occupied(state, { skipTail: !eats });
  return reachable(target, blocked);
}

function foodReachable(state) {
  const blocked = occupied(state, { skipTail: true });
  blocked.delete(key(state.snake[0].x, state.snake[0].y));
  const seen = new Set();
  const queue = [state.snake[0]];
  seen.add(key(state.snake[0].x, state.snake[0].y));
  while (queue.length) {
    const c = queue.pop();
    if (c.x === state.food.x && c.y === state.food.y) return true;
    for (const [dx, dy] of Object.values(DIRS)) {
      const n = { x: c.x + dx, y: c.y + dy };
      const k = key(n.x, n.y);
      if (!inside(n) || seen.has(k) || blocked.has(k)) continue;
      seen.add(k);
      queue.push(n);
    }
  }
  return false;
}

function placeFood(state) {
  const body = occupied(state);
  const free = [];
  for (let y = 0; y < H; y += 1) {
    for (let x = 0; x < W; x += 1) if (!body.has(key(x, y))) free.push({ x, y });
  }
  return free[Math.floor(rnd(state) * free.length)];
}

function foodDelta(state, dir) {
  const head = state.snake[0];
  const target = ahead(state, dir);
  const before = Math.abs(head.x - state.food.x) + Math.abs(head.y - state.food.y);
  const after = Math.abs(target.x - state.food.x) + Math.abs(target.y - state.food.y);
  return before - after;                  // positive means the move closes in
}

// -- the contract ------------------------------------------------------------

// Reversing into the neck is an instant loss in every snake, so it is not on the menu.
export function legal(state) {
  return ORDER.filter((d) => state.snake.length < 2 || d !== OPPOSITE[state.dir]);
}

export function encode(state) {
  const head = state.snake[0];
  const dy = state.food.y - head.y;
  const dx = state.food.x - head.x;
  const where = [];
  if (dy < 0) where.push(`${-dy} up`);
  if (dy > 0) where.push(`${dy} down`);
  if (dx < 0) where.push(`${-dx} left`);
  if (dx > 0) where.push(`${dx} right`);
  // Deliberately short. Everything about the individual moves lives in the criteria, and a state
  // that repeats it only costs tokens: 460 tokens per tick measured 73 ms, 100 tokens measure 35.
  return [
    `Snake on a 12 by 12 grid, ${state.snake.length} segments long, moving ${state.dir}.`,
    `Food is ${where.length ? where.join(" and ") : "under the head"}.`,
    `The head can still reach ${headRoom(state)} free cells and the food is`,
    foodReachable(state) ? "among them." : "cut off by the body.",
  ].join(" ");
}

// One sentence per move, in the same words the criteria use: this is the whole trick.
function moveFacts(state, dir) {
  if (!inside(ahead(state, dir))) return "the wall is one step away, the snake dies";
  const body = bodyDistance(state, dir);
  if (body === 1) return "the snake's own body is one step away, the snake dies";
  const space = spaceAfter(state, dir);
  const len = state.snake.length;
  const parts = [
    foodDelta(state, dir) > 0 ? "food gets closer" : "food gets farther",
    `${wallDistance(state, dir)} steps to the wall`,
    // A raw cell count means nothing to a reader; what matters is the count against the snake.
    space <= len ? `a trap: only ${space} free cells, less than the snake is long`
      : space < len * 3 ? `${space} free cells, tight`
      : "open space beyond",
  ];
  if (body !== null) parts.push(`own body ${body} steps ahead`);
  return parts.join(", ");
}

export function questions(state) {
  const criteria = {};
  for (const dir of legal(state)) criteria[dir] = moveFacts(state, dir);
  return {
    move: {
      type: "choice",
      instructions: "Pick the direction that eats the food without hitting a wall or the snake itself.",
      criteria,
    },
    dead_end: {
      type: "noul",
      instructions: "Taking that direction traps the snake in a space too small to fit its own body.",
      criteria: {
        true: "the space ahead closes in and is smaller than the snake",
        false: "there is open space ahead, larger than the snake",
      },
    },
    food_reachable: {
      type: "noul",
      instructions: "There is a clear path from the head to the food, around the snake's body.",
      criteria: { true: "the food can be walked to", false: "the body cuts the food off" },
    },
  };
}

export function decide(state, answers) {
  const options = legal(state);
  const probabilities = (answers.move && answers.move.probabilities) || {};
  let action = options[0];
  let best = -1;
  for (const dir of options) {
    const p = probabilities[dir] || 0;
    if (p > best) {
      best = p;
      action = dir;
    }
  }
  return {
    action,
    extras: {
      probabilities,
      confidence: answers.move ? answers.move.confidence : null,
      dead_end: answers.dead_end ? answers.dead_end.noul : null,
      food_reachable: answers.food_reachable ? answers.food_reachable.noul : null,
    },
  };
}

// Code-owned safety. Never lets a fatal move through while a safe one is on the board, and when
// it has to overrule the model it takes the model's next choice by probability, not its own.
export function shield(state, action, extras = {}) {
  if (!fatal(state, action)) return { action, intervened: false, reason: "" };
  const safe = legal(state).filter((d) => !fatal(state, d));
  if (!safe.length) return { action, intervened: false, reason: "no safe move left" };
  const probabilities = extras.probabilities || {};
  const ranked = safe.slice().sort((a, b) => (probabilities[b] || 0) - (probabilities[a] || 0));
  const pick = probabilities[ranked[0]] ? ranked[0] : preferred(state, safe);
  const why = !inside(ahead(state, action)) ? "into the wall" : "into its own body";
  return { action: pick, intervened: true, reason: `${action} goes ${why}; took ${pick}` };
}

function preferred(state, options) {
  return options
    .slice()
    .sort((a, b) => foodDelta(state, b) - foodDelta(state, a) || spaceAfter(state, b) - spaceAfter(state, a))[0];
}

export function step(state, action) {
  const next = structuredClone(state);
  const events = [];
  const dir = legal(state).includes(action) ? action : state.dir;
  next.dir = dir;
  next.steps += 1;
  next.sinceFood += 1;

  const target = ahead(state, dir);
  const eats = target.x === state.food.x && target.y === state.food.y;
  if (!inside(target)) {
    next.alive = false;
    events.push("hit the wall");
    return { state: next, reward: -1, done: true, events };
  }
  if (occupied(state, { skipTail: !eats }).has(key(target.x, target.y))) {
    next.alive = false;
    events.push("hit its own body");
    return { state: next, reward: -1, done: true, events };
  }

  next.snake.unshift(target);
  let reward = 0;
  if (eats) {
    next.score += 1;
    next.sinceFood = 0;
    next.food = placeFood(next);
    reward = 1;
    events.push("ate the food");
  } else {
    next.snake.pop();
  }
  if (next.sinceFood >= STARVE) {
    next.alive = false;
    events.push(`no food for ${STARVE} ticks`);
    return { state: next, reward, done: true, events };
  }
  return { state: next, reward, done: false, events };
}

export function random(state) {
  const options = legal(state);
  return options[Math.floor(prnd(state) * options.length)];
}

export function heuristic(state) {
  const options = legal(state);
  const safe = options.filter((d) => !fatal(state, d));
  if (!safe.length) return options[0];
  // Greedy towards the food, but never into a pocket smaller than the snake.
  const roomy = safe.filter((d) => spaceAfter(state, d) > state.snake.length);
  return preferred(state, roomy.length ? roomy : safe);
}

// Ground truth for the two nouls, used by measure.mjs for the calibration table.
export function truth(state, action) {
  const trapped = fatal(state, action) || spaceAfter(state, action) <= state.snake.length;
  return { dead_end: trapped, food_reachable: foodReachable(state) };
}

export function summary(state) {
  return {
    score: state.score,
    steps: state.steps,
    length: state.snake.length,
    alive: state.alive,
    since_food: state.sinceFood,
  };
}
