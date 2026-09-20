// The speed game. One three-way choice per tick on a 60 ms clock, the shortest state in the
// showcase, and nothing to think about except which way the paddle should go. What it measures
// is how many typed decisions a second the server can actually deliver into a running game.

import { seedStreams, rnd, prnd } from "../_lib/prng.mjs";

export const meta = {
  id: "paddle",
  name: "Paddle",
  tick_ms: 60,
  max_steps: 300,
  description: "Keep the ball in play. One choice of three per tick, on the fastest clock here.",
  actions: ["left", "stay", "right"],
  shield: "keeps the paddle on the board",
};

const W = 20;
const H = 14;
const HALF = 1;                           // paddle covers px-1 .. px+1

export function init(seed) {
  const state = {
    w: W,
    h: H,
    half: HALF,
    bx: 0,
    by: 2,
    vx: 1,
    vy: 1,
    px: Math.floor(W / 2),
    score: 0,
    steps: 0,
    alive: true,
    ...seedStreams(seed),
  };
  state.bx = 2 + Math.floor(rnd(state) * (W - 4));
  state.vx = rnd(state) < 0.5 ? -1 : 1;
  return state;
}

const offset = (state) => state.bx - state.px;   // positive: the ball is right of the paddle

export function legal(state) {
  return ["left", "stay", "right"].filter(
    (m) => !(m === "left" && state.px - 1 < HALF) && !(m === "right" && state.px + 1 > W - 1 - HALF));
}

// The shortest encoding in the showcase, and the measured one. Every sentence added here made
// the model play worse, because the words in the state collide with the names of the options:
// with "the ball is 3 columns to the right of the paddle centre, moving down and to the left"
// the model matched the reference paddle on 10 of 20 sampled positions and picked "left" 18
// times; with "3 columns off the paddle centre" it picked "stay" 15 times; with this one
// sentence it matched on 18 of 20. Everything the decision needs is in the criteria.
export function encode() {
  return "A paddle must be moved under a falling ball.";
}

function moveFacts(state, move) {
  const shift = move === "left" ? -1 : move === "right" ? 1 : 0;
  const after = Math.abs(offset(state) - shift);
  const now = Math.abs(offset(state));
  const verdict = after < now ? "closer to the ball" : after > now ? "further from the ball" : "no change";
  // "columns" even at one column, and that is measured, not sloppy: with "1 column" the model
  // stops closing the last step and answers "stay", and the score over 20 episodes falls from 12
  // to under 1. Six phrasings and their numbers are in the README.
  return `${verdict}: the paddle ends ${after} columns from the ball`;
}

export function questions(state) {
  const criteria = {};
  for (const move of legal(state)) criteria[move] = moveFacts(state, move);
  return {
    move: {
      type: "choice",
      instructions: "Move the paddle so that it is under the ball when the ball comes down.",
      criteria,
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
  return { action, extras: { probabilities, confidence: answers.move ? answers.move.confidence : null } };
}

// The only unsafe move here is walking the paddle off the edge of the box, so that is all the
// shield does. It never saves a rally; it keeps the paddle on the board.
export function shield(state, action) {
  if (legal(state).includes(action)) return { action, intervened: false, reason: "" };
  return { action: "stay", intervened: true, reason: `${action} would push the paddle off the board` };
}

export function step(state, action) {
  const next = structuredClone(state);
  const events = [];
  next.steps += 1;
  if (action === "left") next.px = Math.max(HALF, next.px - 1);
  if (action === "right") next.px = Math.min(W - 1 - HALF, next.px + 1);

  next.bx += next.vx;
  next.by += next.vy;
  if (next.bx <= 0) {
    next.bx = 0;
    next.vx = 1;
  }
  if (next.bx >= W - 1) {
    next.bx = W - 1;
    next.vx = -1;
  }
  if (next.by <= 0) {
    next.by = 0;
    next.vy = 1;
  }

  let reward = 0;
  if (next.by >= H - 1) {
    const hit = Math.abs(next.bx - next.px);
    if (hit <= HALF) {
      next.by = H - 1;
      next.vy = -1;
      // The edge of the paddle sends the ball back the way it came.
      if (next.bx !== next.px) next.vx = next.bx > next.px ? 1 : -1;
      next.score += 1;
      reward = 1;
      events.push("returned the ball");
    } else {
      next.alive = false;
      events.push("missed the ball");
      return { state: next, reward: -1, done: true, events };
    }
  }
  return { state: next, reward, done: false, events };
}

export function random(state) {
  const options = legal(state);
  return options[Math.floor(prnd(state) * options.length)];
}

export function heuristic(state) {
  const gap = offset(state);
  const want = gap === 0 ? "stay" : gap > 0 ? "right" : "left";
  return legal(state).includes(want) ? want : "stay";
}

export function summary(state) {
  return { score: state.score, steps: state.steps, alive: state.alive, gap: Math.abs(offset(state)) };
}
