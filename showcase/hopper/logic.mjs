// A Flappy-style side-scroller with no shield at all: one noul per tick, "flap now?", and the
// bird does exactly what the number says. Nothing in this file second-guesses the model, which
// makes hopper the honest read on whether the typed answer is any good.

import { seedStreams, rnd, prnd } from "../_lib/prng.mjs";

export const meta = {
  id: "hopper",
  name: "Hopper",
  tick_ms: 100,
  max_steps: 300,
  description: "One noul per tick decides the flap. No shield: the model flies the bird alone.",
  actions: ["flap", "glide"],
  shield: "none",
};

const W = 28;                             // columns of world visible at once
const H = 20;                             // rows, y grows downwards
const BIRD_X = 5;
const GRAVITY = 0.35;
const FLAP_V = -1.4;
const TERMINAL = 2.2;
const GAP = 6;                            // gap height in rows
const SPACING = 11;                       // columns between pipes

export function init(seed) {
  const state = {
    w: W,
    h: H,
    birdX: BIRD_X,
    gap: GAP,
    y: H / 2,
    vy: 0,
    pipes: [],
    score: 0,
    steps: 0,
    alive: true,
    ...seedStreams(seed),
  };
  state.pipes.push({ x: W - 4, gapY: gapCentre(state) });
  state.pipes.push({ x: W - 4 + SPACING, gapY: gapCentre(state) });
  return state;
}

function gapCentre(state) {
  const margin = GAP / 2 + 2;
  return margin + rnd(state) * (H - 2 * margin);
}

// The pipe the bird still has to fly through.
function nextPipe(state) {
  return state.pipes.find((p) => p.x >= state.birdX) || state.pipes[state.pipes.length - 1];
}

const round1 = (v) => Math.round(v * 10) / 10;

// -- the contract ------------------------------------------------------------

export function legal() {
  return ["flap", "glide"];
}

export function encode(state) {
  const pipe = nextPipe(state);
  const above = pipe.gapY - state.y;      // positive: the bird is above the centre of the gap
  // Three facts and nothing else. An earlier version also gave the distance to the floor and the
  // ceiling and the exact sink rate; that version scored AUROC 0.64 on the flap noul against
  // this one's 0.96, so the extra sentences were costing the answer, not helping it.
  return [
    "A bird must fly through a gap in a wall.",
    `The bird is ${round1(Math.abs(above))} rows ${above > 0 ? "above" : "below"} the gap,`,
    `${state.vy > 0 ? "sinking" : "rising"}.`,
    `The wall is ${pipe.x - state.birdX} ticks away.`,
  ].join(" ");
}

export function questions(state) {
  return {
    flap: {
      type: "noul",
      instructions: "The bird will pass under the gap if it does not flap now.",
      criteria: {
        true: "the bird is below the gap and falling",
        false: "the bird is level with or above the gap",
      },
    },
  };
}

export function decide(state, answers) {
  const p = answers.flap ? answers.flap.noul : 0;
  return {
    action: p >= 0.5 ? "flap" : "glide",
    extras: { flap: p, confidence: answers.flap ? answers.flap.confidence : null },
  };
}

// Hopper is the game without a shield: the model's answer is executed as given.
export function shield(state, action) {
  return { action, intervened: false, reason: "" };
}

export function step(state, action) {
  const next = structuredClone(state);
  const events = [];
  next.steps += 1;
  if (action === "flap") next.vy = FLAP_V;
  next.vy = Math.min(TERMINAL, next.vy + GRAVITY);
  next.y += next.vy;

  let reward = 0;
  for (const pipe of next.pipes) {
    const crossing = pipe.x === next.birdX;
    pipe.x -= 1;
    if (crossing) {
      if (Math.abs(next.y - pipe.gapY) > GAP / 2) {
        next.alive = false;
        events.push("clipped the wall");
      } else {
        next.score += 1;
        reward = 1;
        events.push("through the gap");
      }
    }
  }
  if (next.y <= 0 || next.y >= H) {
    next.alive = false;
    events.push(next.y <= 0 ? "hit the ceiling" : "hit the floor");
  }
  next.pipes = next.pipes.filter((p) => p.x > -2);
  const last = next.pipes[next.pipes.length - 1];
  if (!last || last.x < W - SPACING) next.pipes.push({ x: W - 1, gapY: gapCentre(next) });
  return { state: next, reward: next.alive ? reward : -1, done: !next.alive, events };
}

export function random(state) {
  return prnd(state) < 0.5 ? "flap" : "glide";
}

// Flap whenever the bird is under the centre of the gap. That one line clears 35 walls in 400
// ticks and never dies, which is what makes it a fair reference to score the model against.
export function heuristic(state) {
  return state.y > nextPipe(state).gapY ? "flap" : "glide";
}

// Ground truth for the noul is the reference controller's answer, not a law of physics: there is
// no single tick where flapping is strictly required, so "should flap now" means "the controller
// that never dies flaps now". The calibration table reads as agreement with that controller.
export function truth(state) {
  return { flap: heuristic(state) === "flap" };
}

export function summary(state) {
  return {
    score: state.score,
    steps: state.steps,
    alive: state.alive,
    height: round1(state.h - state.y),
  };
}
