// The shield is the only thing in the showcase allowed to overrule the model, so it gets its own
// property: with a survivable move on the board, it must never let a fatal one through.
import test from "node:test";
import assert from "node:assert/strict";

import { load, play } from "./helpers.mjs";

// Death is deterministic in these two: step tells us, and the state says it was not survived.
// Paddle is not one of them on purpose - its shield only keeps the paddle on the board, and a
// missed ball is the model's own doing. That is checked separately at the bottom.
const DETERMINISTIC = ["snake", "crossing"];

function fatal(logic, state, action) {
  const out = logic.step(state, action);
  return out.done && out.state.alive === false;
}

for (const game of DETERMINISTIC) {
  const logic = await load(game);

  test(`${game}: the shield never plays a fatal move while a safe one exists`, () => {
    let checked = 0;
    let saved = 0;
    for (const seed of [1, 2, 3, 4, 5, 6, 7, 8]) {
      // walk the board with the random policy, which is what gets into trouble
      const { trace } = play(logic, seed, (s) => logic.random(s), 120);
      for (const state of trace) {
        const options = logic.legal(state);
        const safe = options.filter((a) => !fatal(logic, state, a));
        if (!safe.length) continue;                       // nothing to ask of the shield
        for (const action of options) {
          const guard = logic.shield(state, action, {});
          assert.ok(!fatal(logic, state, guard.action),
            `${game} seed ${seed}: shield passed ${guard.action} with ${safe} available`);
          if (guard.intervened) saved += 1;
          checked += 1;
        }
      }
    }
    assert.ok(checked > 200, `only ${checked} positions checked`);
    assert.ok(saved > 0, "the shield never fired, so the property proved nothing");
  });

  test(`${game}: the shield leaves a survivable move alone`, () => {
    for (const seed of [1, 5, 9]) {
      const { trace } = play(logic, seed, (s) => logic.heuristic(s), 60);
      for (const state of trace) {
        for (const action of logic.legal(state)) {
          if (fatal(logic, state, action)) continue;
          const guard = logic.shield(state, action, {});
          assert.equal(guard.action, action);
          assert.equal(guard.intervened, false);
        }
      }
    }
  });

  test(`${game}: an overruled action falls back to the model's next choice`, () => {
    for (const seed of [2, 3, 11]) {
      const { trace } = play(logic, seed, (s) => logic.random(s), 90);
      for (const state of trace) {
        const options = logic.legal(state);
        const safe = options.filter((a) => !fatal(logic, state, a));
        const doomed = options.find((a) => fatal(logic, state, a));
        if (!doomed || safe.length < 2) continue;
        // hand the shield a ranking that prefers the second survivable option
        const probabilities = Object.fromEntries(options.map((a) => [a, a === safe[1] ? 0.9 : 0.01]));
        const guard = logic.shield(state, doomed, { probabilities });
        assert.equal(guard.action, safe[1]);
        assert.ok(guard.reason.length > 0);
      }
    }
  });
}

test("paddle's shield only keeps the paddle on the board", async () => {
  const logic = await load("paddle");
  let stopped = 0;
  // Drive the paddle into both walls on purpose: the random policy misses the ball long before
  // it ever gets there.
  for (const [seed, only] of [[1, "left"], [2, "right"], [3, "left"], [4, "right"]]) {
    const { trace } = play(logic, seed, () => only, 120);
    for (const state of trace) {
      for (const action of ["left", "stay", "right"]) {
        const guard = logic.shield(state, action, {});
        if (logic.legal(state).includes(action)) {
          assert.deepEqual(guard, { action, intervened: false, reason: "" });
        } else {
          assert.equal(guard.action, "stay");
          assert.equal(guard.intervened, true);
          stopped += 1;
        }
        const next = logic.step(state, guard.action).state;
        assert.ok(next.px - next.half >= 0 && next.px + next.half <= next.w - 1,
          `paddle left the board at ${next.px}`);
      }
    }
  }
  assert.ok(stopped > 0, "the paddle never reached a wall, so the property proved nothing");
});

test("hopper and mines pass every action straight through", async () => {
  for (const game of ["hopper", "mines"]) {
    const logic = await load(game);
    assert.equal(logic.meta.shield, "none");
    const { trace } = play(logic, 3, (s) => logic.heuristic(s), 40);
    for (const state of trace) {
      for (const action of logic.legal(state).slice(0, 6)) {
        const guard = logic.shield(state, action, {});
        assert.deepEqual(guard, { action, intervened: false, reason: "" });
      }
    }
  }
});

test("dungeon never fights something that can kill it in one blow", async () => {
  const logic = await load("dungeon");
  let fired = 0;
  for (let seed = 1; seed <= 40; seed += 1) {
    const { trace } = play(logic, seed, (s) => logic.random(s), 120);
    for (const state of trace) {
      const monster = state.room.monster;
      if (!monster || monster.max_hit < state.hp) continue;
      if (!logic.legal(state).includes("flee")) continue;
      for (const action of ["fight", "loot", "heal"]) {
        if (!logic.legal(state).includes(action)) continue;
        const guard = logic.shield(state, action, {});
        assert.notEqual(guard.action, action);
        assert.equal(guard.intervened, true);
        fired += 1;
      }
    }
  }
  assert.ok(fired > 20, `the rule was only exercised ${fired} times`);
});
