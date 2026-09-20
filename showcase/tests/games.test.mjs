// What each game promises beyond the shared contract: the rules of the game itself.
import test from "node:test";
import assert from "node:assert/strict";

import { load, play } from "./helpers.mjs";

const SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

test("snake grows only by eating and never overlaps itself alive", async () => {
  const logic = await load("snake");
  for (const seed of SEEDS) {
    let state = logic.init(seed);
    for (let i = 0; i < 200; i += 1) {
      const out = logic.step(state, logic.heuristic(state));
      const ate = (out.events || []).some((e) => /ate/.test(e));
      if (out.state.alive) {
        assert.equal(out.state.snake.length, state.snake.length + (ate ? 1 : 0));
        assert.equal(out.state.score, state.score + (ate ? 1 : 0));
        const cells = new Set(out.state.snake.map((c) => `${c.x},${c.y}`));
        assert.equal(cells.size, out.state.snake.length, "the snake overlapped itself and lived");
        for (const c of out.state.snake) {
          assert.ok(c.x >= 0 && c.y >= 0 && c.x < out.state.w && c.y < out.state.h);
        }
      }
      state = out.state;
      if (out.done) break;
    }
  }
});

test("snake's dead_end ground truth agrees with what the step actually does", async () => {
  const logic = await load("snake");
  let fatalAndFlagged = 0;
  for (const seed of SEEDS) {
    const { trace } = play(logic, seed, (s) => logic.random(s), 150);
    for (const state of trace) {
      for (const action of logic.legal(state)) {
        const out = logic.step(state, action);
        const died = out.done && !out.state.alive && !/no food/.test(out.events.join(" "));
        if (died) {
          assert.equal(logic.truth(state, action).dead_end, true,
            "a move that ends the game must count as a dead end");
          fatalAndFlagged += 1;
        }
      }
    }
  }
  assert.ok(fatalAndFlagged > 10);
});

test("hopper scores only by passing a wall and dies at the floor or the ceiling", async () => {
  const logic = await load("hopper");
  for (const seed of SEEDS) {
    let state = logic.init(seed);
    for (let i = 0; i < 300; i += 1) {
      const out = logic.step(state, logic.heuristic(state));
      const passed = (out.events || []).some((e) => /through the gap/.test(e));
      assert.equal(out.state.score, state.score + (passed ? 1 : 0));
      if (!out.state.alive) {
        assert.ok((out.events || []).length > 0, "a death has to say what happened");
      }
      state = out.state;
      if (out.done) break;
    }
  }
});

test("hopper's reference controller never dies", async () => {
  const logic = await load("hopper");
  for (const seed of SEEDS) {
    let state = logic.init(seed);
    for (let i = 0; i < 400; i += 1) {
      const out = logic.step(state, logic.heuristic(state));
      assert.ok(out.state.alive, `the reference controller died on seed ${seed} at step ${i}`);
      state = out.state;
    }
    assert.ok(state.score > 20, `only ${state.score} walls on seed ${seed}`);
  }
});

test("crossing keeps the frog on the board and scores one per crossing", async () => {
  const logic = await load("crossing");
  for (const seed of SEEDS) {
    let state = logic.init(seed);
    for (let i = 0; i < 250; i += 1) {
      const out = logic.step(state, logic.heuristic(state));
      assert.ok(out.state.x >= 0 && out.state.x < out.state.w);
      assert.ok(out.state.y >= 0 && out.state.y < out.state.h);
      const crossed = (out.events || []).some((e) => /far bank/.test(e));
      assert.equal(out.state.score, state.score + (crossed ? 1 : 0));
      if (crossed) assert.equal(out.state.y, out.state.h - 1, "a crossing starts the next one");
      state = out.state;
      if (out.done) break;
    }
  }
});

test("crossing's cars stay in their lanes and keep their count", async () => {
  const logic = await load("crossing");
  const start = logic.init(3);
  const counts = start.lanes.map((l) => l.cars.length);
  const { trace } = play(logic, 3, (s) => logic.random(s), 200);
  for (const state of trace) {
    state.lanes.forEach((lane, i) => {
      assert.equal(lane.cars.length, counts[i]);
      for (const x of lane.cars) assert.ok(x >= 0 && x < state.w);
    });
  }
});

test("paddle keeps the ball inside the box and the paddle under it", async () => {
  const logic = await load("paddle");
  for (const seed of SEEDS) {
    let state = logic.init(seed);
    for (let i = 0; i < 300; i += 1) {
      const out = logic.step(state, logic.heuristic(state));
      assert.ok(out.state.bx >= 0 && out.state.bx <= out.state.w - 1, `ball at ${out.state.bx}`);
      assert.ok(out.state.by >= 0 && out.state.by <= out.state.h - 1, `ball at ${out.state.by}`);
      assert.ok(Math.abs(out.state.px - state.px) <= 1, "the paddle moved more than one column");
      const returned = (out.events || []).some((e) => /returned/.test(e));
      assert.equal(out.state.score, state.score + (returned ? 1 : 0));
      state = out.state;
      if (out.done) break;
    }
    assert.ok(state.score > 5, `the reference paddle only returned ${state.score} on seed ${seed}`);
  }
});

test("mines never puts a mine under the opening cell and only ever opens more", async () => {
  const logic = await load("mines");
  for (let seed = 1; seed <= 30; seed += 1) {
    let state = logic.init(seed);
    assert.equal(state.mines.filter(Boolean).length, state.total_mines);
    assert.ok(state.score > 0, "the opening reveal has to open something");
    assert.ok(state.revealed.every((open, i) => !open || !state.mines[i]),
      "the opening reveal uncovered a mine");
    for (let i = 0; i < 60; i += 1) {
      const out = logic.step(state, logic.heuristic(state));
      assert.ok(out.state.score >= state.score);
      assert.ok(out.state.revealed.filter(Boolean).length >= state.revealed.filter(Boolean).length);
      state = out.state;
      if (out.done) break;
    }
  }
});

test("mines' ground truth is the board, and the questions are the cells it is about", async () => {
  const logic = await load("mines");
  for (const seed of SEEDS) {
    const { trace } = play(logic, seed, (s) => logic.heuristic(s), 40);
    for (const state of trace.slice(0, 8)) {
      const truth = logic.truth(state);
      const ids = Object.keys(logic.questions(state));
      assert.deepEqual(Object.keys(truth).sort(), ids.slice().sort());
      assert.ok(ids.length <= 12, `${ids.length} questions in one call`);
      for (const [id, safe] of Object.entries(truth)) {
        const [, r, c] = /^r(\d+)c(\d+)$/.exec(id).map(Number);
        assert.equal(safe, !state.mines[r * state.w + c]);
      }
      const base = logic.baseline(state);
      assert.deepEqual(Object.keys(base).sort(), ids.slice().sort());
      for (const v of Object.values(base)) assert.ok(v >= 0 && v <= 1);
    }
  }
});

test("dungeon only goes deeper, never heals past full and never spends gold", async () => {
  const logic = await load("dungeon");
  for (let seed = 1; seed <= 30; seed += 1) {
    let state = logic.init(seed);
    for (let i = 0; i < 150; i += 1) {
      const out = logic.step(state, logic.heuristic(state));
      assert.ok(out.state.depth >= state.depth);
      assert.ok(out.state.depth <= state.rooms);
      assert.ok(out.state.gold >= state.gold);
      assert.ok(out.state.hp <= out.state.max_hp);
      assert.ok(out.state.hp >= 0);
      assert.equal(out.state.alive, out.state.hp > 0);
      state = out.state;
      if (out.done) break;
    }
  }
});

test("dungeon's danger level follows the monster in the room", async () => {
  const logic = await load("dungeon");
  const seen = new Set();
  for (let seed = 1; seed <= 40; seed += 1) {
    const { trace } = play(logic, seed, (s) => logic.random(s), 120);
    for (const state of trace) {
      const level = logic.truth(state).danger;
      seen.add(level);
      const monster = state.room.monster;
      if (!monster) assert.equal(level, 0);
      else if (monster.max_hit >= state.hp) assert.equal(level, 2);
      else if (monster.max_hit * 2 >= state.hp) assert.equal(level, 1);
      else assert.equal(level, 0);
    }
  }
  assert.deepEqual([...seen].sort(), [0, 1, 2], "the three danger levels all have to occur");
});
