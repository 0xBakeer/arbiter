// Every game has to satisfy the same contract, so the harness, the panel and measure.mjs can
// treat them all the same way. One file, six games, no server.
import test from "node:test";
import assert from "node:assert/strict";

import { GAMES, SEEDS, fakeAnswers, load, play } from "./helpers.mjs";

for (const game of GAMES) {
  const logic = await load(game);

  test(`${game}: exports the whole contract`, () => {
    for (const name of ["init", "encode", "questions", "decide", "legal", "shield", "step",
                        "random", "heuristic", "summary"]) {
      assert.equal(typeof logic[name], "function", `${game}.${name}`);
    }
    for (const key of ["id", "name", "tick_ms", "max_steps", "description", "actions", "shield"]) {
      assert.ok(logic.meta[key] != null, `${game}.meta.${key}`);
    }
    assert.equal(logic.meta.id, game);
    assert.ok(logic.meta.tick_ms >= 50 && logic.meta.tick_ms <= 2000);
    assert.ok(logic.meta.max_steps >= 20 && logic.meta.max_steps <= 1000);
  });

  test(`${game}: the same seed gives the same episode`, () => {
    for (const seed of SEEDS) {
      const a = play(logic, seed, (s) => logic.heuristic(s));
      const b = play(logic, seed, (s) => logic.heuristic(s));
      assert.deepEqual(a.actions, b.actions);
      assert.deepEqual(a.state, b.state);
      assert.deepEqual(logic.summary(a.state), logic.summary(b.state));
    }
  });

  test(`${game}: different seeds give different episodes`, () => {
    const runs = SEEDS.map((seed) => JSON.stringify(play(logic, seed, (s) => logic.random(s)).actions));
    assert.ok(new Set(runs).size > 1, "every seed produced the same run");
  });

  test(`${game}: encode is a compact human-readable string`, () => {
    for (const seed of SEEDS) {
      const text = logic.encode(logic.init(seed));
      assert.equal(typeof text, "string");
      assert.ok(text.length > 20 && text.length < 1200, `length ${text.length}`);
      assert.ok(!text.includes("undefined") && !text.includes("NaN"), text);
      assert.ok(!text.includes("[object"), text);
    }
  });

  test(`${game}: the questions are valid Jev questions`, () => {
    for (const seed of SEEDS) {
      const { trace } = play(logic, seed, (s) => logic.heuristic(s), 20);
      for (const state of trace) {
        const questions = logic.questions(state);
        assert.ok(Object.keys(questions).length >= 1);
        // A game with a single legal action must not send a one-option choice: the server
        // rejects it, and there is nothing to decide anyway.
        if (logic.legal(state).length < 2) {
          assert.ok(!Object.values(questions).some((q) => q.type === "choice"),
            "a choice question with one legal action");
        }
        for (const [id, q] of Object.entries(questions)) {
          assert.ok(["choice", "noul", "score"].includes(q.type), `${id}: ${q.type}`);
          assert.ok(typeof q.instructions === "string" && q.instructions.trim().length > 0, id);
          assert.ok(!q.instructions.includes("undefined"), q.instructions);
          if (q.type === "choice") {
            const options = Object.keys(q.criteria);
            assert.ok(options.length >= 2 && options.length <= 255, `${id}: ${options.length} options`);
            assert.ok(options.every((o) => logic.legal(state).includes(o)), `${id}: offered an illegal option`);
            for (const text of Object.values(q.criteria)) {
              assert.ok(typeof text === "string" && text.length > 0);
              assert.ok(!text.includes("undefined") && !text.includes("NaN"), text);
            }
          } else if (q.type === "score") {
            assert.ok(Array.isArray(q.criteria));
            assert.ok(q.criteria.length >= 2 && q.criteria.length <= 10);
          } else if (q.criteria) {
            assert.deepEqual(Object.keys(q.criteria).sort(), ["false", "true"]);
          }
        }
      }
    }
  });

  test(`${game}: decide turns answers into a legal action`, () => {
    for (const seed of SEEDS) {
      const state = logic.init(seed);
      const decision = logic.decide(state, fakeAnswers(logic.questions(state)));
      assert.ok(logic.legal(state).includes(decision.action), `${decision.action} not legal`);
      assert.equal(typeof decision.extras, "object");
    }
  });

  test(`${game}: step returns state, reward, done and events`, () => {
    for (const seed of SEEDS) {
      let state = logic.init(seed);
      for (let i = 0; i < 40; i += 1) {
        const before = JSON.stringify(state);
        const out = logic.step(state, logic.heuristic(state));
        assert.equal(JSON.stringify(state), before, "step mutated the state it was given");
        assert.equal(typeof out.reward, "number");
        assert.equal(typeof out.done, "boolean");
        assert.ok(Array.isArray(out.events));
        assert.ok(out.state !== state);
        assert.equal(out.state.steps, state.steps + 1);
        assert.ok(logic.summary(out.state).score >= 0);
        state = out.state;
        if (out.done) break;
      }
    }
  });

  test(`${game}: the baseline policies only play legal actions`, () => {
    for (const seed of SEEDS) {
      let state = logic.init(seed);
      for (let i = 0; i < 60; i += 1) {
        for (const action of [logic.random(state), logic.heuristic(state)]) {
          assert.ok(logic.legal(state).includes(action), `${action} not in ${logic.legal(state)}`);
        }
        const out = logic.step(state, logic.heuristic(state));
        state = out.state;
        if (out.done) break;
      }
    }
  });

  test(`${game}: the shield answers with an action, a flag and a reason`, () => {
    for (const seed of SEEDS) {
      const state = logic.init(seed);
      for (const action of logic.legal(state)) {
        const guard = logic.shield(state, action, {});
        assert.ok(logic.legal(state).includes(guard.action));
        assert.equal(typeof guard.intervened, "boolean");
        assert.equal(typeof guard.reason, "string");
        if (!guard.intervened) assert.equal(guard.action, action);
        else assert.ok(guard.reason.length > 0, "an intervention has to say why");
      }
    }
  });

  if (logic.truth) {
    test(`${game}: ground truth is keyed like the questions`, () => {
      for (const seed of SEEDS) {
        const state = logic.init(seed);
        const ids = Object.keys(logic.questions(state));
        const truth = logic.truth(state, logic.heuristic(state));
        const matched = Object.keys(truth).filter((k) => ids.includes(k));
        assert.ok(matched.length > 0, `no ground truth matches a question id in ${game}`);
      }
    });
  }
}
