// The loop itself, with a client that answers from a table instead of a server: policies, the
// shield counter, and a recording that replays into exactly the same episode.
import test from "node:test";
import assert from "node:assert/strict";

import { runEpisode, policyParts, POLICIES } from "../_lib/harness.mjs";
import { createRecorder, replay } from "../_lib/recorder.mjs";
import { createClient } from "../_lib/client.mjs";
import { fakeAnswers, load } from "./helpers.mjs";

// A stand-in for the server: the same shape /v1/systemone returns, no network.
function stubClient({ bias = null, latency = 7 } = {}) {
  const calls = [];
  return {
    calls,
    async ask(state, questions) {
      calls.push({ state, questions });
      return {
        answers: fakeAnswers(questions, bias),
        usage: { input_tokens: 100, output_tokens: 0 },
        routing: { model: "english" },
        model: "laya-english",
        latency_ms: latency,
        inference_ms: latency,
        roundtrip_ms: latency + 1,
      };
    },
  };
}

test("policyParts splits the four policies and rejects anything else", () => {
  assert.deepEqual(policyParts("model+shield"), { source: "model", shielded: true });
  assert.deepEqual(policyParts("random+shield"), { source: "random", shielded: true });
  assert.deepEqual(policyParts("model"), { source: "model", shielded: false });
  assert.throws(() => policyParts("model+guard"), /unknown policy/);
  assert.equal(POLICIES.length, 4);
});

test("a model policy asks once per step and counts what came back", async () => {
  const logic = await load("snake");
  const client = stubClient();
  const out = await runEpisode(logic, "model+shield", { client, seed: 4, maxSteps: 25 });
  assert.equal(client.calls.length, out.stats.steps);
  assert.equal(out.stats.decisions, out.stats.steps);
  assert.equal(out.stats.tokens, 100 * out.stats.steps);
  assert.equal(out.stats.inferences.length, out.stats.steps);
  assert.equal(out.stats.roundtrips.length, out.stats.steps);
  assert.ok(out.stats.decisions_per_s > 0);
  assert.equal(out.summary.steps, out.stats.steps);
});

test("the baselines never call the server", async () => {
  const logic = await load("crossing");
  for (const policy of ["random+shield", "heuristic+shield"]) {
    const client = stubClient();
    const out = await runEpisode(logic, policy, { client, seed: 2, maxSteps: 30 });
    assert.equal(client.calls.length, 0);
    assert.equal(out.stats.inferences.length, 0);
    assert.equal(out.stats.decisions_per_s, null);
  }
});

test("a model policy without a client is an error, not a silent baseline", async () => {
  const logic = await load("snake");
  await assert.rejects(() => runEpisode(logic, "model", { seed: 1 }), /needs a client/);
});

test("maxSteps ends the episode even when the game would go on", async () => {
  const logic = await load("paddle");
  const out = await runEpisode(logic, "heuristic+shield", { seed: 3, maxSteps: 12 });
  assert.equal(out.stats.steps, 12);
  assert.equal(out.stats.done, false);
});

test("the same seed and the same answers give the same episode twice", async () => {
  const logic = await load("snake");
  const first = await runEpisode(logic, "model+shield", { client: stubClient(), seed: 9, maxSteps: 40 });
  const second = await runEpisode(logic, "model+shield", { client: stubClient(), seed: 9, maxSteps: 40 });
  assert.deepEqual(first.summary, second.summary);
  assert.deepEqual(first.stats.steps, second.stats.steps);
  assert.deepEqual(first.state, second.state);
});

test("running without the shield counts no interventions and dies sooner", async () => {
  const logic = await load("crossing");
  // bias every answer towards walking forward, straight into the traffic
  const shielded = await runEpisode(logic, "model+shield",
    { client: stubClient({ bias: "forward" }), seed: 5, maxSteps: 120 });
  const bare = await runEpisode(logic, "model",
    { client: stubClient({ bias: "forward" }), seed: 5, maxSteps: 120 });
  assert.equal(bare.stats.interventions, 0);
  assert.ok(shielded.stats.interventions > 0);
  assert.ok(shielded.stats.steps > bare.stats.steps,
    `shielded ${shielded.stats.steps} steps, bare ${bare.stats.steps}`);
});

test("onTick sees both sides of every step", async () => {
  const logic = await load("hopper");
  const seen = [];
  await runEpisode(logic, "model", {
    client: stubClient(), seed: 6, maxSteps: 15,
    onTick: (tick) => seen.push(tick),
  });
  assert.ok(seen.length > 0);
  for (const tick of seen) {
    assert.ok(tick.state && tick.next && tick.next !== tick.state);
    assert.equal(typeof tick.timing.inference_ms, "number");
    assert.ok(tick.answers.flap);
    assert.deepEqual(Object.keys(tick.truth), ["flap"]);
  }
  assert.deepEqual(seen.map((t) => t.step), seen.map((_, i) => i + 1));
});

test("a recording replays into exactly the same episode", async () => {
  const logic = await load("snake");
  const recorder = createRecorder({ game: "snake", policy: "model+shield", seed: 11, meta: logic.meta });
  const live = [];
  await runEpisode(logic, "model+shield", {
    client: stubClient(), seed: 11, maxSteps: 50, recorder,
    onTick: (tick) => live.push({ action: tick.action, summary: tick.summary }),
  });
  const log = recorder.log;
  assert.equal(log.ticks.length, live.length);
  assert.equal(log.seed, 11);
  assert.equal(log.tick_ms, logic.meta.tick_ms);
  assert.deepEqual(log.summary, live[live.length - 1].summary);
  assert.equal(log.stats.policy, "model+shield");

  const played = [];
  await replay(log, { logic, tickMs: 0, onTick: (tick) => played.push(tick) });
  assert.equal(played.length, live.length);
  played.forEach((tick, i) => {
    assert.equal(tick.action, live[i].action);
    assert.deepEqual(logic.summary(tick.next), live[i].summary);
    assert.equal(tick.index, i);
  });
});

test("a recording survives a round trip through JSON", async () => {
  const logic = await load("mines");
  const recorder = createRecorder({ game: "mines", policy: "model", seed: 2, meta: logic.meta });
  await runEpisode(logic, "model", { client: stubClient(), seed: 2, maxSteps: 20, recorder });
  const log = JSON.parse(JSON.stringify(recorder.log));
  const played = [];
  await replay(log, { logic, tickMs: 0, onTick: (tick) => played.push(tick) });
  assert.equal(played.length, log.ticks.length);
  assert.ok(played.every((tick) => typeof tick.action === "string"));
});

test("the client posts the Jev body and reports both timings", async () => {
  let sent = null;
  const client = createClient({
    baseUrl: "http://example.invalid/",
    model: "typed-decisions",
    fetchImpl: async (url, init) => {
      sent = { url, init };
      return {
        ok: true,
        status: 200,
        text: async () => JSON.stringify({
          answers: { a: { type: "noul", noul: 0.7 } },
          usage: { input_tokens: 42, output_tokens: 0 },
          latency_ms: 12.5,
          routing: { model: "typed-decisions" },
          model: "laya-typed-decisions",
        }),
      };
    },
  });
  const out = await client.ask("a state", { a: { type: "noul", instructions: "x" } });
  assert.equal(sent.url, "http://example.invalid/v1/systemone");
  assert.equal(sent.init.method, "POST");
  assert.deepEqual(JSON.parse(sent.init.body).model, "typed-decisions");
  assert.equal(JSON.parse(sent.init.body).state, "a state");
  assert.equal(out.inference_ms, 12.5);
  assert.ok(out.roundtrip_ms >= 0);
  assert.equal(client.counters.calls, 1);
  assert.equal(client.counters.tokens, 42);
});

test("a server error is raised, not swallowed", async () => {
  const client = createClient({
    baseUrl: "http://example.invalid",
    fetchImpl: async () => ({ ok: false, status: 422, text: async () => '{"error":{"message":"bad"}}' }),
  });
  await assert.rejects(() => client.ask("s", { a: { type: "noul", instructions: "x" } }), /systemone 422/);
  assert.equal(client.counters.errors, 1);
});
