// The loop every game and every measurement runs through.
//
//   encode(state) -> questions(state) -> one POST -> decide(state, answers) -> shield -> step
//
// A policy is who answers the choice: the model, or one of the two baselines. The "+shield"
// suffix puts the code-owned safety check between the decision and the step. The shield is the
// honest part of the demo: it is plain code, it is counted, and its interventions are reported
// per episode so nobody can mistake them for the model playing well.

const now = () => (globalThis.performance ? globalThis.performance.now() : Date.now());
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

export const POLICIES = ["model", "model+shield", "random+shield", "heuristic+shield"];

const NO_SHIELD = { intervened: false, reason: "" };

export function policyParts(policy) {
  if (!POLICIES.includes(policy)) {
    throw new Error(`unknown policy ${policy}; expected one of ${POLICIES.join(", ")}`);
  }
  return { source: policy.split("+")[0], shielded: policy.endsWith("+shield") };
}

export async function runEpisode(logic, policy, {
  client = null,
  seed = 1,
  onTick = null,
  maxSteps = 400,
  tickMs = 0,
  recorder = null,
  signal = null,
  window = 20,
} = {}) {
  const { source, shielded } = policyParts(policy);
  if (source === "model" && !client) throw new Error(`policy ${policy} needs a client`);

  let state = logic.init(seed);
  const roundtrips = [];
  const inferences = [];
  const counters = { decisions: 0, interventions: 0, tokens: 0, questions: 0, calls: 0 };
  let reward = 0;
  let done = false;
  let step = 0;
  let lastEvents = [];

  while (!done && step < maxSteps) {
    if (signal && signal.aborted) break;
    const started = now();

    let answers = null;
    let encoded = null;
    let timing = null;
    let decision;

    if (source === "model") {
      encoded = logic.encode(state);
      const questions = logic.questions(state);
      const out = await client.ask(encoded, questions);
      answers = out.answers;
      roundtrips.push(out.roundtrip_ms);
      inferences.push(out.inference_ms);
      counters.tokens += (out.usage && out.usage.input_tokens) || 0;
      counters.questions += Object.keys(questions).length;
      counters.calls += 1;
      timing = {
        inference_ms: out.inference_ms,
        roundtrip_ms: out.roundtrip_ms,
        decisions_per_s: rate(roundtrips, window),
      };
      decision = logic.decide(state, answers);
    } else {
      // The baselines see the same state object, so they draw from the policy stream and leave
      // the world stream untouched (see _lib/prng.mjs).
      decision = { action: source === "random" ? logic.random(state) : logic.heuristic(state), extras: {} };
    }

    const guard = shielded
      ? logic.shield(state, decision.action, decision.extras)
      : { action: decision.action, ...NO_SHIELD };
    if (guard.intervened) counters.interventions += 1;
    counters.decisions += 1;

    // Ground truth is about what the model wanted, not about what the shield allowed: that is
    // the pair the calibration table needs.
    const truth = logic.truth ? logic.truth(state, decision.action) : null;

    const before = logic.summary(state);
    const out = logic.step(state, guard.action);
    reward += out.reward;
    done = out.done;
    lastEvents = out.events;
    step += 1;

    const entry = {
      step,
      action: guard.action,
      model_action: decision.action,
      intervened: guard.intervened,
      reason: guard.reason,
      extras: decision.extras || {},
      answers,
      encoded,
      truth,
      timing,
      events: out.events,
      reward: out.reward,
      done: out.done,
      summary: logic.summary(out.state),
    };
    if (recorder) recorder.tick(entry);
    if (onTick) await onTick({ ...entry, state, next: out.state, before, counters });

    state = out.state;
    if (tickMs > 0) {
      const left = tickMs - (now() - started);
      if (left > 0) await sleep(left);
    }
  }

  const summary = logic.summary(state);
  const stats = {
    policy,
    seed,
    steps: step,
    reward,
    done,
    ...counters,
    roundtrips,
    inferences,
    decisions_per_s: rate(roundtrips, roundtrips.length),
  };
  if (recorder) recorder.finish(summary, stats);
  return { state, summary, stats, events: lastEvents };
}

// Decisions per second from the round-trips we actually waited for, over the last `n` of them.
function rate(roundtrips, n) {
  if (!roundtrips.length) return null;
  const slice = roundtrips.slice(-Math.max(1, n));
  const total = slice.reduce((a, b) => a + b, 0);
  return total > 0 ? (slice.length * 1000) / total : null;
}
