// A run log is a seed plus one entry per tick. Because every `logic.mjs` is deterministic, the
// actions in the log are enough to rebuild every board position: `replay` re-simulates from the
// seed and hands the caller the real state, so a recorded run renders exactly like a live one
// with no server in the loop. The summaries, answers and timings ride along for the panel.

const now = () => (globalThis.performance ? globalThis.performance.now() : Date.now());
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

export function createRecorder({ game, policy, seed, meta = null, base = null, model = null } = {}) {
  const log = {
    game,
    policy,
    seed,
    model,
    base,
    tick_ms: meta ? meta.tick_ms : null,
    recorded_at: new Date().toISOString(),
    meta,
    ticks: [],
    summary: null,
    stats: null,
  };

  return {
    log,
    tick(entry) {
      log.ticks.push(entry);
    },
    finish(summary, stats) {
      log.summary = summary;
      log.stats = stats || null;
      return log;
    },
  };
}

// Walks a recorded log at its own pace. Pass `logic` to get the rebuilt state per tick.
export async function replay(log, { onTick = null, logic = null, tickMs = null, signal = null } = {}) {
  const wait = tickMs == null ? log.tick_ms || 0 : tickMs;
  let state = logic ? logic.init(log.seed) : null;
  for (let i = 0; i < log.ticks.length; i += 1) {
    if (signal && signal.aborted) break;
    const started = now();
    const entry = log.ticks[i];
    // Same two states a live tick hands the UI: the board the decision was made on, and the
    // board it produced.
    const next = logic ? logic.step(state, entry.action).state : null;
    if (onTick) await onTick({ ...entry, index: i, total: log.ticks.length, state, next });
    state = next;
    const left = wait - (now() - started);
    if (left > 0) await sleep(left);
  }
  return { state, summary: log.summary };
}
