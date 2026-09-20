// Shared by the test files, and not a test file itself, so the runner does not pick it up twice.

export const GAMES = ["snake", "hopper", "crossing", "paddle", "mines", "dungeon"];
export const SEEDS = [1, 2, 3, 7, 42];

export async function load(game) {
  return import(`../${game}/logic.mjs`);
}

// Plays a whole episode with a fixed policy and returns every state it went through.
export function play(logic, seed, pick, maxSteps = 120) {
  let state = logic.init(seed);
  const trace = [state];
  const actions = [];
  for (let i = 0; i < maxSteps; i += 1) {
    const action = pick(state, i);
    const guarded = logic.shield(state, action, {}).action;
    actions.push(guarded);
    const out = logic.step(state, guarded);
    state = out.state;
    trace.push(state);
    if (out.done) break;
  }
  return { trace, actions, state };
}

// What the server would answer, without a server: a flat choice, a half noul, a middle score.
export function fakeAnswers(questions, bias = null) {
  const answers = {};
  for (const [id, q] of Object.entries(questions)) {
    if (q.type === "choice") {
      const options = Object.keys(q.criteria);
      const probabilities = {};
      options.forEach((o, i) => {
        probabilities[o] = o === bias ? 0.9 : (1 - (bias ? 0.9 : 0)) / (options.length - (bias ? 1 : 0));
        if (!bias) probabilities[o] = (options.length - i) / ((options.length * (options.length + 1)) / 2);
      });
      answers[id] = { type: "choice", choice: bias || options[0], probabilities, confidence: 0.1 };
    } else if (q.type === "noul") {
      answers[id] = { type: "noul", noul: 0.5, confidence: 0.5 };
    } else {
      const probabilities = {};
      q.criteria.forEach((_, i) => { probabilities[i] = 1 / q.criteria.length; });
      answers[id] = { type: "score", score: 0.5, probabilities, confidence: 0.1 };
    }
  }
  return answers;
}
