// One seeded generator for every game, so a seed reproduces a run exactly in node and in the
// browser. mulberry32: 32 bits of state, carried inside the game state so `step` stays a pure
// function of (state, action).
//
// Each state keeps two independent streams:
//   state.rnd   the world (food placement, car spawns, dice rolls)
//   state.prnd  the baseline policies, so `random` and `heuristic` do not shift the world
//               stream and every policy sees the same world for the same seed.

export function seedStreams(seed) {
  const s = (seed >>> 0) || 0x9e3779b9;
  return { rnd: s, prnd: (s ^ 0x5bf03635) >>> 0 };
}

function mulberry(x) {
  let t = (x + 0x6d2b79f5) >>> 0;
  let z = t;
  z = Math.imul(z ^ (z >>> 15), z | 1);
  z ^= z + Math.imul(z ^ (z >>> 7), z | 61);
  return [t, ((z ^ (z >>> 14)) >>> 0) / 4294967296];
}

// Advances the world stream and returns a float in [0, 1).
export function rnd(state) {
  const [next, value] = mulberry(state.rnd);
  state.rnd = next;
  return value;
}

// Advances the policy stream. Only the baseline policies call this.
export function prnd(state) {
  const [next, value] = mulberry(state.prnd);
  state.prnd = next;
  return value;
}

export function rndInt(state, n) {
  return Math.floor(rnd(state) * n);
}

export function prndPick(state, items) {
  return items[Math.floor(prnd(state) * items.length)];
}
