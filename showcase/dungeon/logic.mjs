// Thirty rooms of prose. This is the one game whose state is native to a text encoder: the room
// is a paragraph, the options are sentences about that paragraph, and nothing is a grid. The
// model gets the room text plus a choice of three to five actions and a three-level danger
// score; the shield only steps in when a fight would certainly be the last one.

import { seedStreams, rnd, prnd } from "../_lib/prng.mjs";

export const meta = {
  id: "dungeon",
  name: "Dungeon",
  tick_ms: 500,
  max_steps: 150,
  description: "A rogue-lite in prose. Three to five actions per room, plus a danger score.",
  // A map rather than a list, because the panel draws these glyphs instead of the first three
  // letters of the action name.
  actions: {
    fight: { glyph: "\u2694" },
    flee: { glyph: "\u21a9" },
    loot: { glyph: "\u25c6" },
    heal: { glyph: "+" },
    descend: { glyph: "\u2193" },
  },
  shield: "vetoes a fight, a loot or a heal that one hit could end",
};

const MAX_HP = 20;
const ROOMS = 30;
const POTION_HEAL = 8;

const PLACES = [
  "a flooded undercroft, knee deep in black water",
  "a collapsed library, shelves fallen against each other",
  "a hall of broken statues facing the wrong way",
  "a narrow gallery above a drop with no bottom in sight",
  "a kitchen gone cold centuries ago, pots still hanging",
  "a chapel with the altar hacked apart",
  "a stable where something much larger than a horse was kept",
  "a corridor of doors, all of them nailed shut but one",
];
const MONSTERS = [
  { name: "a rat the size of a dog", strength: 1 },
  { name: "a ghoul with a broken spear", strength: 2 },
  { name: "a rusted iron sentry", strength: 3 },
  { name: "a cave troll, half asleep", strength: 4 },
  { name: "a wight in a mouldering crown", strength: 5 },
];
const TREASURES = [
  "a chest sits half submerged against the wall",
  "coins are scattered where someone dropped a purse",
  "a strongbox has been pried open and abandoned",
  "a corpse still wears a heavy gold chain",
];

export function init(seed) {
  const state = {
    hp: MAX_HP,
    max_hp: MAX_HP,
    gold: 0,
    potions: 2,
    depth: 1,
    rooms: ROOMS,
    room: null,
    alive: true,
    escaped: false,
    steps: 0,
    ...seedStreams(seed),
  };
  state.room = makeRoom(state);
  return state;
}

function makeRoom(state) {
  const place = PLACES[Math.floor(rnd(state) * PLACES.length)];
  // Deeper rooms lean towards the stronger half of the bestiary.
  const hasMonster = rnd(state) < 0.7;
  let monster = null;
  if (hasMonster) {
    const floor = Math.min(MONSTERS.length - 1, Math.floor((state.depth - 1) / 8));
    const pick = floor + Math.floor(rnd(state) * (MONSTERS.length - floor));
    const base = MONSTERS[pick];
    monster = {
      name: base.name,
      strength: base.strength,
      hp: 2 + base.strength * 2,
      max_hit: base.strength + 2,
      gold: base.strength * 6 + Math.floor(rnd(state) * 10),
    };
  }
  const treasure = rnd(state) < 0.55 ? {
    text: TREASURES[Math.floor(rnd(state) * TREASURES.length)],
    gold: 8 + Math.floor(rnd(state) * 30),
    potion: rnd(state) < 0.3,
  } : null;
  return { place, monster, treasure, looted: false };
}

function roomText(state) {
  const { room } = state;
  const parts = [`Room ${state.depth} of ${ROOMS}: ${room.place}.`];
  if (room.monster) parts.push(`${capitalise(room.monster.name)} blocks the way.`);
  if (room.treasure && !room.looted) parts.push(`${capitalise(room.treasure.text)}.`);
  if (!room.monster) parts.push("A stair leads further down.");
  return parts.join(" ");
}

const capitalise = (s) => s[0].toUpperCase() + s.slice(1);

// -- the contract ------------------------------------------------------------

export function legal(state) {
  const { room } = state;
  const out = [];
  if (room.monster) out.push("fight", "flee");
  if (room.treasure && !room.looted) out.push("loot");
  if (state.potions > 0 && state.hp < state.max_hp) out.push("heal");
  if (!room.monster) out.push("descend");
  return out;
}

export function encode(state) {
  return [
    roomText(state),
    `You have ${state.hp} of ${state.max_hp} hit points, ${state.potions} potions and ${state.gold} gold.`,
  ].join(" ");
}

function moveFacts(state, action) {
  const { room } = state;
  const m = room.monster;
  if (action === "fight") {
    const verdict = m.max_hit >= state.hp ? "deadly" : m.max_hit * 2 >= state.hp ? "risky" : "safe";
    return `${verdict}: it hits for up to ${m.max_hit} and you have ${state.hp} hit points, `
      + `it needs about ${Math.ceil(m.hp / 3)} more rounds to go down`;
  }
  if (action === "flee") {
    return `safe: you run for the stair and leave the room behind, it may land one parting blow of up to ${m.max_hit}`;
  }
  if (action === "loot") {
    return m
      ? `risky: you reach for the treasure with ${m.name} still standing, it hits for up to ${m.max_hit}`
      : "safe: nothing is watching, the treasure is yours";
  }
  if (action === "heal") {
    const danger = m ? `, but ${m.name} hits for up to ${m.max_hit} while you drink` : "";
    return `${m ? "risky" : "safe"}: a potion puts back ${POTION_HEAL} hit points and you have `
      + `${state.potions}${danger}`;
  }
  return `safe: the stair goes down to room ${state.depth + 1}, you leave with ${state.hp} hit points`;
}

export function questions(state) {
  const options = legal(state);
  const criteria = {};
  for (const action of options) criteria[action] = moveFacts(state, action);
  // A choice needs two options. When the room leaves one thing to do there is nothing to choose,
  // so only the danger score goes out and `decide` takes the forced action.
  const move = options.length < 2 ? {} : {
    move: {
      type: "choice",
      instructions: "You are working your way down thirty rooms. Pick the action that gets deeper "
        + "and richer without getting killed.",
      criteria,
    },
  };
  return {
    ...move,
    danger: {
      type: "score",
      instructions: "How dangerous is this room for you right now?",
      criteria: ["nothing here can hurt you badly", "a fight would cost real blood", "one more hit would kill you"],
    },
  };
}

export function decide(state, answers) {
  const options = legal(state);
  const probabilities = (answers.move && answers.move.probabilities) || {};
  if (!answers.move) {
    return {
      action: options[0],
      extras: {
        probabilities: {},
        forced: true,
        danger: answers.danger ? answers.danger.score : null,
        danger_level: answers.danger ? topLevel(answers.danger.probabilities) : null,
      },
    };
  }
  let action = options[0];
  let best = -1;
  for (const option of options) {
    const p = probabilities[option] || 0;
    if (p > best) {
      best = p;
      action = option;
    }
  }
  return {
    action,
    extras: {
      probabilities,
      confidence: answers.move ? answers.move.confidence : null,
      danger: answers.danger ? answers.danger.score : null,
      danger_level: answers.danger ? topLevel(answers.danger.probabilities) : null,
    },
  };
}

function topLevel(probabilities) {
  if (!probabilities) return null;
  return Number(Object.entries(probabilities).sort((a, b) => b[1] - a[1])[0][0]);
}

// The one rule: do not stand in front of something that can kill you in a single blow when
// there is any other door out of the room.
export function shield(state, action, extras = {}) {
  const m = state.room.monster;
  const risky = (action === "fight" || action === "loot" || action === "heal") && m && m.max_hit >= state.hp;
  if (!risky) return { action, intervened: false, reason: "" };
  const safe = legal(state).filter((a) => a === "flee" || (a === "heal" && m.max_hit < state.hp));
  if (!safe.length) return { action, intervened: false, reason: "nothing safe left to do" };
  const probabilities = extras.probabilities || {};
  const ranked = safe.slice().sort((a, b) => (probabilities[b] || 0) - (probabilities[a] || 0));
  return {
    action: ranked[0],
    intervened: true,
    reason: `${m.name} hits for up to ${m.max_hit} and you have ${state.hp}; took ${ranked[0]}`,
  };
}

export function step(state, action) {
  const next = structuredClone(state);
  const events = [];
  const chosen = legal(state).includes(action) ? action : legal(state)[0];
  next.steps += 1;
  const m = next.room.monster;

  if (chosen === "fight") {
    const damage = 2 + Math.floor(rnd(next) * 3);
    m.hp -= damage;
    events.push(`you hit ${m.name} for ${damage}`);
    if (m.hp <= 0) {
      next.gold += m.gold;
      events.push(`${m.name} falls, ${m.gold} gold`);
      next.room.monster = null;
    } else {
      hurt(next, m, events);
    }
  } else if (chosen === "flee") {
    if (rnd(next) < 0.4) hurt(next, m, events, "as you turn");
    if (next.alive) {
      events.push("you slip past into the next room");
      descend(next);
    }
  } else if (chosen === "loot") {
    const t = next.room.treasure;
    next.gold += t.gold;
    if (t.potion) next.potions += 1;
    next.room.looted = true;
    events.push(`${t.gold} gold${t.potion ? " and a potion" : ""}`);
    if (m) hurt(next, m, events);
  } else if (chosen === "heal") {
    next.potions -= 1;
    next.hp = Math.min(next.max_hp, next.hp + POTION_HEAL);
    events.push(`the potion puts you back to ${next.hp}`);
    if (m) hurt(next, m, events);
  } else {
    descend(next);
    events.push(`down to room ${next.depth}`);
  }

  if (!next.alive) return { state: next, reward: -1, done: true, events };
  if (next.escaped) return { state: next, reward: 1, done: true, events };
  return { state: next, reward: chosen === "descend" || chosen === "flee" ? 1 : 0, done: false, events };
}

function hurt(state, monster, events, when = "") {
  const damage = 1 + Math.floor(rnd(state) * monster.max_hit);
  state.hp -= damage;
  events.push(`${monster.name} hits you for ${damage}${when ? " " + when : ""}`);
  if (state.hp <= 0) {
    state.hp = 0;
    state.alive = false;
    events.push("you die here");
  }
}

function descend(state) {
  if (state.depth >= ROOMS) {
    state.escaped = true;
    return;
  }
  state.depth += 1;
  state.room = makeRoom(state);
}

export function random(state) {
  const options = legal(state);
  return options[Math.floor(prnd(state) * options.length)];
}

export function heuristic(state) {
  const options = legal(state);
  const m = state.room.monster;
  if (options.includes("heal") && state.hp <= state.max_hp * 0.4) return "heal";
  if (m && m.max_hit >= state.hp) return options.includes("flee") ? "flee" : options[0];
  if (options.includes("loot") && !m) return "loot";
  if (m) return m.max_hit * 2 < state.hp || m.strength <= 2 ? "fight" : "flee";
  return options.includes("descend") ? "descend" : options[0];
}

// Ground truth for the danger score, from the numbers the room was built with.
export function truth(state) {
  const m = state.room.monster;
  if (!m) return { danger: 0 };
  return { danger: m.max_hit >= state.hp ? 2 : m.max_hit * 2 >= state.hp ? 1 : 0 };
}

export function summary(state) {
  return {
    score: state.gold + state.depth * 10,
    gold: state.gold,
    depth: state.depth,
    hp: state.hp,
    potions: state.potions,
    steps: state.steps,
    alive: state.alive,
    escaped: state.escaped,
  };
}
