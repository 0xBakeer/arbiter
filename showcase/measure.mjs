#!/usr/bin/env node
// Runs whole episodes against a live server and writes down what happened.
//
//   node showcase/measure.mjs --game snake --episodes 20 \
//        --policies model+shield,random+shield,heuristic+shield --base http://localhost:8010
//
// Per game it writes showcase/<game>/measurements.json and replaces the table between the
// <!-- measurements --> markers in showcase/<game>/README.md. The best model episode is saved to
// showcase/replay/<game>.json so the page can play a recorded run with no server.
//
// Two speed numbers are reported and they mean different things. `decisions_per_s` is taken from
// the median round trip, which is what a running game feels like. `throughput_per_s` is total
// decisions over total time waited, which the occasional 2-second outlier on a laptop pulls down
// hard. Both are in the JSON; the table shows the median one and the p95 next to it.

import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

import { createClient } from "./_lib/client.mjs";
import { runEpisode, policyParts } from "./_lib/harness.mjs";
import { createRecorder } from "./_lib/recorder.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));

// Episodes per game, chosen so the whole sweep fits in about a quarter of an hour on the Mac
// server. The step cap lives in each game's `meta.max_steps`, because the page needs it too.
const GAMES = {
  snake: { episodes: 20 },
  hopper: { episodes: 20 },
  crossing: { episodes: 20 },
  paddle: { episodes: 20 },
  mines: { episodes: 20 },
  dungeon: { episodes: 20 },
};

const DEFAULT_POLICIES = ["model", "model+shield", "random+shield", "heuristic+shield"];

async function main() {
  const { values } = parseArgs({
    options: {
      game: { type: "string", default: "all" },
      episodes: { type: "string" },
      policies: { type: "string" },
      base: { type: "string", default: "http://localhost:8010" },
      model: { type: "string", default: "auto" },
      "max-steps": { type: "string" },
      seed: { type: "string", default: "1" },
      "no-readme": { type: "boolean", default: false },
      "render-only": { type: "boolean", default: false },
      "no-replay": { type: "boolean", default: false },
    },
  });

  const games = values.game === "all" ? Object.keys(GAMES) : values.game.split(",");
  // Re-render the README tables from the measurements already on disk, without touching a server.
  if (values["render-only"]) {
    for (const game of games) {
      const saved = JSON.parse(await readFile(join(HERE, game, "measurements.json"), "utf8"));
      await patchReadme(game, saved);
      console.error(`${game}: README table rendered from ${game}/measurements.json`);
    }
    return;
  }

  const client = createClient({ baseUrl: values.base, model: values.model });
  const health = await client.ready().catch((err) => ({ ready: false, error: String(err) }));
  if (!health.ready) {
    console.error(`server at ${values.base} is not ready: ${JSON.stringify(health)}`);
    process.exit(1);
  }
  console.error(`server ${values.base}: ${health.models} on ${health.device} (${health.mode}/${health.dtype})`);

  for (const game of games) {
    if (!GAMES[game]) throw new Error(`unknown game ${game}; known: ${Object.keys(GAMES).join(", ")}`);
    await measureGame(game, client, values, health);
  }
}

async function measureGame(game, client, values, health) {
  const logic = await import(`./${game}/logic.mjs`);
  const defaults = GAMES[game];
  const episodes = Number(values.episodes || defaults.episodes);
  const maxSteps = Number(values["max-steps"] || logic.meta.max_steps);
  const seed0 = Number(values.seed);
  let policies = values.policies ? values.policies.split(",") : DEFAULT_POLICIES.slice();
  // A game without a shield would report the same row twice.
  if (logic.meta.shield === "none") policies = policies.filter((p) => p !== "model+shield");

  const started = Date.now();
  const rows = {};
  let bestReplay = null;

  for (const policy of policies) {
    const { source } = policyParts(policy);
    const scores = [];
    const stepCounts = [];
    const decisions = [];
    const interventions = [];
    const inference = [];
    const roundtrip = [];
    const pairs = {};                     // question id -> calibration samples
    let tokens = 0;

    for (let i = 0; i < episodes; i += 1) {
      const seed = seed0 + i;
      const recorder = source === "model"
        ? createRecorder({ game, policy, seed, meta: logic.meta, base: values.base, model: health.models })
        : null;
      const collected = [];
      const out = await runEpisode(logic, policy, {
        client,
        seed,
        maxSteps,
        recorder,
        onTick: (tick) => {
          if (tick.answers && tick.truth) collected.push({ answers: tick.answers, truth: tick.truth, state: tick.state });
        },
      });
      scores.push(out.summary.score);
      stepCounts.push(out.stats.steps);
      decisions.push(out.stats.decisions);
      interventions.push(out.stats.interventions);
      inference.push(...out.stats.inferences);
      roundtrip.push(...out.stats.roundtrips);
      tokens += out.stats.tokens;
      for (const tick of collected) gather(pairs, logic, tick);
      if (recorder && (!bestReplay || out.summary.score > bestReplay.summary.score)) {
        bestReplay = recorder.log;
      }
      process.stderr.write(
        `\r${game} ${policy} ${i + 1}/${episodes} score ${out.summary.score} steps ${out.stats.steps}      `);
    }
    process.stderr.write("\n");

    rows[policy] = {
      episodes,
      max_steps: maxSteps,
      score: { mean: mean(scores), median: median(scores), max: Math.max(...scores), all: scores },
      steps: { mean: mean(stepCounts), median: median(stepCounts), max: Math.max(...stepCounts) },
      decisions: { total: sum(decisions), per_episode: mean(decisions) },
      shield_interventions: { total: sum(interventions), per_episode: mean(interventions) },
      input_tokens: tokens,
      inference_ms: quantiles(inference),
      roundtrip_ms: quantiles(roundtrip),
      decisions_per_s: roundtrip.length ? round(1000 / median(roundtrip), 1) : null,
      throughput_per_s: roundtrip.length ? round((roundtrip.length * 1000) / sum(roundtrip), 1) : null,
      calibration: calibrate(pairs),
    };
  }

  const measurements = {
    game,
    generated_at: new Date().toISOString(),
    server: { base: values.base, model: values.model, device: health.device, mode: health.mode, dtype: health.dtype },
    tick_ms: logic.meta.tick_ms,
    questions_per_tick: Object.keys(logic.questions(logic.init(seed0))).length,
    wall_seconds: round((Date.now() - started) / 1000, 1),
    policies: rows,
  };
  await writeFile(join(HERE, game, "measurements.json"), JSON.stringify(measurements, null, 2) + "\n");
  if (!values["no-readme"]) await patchReadme(game, measurements);
  if (!values["no-replay"] && bestReplay) {
    await mkdir(join(HERE, "replay"), { recursive: true });
    await writeFile(join(HERE, "replay", `${game}.json`), JSON.stringify(bestReplay) + "\n");
  }
  console.error(`${game}: ${measurements.wall_seconds}s -> ${game}/measurements.json`);
}

// -- calibration -------------------------------------------------------------

// A tick contributes one sample per question that has both an answer and a ground truth.
function gather(pairs, logic, tick) {
  const baseline = logic.baseline && tick.state ? logic.baseline(tick.state) : null;
  const group = logic.meta.calibration_group || null;
  for (const [id, label] of Object.entries(tick.truth)) {
    const answer = tick.answers[id];
    if (!answer) continue;
    // Games that ask the same question about many things (mines asks about twelve cells) pool
    // their answers into one table; everything else keeps a table per question id.
    const key = group || id;
    const bucket = pairs[key] || (pairs[key] = { type: answer.type, samples: [] });
    if (answer.type === "noul") {
      bucket.samples.push({ p: answer.noul, label: label ? 1 : 0, baseline: baseline ? baseline[id] : null });
    } else if (answer.type === "score") {
      const levels = Object.keys(answer.probabilities || {}).length;
      bucket.levels = levels;
      bucket.samples.push({
        p: answer.score,
        level: Number(Object.entries(answer.probabilities).sort((a, b) => b[1] - a[1])[0][0]),
        label,
        top: label >= levels - 1 ? 1 : 0,
      });
    }
  }
}

function calibrate(pairs) {
  const out = {};
  for (const [id, bucket] of Object.entries(pairs)) {
    if (!bucket.samples.length) continue;
    if (bucket.type === "noul") {
      const table = { true_positive: 0, false_positive: 0, false_negative: 0, true_negative: 0 };
      for (const s of bucket.samples) {
        if (s.p >= 0.5 && s.label) table.true_positive += 1;
        else if (s.p >= 0.5) table.false_positive += 1;
        else if (s.label) table.false_negative += 1;
        else table.true_negative += 1;
      }
      const baselineSamples = bucket.samples.filter((s) => s.baseline != null);
      out[id] = {
        type: "noul",
        n: bucket.samples.length,
        positives: bucket.samples.filter((s) => s.label).length,
        mean_when_true: round(mean(bucket.samples.filter((s) => s.label).map((s) => s.p)), 3),
        mean_when_false: round(mean(bucket.samples.filter((s) => !s.label).map((s) => s.p)), 3),
        auroc: round(auroc(bucket.samples.map((s) => [s.p, s.label])), 3),
        baseline_auroc: baselineSamples.length
          ? round(auroc(baselineSamples.map((s) => [s.baseline, s.label])), 3) : null,
        table,
      };
    } else {
      const levels = bucket.levels || 3;
      const matrix = Array.from({ length: levels }, () => new Array(levels).fill(0));
      for (const s of bucket.samples) matrix[s.label][s.level] += 1;
      out[id] = {
        type: "score",
        n: bucket.samples.length,
        levels,
        auroc_top_level: round(auroc(bucket.samples.map((s) => [s.p, s.top])), 3),
        exact_match: round(bucket.samples.filter((s) => s.level === s.label).length / bucket.samples.length, 3),
        matrix,                           // matrix[true level][predicted level]
      };
    }
  }
  return out;
}

// Probability that a random positive scores above a random negative. Ties count a half.
function auroc(rows) {
  const positives = rows.filter(([, l]) => l);
  const negatives = rows.filter(([, l]) => !l);
  if (!positives.length || !negatives.length) return null;
  let wins = 0;
  for (const [p] of positives) for (const [n] of negatives) wins += p > n ? 1 : p === n ? 0.5 : 0;
  return wins / (positives.length * negatives.length);
}

// -- numbers and markdown ----------------------------------------------------

const sum = (a) => a.reduce((x, y) => x + y, 0);
const mean = (a) => (a.length ? round(sum(a) / a.length, 2) : null);
const round = (v, n) => (v == null ? null : Math.round(v * 10 ** n) / 10 ** n);

function median(a) {
  if (!a.length) return null;
  const s = a.slice().sort((x, y) => x - y);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

function quantiles(a) {
  if (!a.length) return { n: 0, p50: null, p95: null, max: null };
  const s = a.slice().sort((x, y) => x - y);
  const at = (q) => round(s[Math.min(s.length - 1, Math.floor(q * s.length))], 1);
  return { n: s.length, p50: at(0.5), p95: at(0.95), max: round(s[s.length - 1], 1) };
}

function table(measurements) {
  const head = [
    "| policy | score mean | median | best | steps mean | decisions/ep | decisions/s | p50 inference | p95 inference | p50 round trip | shield/ep |",
    "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
  ];
  const rows = Object.entries(measurements.policies).map(([policy, r]) => {
    const dash = (v, unit = "") => (v == null ? "-" : `${v}${unit}`);
    return `| \`${policy}\` | ${r.score.mean} | ${r.score.median} | ${r.score.max} | ${r.steps.mean} | `
      + `${r.decisions.per_episode} | ${dash(r.decisions_per_s)} | ${dash(r.inference_ms.p50, " ms")} | `
      + `${dash(r.inference_ms.p95, " ms")} | ${dash(r.roundtrip_ms.p50, " ms")} | `
      + `${r.shield_interventions.per_episode} |`;
  });
  return head.concat(rows).join("\n");
}

function calibrationText(measurements) {
  const out = [];
  for (const [policy, row] of Object.entries(measurements.policies)) {
    if (!policy.startsWith("model")) continue;
    const entries = Object.entries(row.calibration);
    if (!entries.length) continue;
    const nouls = entries.filter(([, c]) => c.type === "noul");
    const scores = entries.filter(([, c]) => c.type === "score");
    if (nouls.length) {
      out.push("", `Nouls against ground truth, policy \`${policy}\`, at the 0.5 threshold:`, "");
      out.push("| question | answers | AUROC | mean when true | mean when false | true pos | false pos | false neg | true neg |");
      out.push("| --- | --- | --- | --- | --- | --- | --- | --- | --- |");
      for (const [id, c] of nouls) {
        out.push(`| \`${id}\` | ${c.n} | ${c.auroc ?? "-"} | ${c.mean_when_true ?? "-"} | ${c.mean_when_false ?? "-"} | `
          + `${c.table.true_positive} | ${c.table.false_positive} | ${c.table.false_negative} | ${c.table.true_negative} |`);
      }
      const withBaseline = nouls.filter(([, c]) => c.baseline_auroc != null);
      for (const [id, c] of withBaseline) {
        out.push("", `The same ranking done by the number the question was handed (\`baseline\`, see `
          + `\`logic.mjs\`) scores AUROC ${c.baseline_auroc} on \`${id}\`, against the model's ${c.auroc}.`);
      }
    }
    for (const [id, c] of scores) {
      out.push("", `Score \`${id}\`, ${c.n} answers, exact level match ${c.exact_match}, `
        + `AUROC ${c.auroc_top_level ?? "-"} for the top level. Rows are the true level, columns the answered one:`, "");
      out.push("| true \\ answered | " + c.matrix[0].map((_, i) => i).join(" | ") + " |");
      out.push("| --- |" + c.matrix[0].map(() => " --- |").join(""));
      c.matrix.forEach((row, i) => out.push(`| ${i} | ${row.join(" | ")} |`));
    }
  }
  return out.join("\n");
}

async function patchReadme(game, measurements) {
  const path = join(HERE, game, "README.md");
  let text;
  try {
    text = await readFile(path, "utf8");
  } catch {
    return;                               // the README is written by hand; nothing to patch yet
  }
  const marker = "<!-- measurements -->";
  const parts = text.split(marker);
  if (parts.length !== 3) return;
  const server = measurements.server;
  const body = [
    "",
    `${measurements.policies[Object.keys(measurements.policies)[0]].episodes} episodes per policy, `
      + `seeds ${1}..${measurements.policies[Object.keys(measurements.policies)[0]].episodes}, `
      + `against ${server.base} (${server.device}, ${server.mode}/${server.dtype}), `
      + `${measurements.wall_seconds} s of wall clock. Generated by \`showcase/measure.mjs\` on `
      + `${measurements.generated_at.slice(0, 10)}.`,
    "",
    table(measurements),
    "",
    calibrationText(measurements).trim(),
    "",
  ].join("\n");
  await writeFile(path, parts[0] + marker + body + marker + parts[2]);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
