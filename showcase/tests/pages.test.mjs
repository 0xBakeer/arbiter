// The runner that lives inside the six game pages. It is plain text in an HTML file, so it is
// read as text: the block between the GENERIC PART banner and the per-game banner has to stay
// the same on all six pages, and the one rule the harness and the page must agree on -- an
// episode ends at meta.max_steps -- is pulled out of the page and run.
import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

import { GAMES, load } from "./helpers.mjs";

const pages = {};
for (const game of GAMES) {
  pages[game] = await readFile(new URL(`../${game}/index.html`, import.meta.url), "utf8");
}

// The generic block, with the handful of lines that are per-game by design normalised away:
// the game's name, its recording, its best-score key and the second HUD stat.
function genericBlock(game, html) {
  const from = html.indexOf("   GENERIC PART");
  const to = html.indexOf("/* =====", from);
  assert.ok(from > 0 && to > from, `${game}: no generic block`);
  return html.slice(from, to)
    .replace(new RegExp(game, "gi"), "<game>")
    .replace(/countUp\(\$\('hudSecond'\)[^\n]*\n|countUp\(\$\('hudLength'\)[^\n]*\n/, "<second stat>\n");
}

// The runner proper: the run state, and then everything from the pulse down to the key map --
// the ask, the tick, the HUD, the frame loop. paintStatus and the key map in between are left
// out because they legitimately differ: hopper and mines have no shield to toggle.
function runner(block) {
  return [["/* ---- run state ---- */", "function paintStatus("],
          ["/* ---- the cobalt pulse", "/* ---- keys ---- */"]]
    .flatMap(([open, close]) => {
      const from = block.indexOf(open);
      const to = block.indexOf(close, from);
      assert.ok(from > 0 && to > from, `no ${open} in the generic block`);
      return block.slice(from, to).split("\n");
    });
}

test("the six pages carry the same runner", () => {
  const snake = runner(genericBlock("snake", pages.snake));
  for (const game of GAMES) {
    if (game === "snake") continue;
    const lines = runner(genericBlock(game, pages[game]));
    const at = lines.findIndex((line, i) => line !== snake[i]);
    assert.equal(at, -1, `${game}'s runner drifted from snake's at line ${at + 1}:\n` +
      `  snake:  ${snake[at]}\n  ${game}: ${lines[at]}`);
    assert.equal(lines.length, snake.length, `${game}'s runner is a different length`);
  }
});

test("an episode ends at max_steps, exactly as the harness ends it", async () => {
  for (const game of GAMES) {
    const line = pages[game].match(/^ {2}const over = .*$/m);
    assert.ok(line, `${game}: the page no longer decides when an episode is over`);
    const over = new Function("run", "ep", "MAX_STEPS", `${line[0]}; return over;`);

    const { meta } = await load(game);
    const live = { mode: "live" };
    assert.equal(over(live, { i: meta.max_steps - 1 }, meta.max_steps), false);
    assert.equal(over(live, { i: meta.max_steps }, meta.max_steps), true);
    // One step over the cap is what the HUD used to show (paddle read 307 / 300).
    assert.equal(over(live, { i: meta.max_steps + 7 }, meta.max_steps), true);

    // A recorded run is unaffected: it ends with its recording, however long that is.
    const replay = { mode: "replay", log: { ticks: new Array(meta.max_steps + 20) } };
    assert.equal(over(replay, { i: meta.max_steps }, meta.max_steps), false);
    assert.equal(over(replay, { i: meta.max_steps + 20 }, meta.max_steps), true);
  }
});
