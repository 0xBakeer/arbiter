/* arbiter plays — the shared side panel every game page mounts.

   const panel = mountPanel(el, { meta, questions, strip: true });   // strip:false hides the last-decisions strip
   panel.update({ questions?, answers, decision:{action, extras}, shield:{intervened, reason, action?},
                  timing:{inference_ms, roundtrip_ms, decisions_per_s},
                  counters:{decisions, interventions, tokens},
                  status:{mode:"live"|"replay", connected:boolean, engine:string, policy?, shield?} });
   panel.setMode("live" | "replay");
   panel.setStatus({ ...partial status });
   panel.flash();                       // the cobalt pulse has arrived: a one-frame hairline on the top section

   The panel renders generically from the game's questions(state):
   choice -> bars per option sorted by probability, the played action highlighted;
   noul   -> a 0-1 gauge labelled with the instruction;
   score  -> the segmented scale with the expected value marker;
   then the readout block, the shield counter and the strip of the last eight decisions.

   Also exported: mountDigits(el, { label, digits }) -> { set(n), hot(bool), dim(bool) } for the
   seven-segment score / length / best readouts, and actionGlyph(action) for the arrows. */

const GLYPHS = {
  up: '↑', down: '↓', left: '←', right: '→',
  north: '↑', south: '↓', west: '←', east: '→',
  jump: '▲', flap: '▲', hop: '▲', wait: '·', stay: '·', noop: '·', none: '·',
  forward: '↑', back: '↓', fire: '•', reveal: '■', flag: '⚑', attack: '⚔', rest: '·',
};
let glyphOverride = {};
export function actionGlyph(action) {
  if (action == null) return '·';
  const a = String(action);
  return glyphOverride[a] || GLYPHS[a.toLowerCase()] || a.slice(0, 3);
}

const f2 = x => (Math.round(x * 100) / 100).toFixed(2);
const f1 = x => (Math.round(x * 10) / 10).toFixed(1);
const el = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; };
const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

export function mountPanel(root, opts = {}) {
  const meta = opts.meta || {};
  if (meta.actions && !Array.isArray(meta.actions)) {
    for (const [k, v] of Object.entries(meta.actions)) if (v && typeof v === 'object' && v.glyph) glyphOverride[k] = v.glyph;
  }
  root.innerHTML = '';
  const box = el('div', 'panel');
  root.appendChild(box);
  const sections = new Map();           // question id -> {kind, el, parts}
  let questions = opts.questions || null;
  const status = { mode: 'live', connected: false, engine: '—', policy: 'model', shield: true };
  const history = [];
  const empty = el('p', 'empty', 'Waiting for the first decision');
  box.appendChild(empty);

  /* --- question sections, rebuilt only when the question set changes --- */
  let qsig = '';
  function buildSections(qs) {
    // the option set may change per tick (only legal moves are offered); the wording of the criteria may not rebuild
    const sig = JSON.stringify(Object.keys(qs).map(k => [k, qs[k].type, Array.isArray(qs[k].criteria) ? qs[k].criteria.length : Object.keys(qs[k].criteria || {})]));
    if (sig === qsig) return;
    qsig = sig;
    for (const s of sections.values()) s.el.remove();
    sections.clear();
    let first = true;
    for (const [id, q] of Object.entries(qs)) {
      const sec = el('section', 'sec ' + q.type);
      const h = el('h2');
      h.appendChild(el('span', 'txt', esc(q.instructions || id)));
      h.appendChild(el('span', 'id', esc(id)));
      const conf = el('span', 'conf'); h.appendChild(conf);
      const parts = { conf };
      if (q.type === 'choice') {
        sec.appendChild(h);
        const bars = el('div', 'bars');
        parts.rows = {};
        const keys = Array.isArray(q.criteria) ? q.criteria : Object.keys(q.criteria || {});
        keys.forEach((k, i) => {
          const row = el('div', 'brow');
          row.style.order = i;
          const desc = !Array.isArray(q.criteria) && q.criteria[k] ? ` title="${esc(q.criteria[k])}"` : '';
          row.innerHTML = `<span class="lbl"${desc}><span class="arrow">${esc(actionGlyph(k))}</span><span class="name">${esc(k)}</span></span><span class="track"><span class="fill" style="width:0%"></span></span><span class="val">0.00</span>`;
          parts.rows[k] = row; bars.appendChild(row);
        });
        sec.appendChild(bars);
        parts.note = el('p', 'note'); parts.note.hidden = true; sec.appendChild(parts.note);
        if (first) { sec.classList.add('primary'); first = false; }
      } else if (q.type === 'noul') {
        const g = el('div', 'gauge');
        g.innerHTML = `<span class="lbl"><span class="txt">${esc(q.instructions || id)}</span><span class="id mono" style="color:var(--faint);font-size:11px">${esc(id)}</span></span><span class="track"><i class="mid"></i><span class="fill" style="width:0%"></span></span><span class="val">0.00</span>`;
        parts.fill = g.querySelector('.fill'); parts.val = g.querySelector('.val'); parts.gauge = g;
        sec.appendChild(g);
        h.remove();
      } else if (q.type === 'score') {
        sec.appendChild(h);
        const scale = el('div', 'scale');
        parts.levels = [];
        const names = Array.isArray(q.criteria) ? q.criteria : Object.values(q.criteria || {});
        names.forEach((name, i) => {
          const lv = el('div', 'lvl');
          lv.innerHTML = `<span class="p">0.00</span><span class="mass"><i style="height:0%"></i></span><span class="lname" title="${esc(name)}">${esc(name)}</span>`;
          parts.levels.push(lv); scale.appendChild(lv);
        });
        parts.exp = el('i', 'exp'); scale.appendChild(parts.exp);
        sec.appendChild(scale);
        parts.note = el('p', 'note'); sec.appendChild(parts.note);
      } else continue;
      sections.set(id, { kind: q.type, el: sec, parts, q });
      box.insertBefore(sec, readout);
    }
  }

  /* --- readout --- */
  const readout = el('div', 'readout');
  const R = {};
  for (const [k, label] of [['inf', 'Inference'], ['rt', 'Round trip'], ['dps', 'Decisions'], ['tok', 'Output tokens'], ['net', 'Network'], ['eng', 'Engine']]) {
    const r = el('div', 'r', `<span>${label}</span><b>—</b>`); R[k] = r; readout.appendChild(r);
  }
  box.appendChild(readout);

  /* --- shield --- */
  const shield = el('div', 'shield');
  shield.innerHTML = `<span class="lbl">Shield interventions<small>illegal moves the game vetoed</small></span><span class="count">0000</span>`;
  const shieldCount = shield.querySelector('.count'), shieldSmall = shield.querySelector('small');
  box.appendChild(shield);

  /* --- last decisions --- */
  const strip = el('div', 'strip');
  strip.innerHTML = `<span class="lbl"><span>Last decisions</span><span>newest on the right</span></span><div class="items"></div>`;
  const items = strip.querySelector('.items');
  if (opts.strip !== false) box.appendChild(strip);        // a board that draws its own trail passes strip:false

  function setCount(n) {
    const s = String(Math.max(0, n | 0)).padStart(4, '0');
    const z = s.match(/^0*/)[0].length;
    shieldCount.innerHTML = `<span class="z">${s.slice(0, Math.min(z, 3))}</span>${s.slice(Math.min(z, 3))}`;
  }
  function paintStatus() {
    R.net.querySelector('b').textContent = status.mode === 'replay' ? 'recorded' : status.connected ? 'local' : 'offline';
    R.net.classList.toggle('off', status.mode !== 'replay' && !status.connected);
    R.eng.querySelector('b').textContent = status.engine || '—';
    shield.classList.toggle('off', status.shield === false);
    shieldSmall.textContent = status.shield === false ? 'shield off: illegal moves are played' : 'illegal moves the game vetoed';
    shieldSmall.classList.remove('hit');
  }

  let flashTimer = 0;
  const api = {
    el: box,
    setMode(mode) { status.mode = mode; paintStatus(); },
    setStatus(s) { Object.assign(status, s || {}); paintStatus(); },
    flash() {
      box.classList.remove('flash'); void box.offsetWidth; box.classList.add('flash');
      clearTimeout(flashTimer); flashTimer = setTimeout(() => box.classList.remove('flash'), 600);
    },
    update(u = {}) {
      if (u.status) Object.assign(status, u.status);
      if (u.questions) questions = u.questions;
      if (!questions) return;
      buildSections(questions);
      if (empty.parentNode) empty.remove();
      const answers = u.answers || {};
      const dec = u.decision || {};
      const sh = u.shield || {};
      const played = sh.intervened && sh.action != null ? sh.action : dec.action;
      const modelPick = dec.extras && dec.extras.model_action != null ? dec.extras.model_action : null;

      for (const [id, s] of sections) {
        const a = answers[id];
        const q = questions[id] || s.q;
        const txt = s.el.querySelector('.txt'); if (txt && q.instructions && txt.textContent !== q.instructions) txt.textContent = q.instructions;
        if (!a) continue;
        if (s.kind === 'choice') {
          const ps = a.probabilities || {};
          const order = Object.keys(s.parts.rows).sort((x, y) => (ps[y] || 0) - (ps[x] || 0));
          order.forEach((k, i) => {
            const row = s.parts.rows[k];
            row.style.order = i;
            if (q.criteria && !Array.isArray(q.criteria) && q.criteria[k]) row.querySelector('.lbl').title = q.criteria[k];
            row.querySelector('.fill').style.width = ((ps[k] || 0) * 100).toFixed(1) + '%';
            row.querySelector('.val').textContent = f2(ps[k] || 0);
            const isPlayed = s.el.classList.contains('primary') ? String(k) === String(played) : String(k) === String(a.choice);
            row.classList.toggle('win', isPlayed);
            row.classList.toggle('veto', s.el.classList.contains('primary') && sh.intervened && String(k) === String(dec.action));
            row.classList.toggle('model', s.el.classList.contains('primary') && modelPick != null && String(k) === String(modelPick) && String(k) !== String(played));
          });
          s.parts.conf.innerHTML = `margin <b>${f2(a.confidence || 0)}</b>`;
          if (s.el.classList.contains('primary')) {
            const n = s.parts.note;
            if (sh.intervened) { n.hidden = false; n.className = 'note veto'; n.innerHTML = `Shield vetoed <b>${esc(dec.action)}</b>${sh.reason ? ': ' + esc(sh.reason) : ''}${sh.action != null ? ', played <b>' + esc(sh.action) + '</b>' : ''}`; }
            else if (modelPick != null && String(modelPick) !== String(played)) { n.hidden = false; n.className = 'note'; n.innerHTML = `Heuristic played <b>${esc(played)}</b>, the model would play <b>${esc(modelPick)}</b>`; }
            else n.hidden = true;
          }
        } else if (s.kind === 'noul') {
          const p = typeof a.noul === 'number' ? a.noul : 0;
          s.parts.fill.style.width = (p * 100).toFixed(1) + '%';
          s.parts.val.textContent = f2(p);
          s.parts.gauge.classList.toggle('low', p < 0.5);
        } else if (s.kind === 'score') {
          const ps = a.probabilities || {};
          const n = s.parts.levels.length;
          let top = 0, topP = -1;
          s.parts.levels.forEach((lv, i) => {
            const p = ps[String(i)] || 0;
            lv.querySelector('.p').textContent = f2(p);
            lv.querySelector('.mass i').style.height = (p * 100).toFixed(1) + '%';
            if (p > topP) { topP = p; top = i; }
          });
          s.parts.levels.forEach((lv, i) => lv.classList.toggle('top', i === top));
          const score = typeof a.score === 'number' ? a.score : top;
          s.parts.exp.style.left = (((score + 0.5) / n) * 100).toFixed(2) + '%';
          s.parts.conf.innerHTML = `margin <b>${f2(a.confidence || 0)}</b>`;
          s.parts.note.innerHTML = `expected <b>${f2(score)}</b> of ${n - 1}`;
        }
      }

      const t = u.timing || {};
      R.inf.querySelector('b').innerHTML = t.inference_ms != null ? `${f1(t.inference_ms)}<b class="unit">ms</b>` : '—';
      R.rt.querySelector('b').innerHTML = t.roundtrip_ms != null ? `${f1(t.roundtrip_ms)}<b class="unit">ms</b>` : '—';
      R.dps.querySelector('b').innerHTML = t.decisions_per_s != null ? `${f1(t.decisions_per_s)}<b class="unit">/s</b>` : '—';
      const c = u.counters || {};
      R.tok.querySelector('b').textContent = c.tokens != null ? String(c.tokens) : '0';
      paintStatus();
      setCount(c.interventions || 0);
      shield.classList.toggle('hit', !!sh.intervened);
      if (sh.intervened) { shieldSmall.textContent = sh.reason || 'illegal move vetoed'; shieldSmall.classList.add('hit'); }

      if (dec.action != null || played != null) {
        const primary = [...sections.values()].find(s => s.el.classList.contains('primary'));
        const pa = primary ? answers[[...sections.entries()].find(([, s]) => s === primary)[0]] : null;
        const prob = pa && pa.probabilities ? pa.probabilities[String(played)] : null;
        history.push({ a: played, p: prob, veto: !!sh.intervened });
        if (history.length > 8) history.shift();
        items.innerHTML = history.map((h, i) => `<span class="it${h.veto ? ' veto' : ''}${i === history.length - 1 ? ' new' : ''}" title="${esc(h.a)}"><span class="a">${esc(actionGlyph(h.a))}</span><span class="p">${h.p != null ? f2(h.p) : '—'}</span></span>`).join('');
      }
    },
    reset() { history.length = 0; items.innerHTML = ''; },
  };
  paintStatus();
  return api;
}

/* ---------- seven-segment digits ---------- */
//    a
//  f   b
//    g
//  e   c
//    d
const SEG = {
  a: '2,0 18,0 15,3 5,3', b: '20,1 20,17 17,15 17,4', c: '20,19 20,35 17,32 17,21', d: '2,36 18,36 15,33 5,33',
  e: '0,19 3,21 3,32 0,35', f: '0,1 3,4 3,15 0,17', g: '3,18 5,16.6 15,16.6 17,18 15,19.4 5,19.4',
};
const DIGIT_SEGS = ['abcdef', 'bc', 'abged', 'abgcd', 'fgbc', 'afgcd', 'afgedc', 'abc', 'abcdefg', 'abcdfg'];
export function mountDigits(root, opts = {}) {
  const n = opts.digits || 4;
  root.classList.add('digit-group');
  root.innerHTML = `<div class="row"></div><span class="lbl">${esc(opts.label || '')}</span>`;
  const row = root.querySelector('.row');
  const cells = [];
  for (let i = 0; i < n; i++) {
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('viewBox', '0 0 20 36');
    for (const s of 'abcdefg') {
      const p = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
      p.setAttribute('points', SEG[s]); p.dataset.s = s; svg.appendChild(p);
    }
    row.appendChild(svg); cells.push(svg);
  }
  let cur = null;
  const api = {
    el: root,
    set(v) {
      if (v === cur) return; cur = v;
      const s = v == null ? '' : String(Math.max(0, Math.min(10 ** n - 1, Math.round(v))));
      const padded = s.padStart(n, ' ');
      cells.forEach((svg, i) => {
        const ch = padded[i];
        const on = ch === ' ' ? '' : DIGIT_SEGS[+ch];
        for (const p of svg.children) p.classList.toggle('on', on.includes(p.dataset.s));
      });
    },
    hot(b) { root.classList.toggle('hot', !!b); },
    dim(b) { root.classList.toggle('dim', !!b); },
  };
  api.set(0);
  return api;
}
