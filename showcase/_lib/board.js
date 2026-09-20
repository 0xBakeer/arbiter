/* arbiter plays — canvas helpers every board can share.

   drawDecisionHalo(ctx, {x, y}, probs, chosen, vetoed, opts)   the model's probabilities as four wedges around the actor
   drawBoardDepth(ctx, W, H, {cell, n, pad | padX, padY, dpr})  vignette + faint grid + inner wall glow
   createParticles()  -> { burst(x, y, opts), step(dt), draw(ctx) }
   createShake()      -> { kick(strength), offset(dt) -> [dx, dy] }
   countUp(el, to, opts)                                        animated number in a HUD element (with an optional pop)
   ringProgress(circleEl, fraction)                             a thin SVG ring for the episode progress

   Colours are the theme's: cobalt #5583FF, cobalt-text #7DA0FF, orange #FF8C33, danger #F0605C. */

export const COLORS = { cobalt: '#5583FF', cobaltText: '#7DA0FF', orange: '#FF8C33', danger: '#F0605C', ink: '#E9EDF7', surface: '#0E1528', line: '#1E2842', line2: '#2B3758' };

const ANGLES = { up: -Math.PI / 2, right: 0, down: Math.PI / 2, left: Math.PI, north: -Math.PI / 2, east: 0, south: Math.PI / 2, west: Math.PI };
const rgba = (hex, a) => `rgba(${parseInt(hex.slice(1, 3), 16)},${parseInt(hex.slice(3, 5), 16)},${parseInt(hex.slice(5, 7), 16)},${a})`;

/* The decision halo: one annular wedge per option, its reach and opacity equal to the model's probability,
   the chosen one lit cobalt, a vetoed one red. `opts.angles` maps option -> radian for games whose actions
   are not compass directions (hopper: { flap: -PI/2, glide: PI/2 }). `opts.flash` (0..1) pulses the veto. */
export function drawDecisionHalo(ctx, at, probs, chosen, vetoed, opts = {}) {
  if (!probs) return;
  const r0 = opts.inner || 14, reach = opts.reach || 34, spread = opts.spread || 0.62; // spread: half-angle in rad
  const angles = opts.angles || ANGLES;
  const alpha = opts.alpha == null ? 1 : opts.alpha;
  const gap = opts.gap == null ? 0.08 : opts.gap;
  ctx.save();
  ctx.translate(at.x, at.y);
  ctx.lineCap = 'butt';
  for (const [k, p0] of Object.entries(probs)) {
    const a = angles[String(k).toLowerCase()];
    if (a == null) continue;
    const p = Math.max(0, Math.min(1, p0 || 0));
    const isChosen = String(k) === String(chosen), isVeto = vetoed != null && String(k) === String(vetoed);
    const r1 = r0 + reach * (0.18 + 0.82 * p);
    const half = spread * (0.55 + 0.45 * p);
    // track: the faint full wedge, so all options read as a compass
    ctx.beginPath(); ctx.arc(0, 0, r0 + reach, a - spread + gap, a + spread - gap); ctx.arc(0, 0, r0, a + spread - gap, a - spread + gap, true); ctx.closePath();
    ctx.fillStyle = rgba(COLORS.ink, 0.035 * alpha); ctx.fill();
    // mass: the probability
    ctx.beginPath(); ctx.arc(0, 0, r1, a - half + gap, a + half - gap); ctx.arc(0, 0, r0, a + half - gap, a - half + gap, true); ctx.closePath();
    if (isVeto) {
      const f = opts.flash == null ? 1 : opts.flash;
      ctx.fillStyle = rgba(COLORS.danger, (0.25 + 0.55 * f) * alpha); ctx.shadowColor = rgba(COLORS.danger, 0.8 * f); ctx.shadowBlur = 14 * f;
    } else if (isChosen) {
      ctx.fillStyle = rgba(COLORS.cobalt, (0.55 + 0.45 * p) * alpha); ctx.shadowColor = rgba(COLORS.cobalt, 0.9); ctx.shadowBlur = 16;
    } else {
      ctx.fillStyle = rgba(COLORS.cobaltText, (0.12 + 0.5 * p) * alpha); ctx.shadowBlur = 0;
    }
    ctx.fill(); ctx.shadowBlur = 0;
    // a tick at the tip of the chosen wedge
    if (isChosen) {
      ctx.beginPath(); ctx.arc(0, 0, r1 + 3, a - 0.12, a + 0.12); ctx.strokeStyle = rgba(COLORS.ink, 0.9 * alpha); ctx.lineWidth = 2; ctx.stroke();
    }
  }
  ctx.restore();
}

/* Depth for the board: a vignette that darkens the edges, a faint grid, and a cobalt inner glow along the walls. */
export function drawBoardDepth(ctx, W, H, opts = {}) {
  const cell = opts.cell || 0, n = opts.n || 0, dpr = opts.dpr || 1;
  const padX = opts.padX != null ? opts.padX : (opts.pad || 0), padY = opts.padY != null ? opts.padY : (opts.pad || 0);
  const g = ctx.createRadialGradient(W / 2, H / 2, Math.min(W, H) * 0.25, W / 2, H / 2, Math.max(W, H) * 0.78);
  g.addColorStop(0, 'rgba(14,21,40,0)'); g.addColorStop(1, `rgba(4,7,16,${opts.vignette == null ? 0.75 : opts.vignette})`);
  ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);
  if (cell && n) {
    ctx.strokeStyle = rgba(COLORS.line2, 0.45); ctx.lineWidth = Math.max(1, dpr * 0.75);
    ctx.beginPath();
    for (let i = 0; i <= n; i++) {
      const x = Math.round(padX + i * cell) + 0.5, y = Math.round(padY + i * cell) + 0.5;
      ctx.moveTo(x, padY); ctx.lineTo(x, padY + n * cell); ctx.moveTo(padX, y); ctx.lineTo(padX + n * cell, y);
    }
    ctx.stroke();
    // wall glow: a blurred cobalt stroke clipped to the inside of the arena
    ctx.save(); ctx.beginPath(); ctx.rect(padX, padY, n * cell, n * cell); ctx.clip();
    ctx.shadowColor = rgba(COLORS.cobalt, opts.glow == null ? 0.45 : opts.glow); ctx.shadowBlur = 22 * dpr;
    ctx.strokeStyle = rgba(COLORS.cobalt, 0.35); ctx.lineWidth = 2 * dpr;
    ctx.strokeRect(padX - dpr, padY - dpr, n * cell + 2 * dpr, n * cell + 2 * dpr);
    ctx.restore();
  }
}

/* A red vignette that flashes on death; strength 0..1. */
export function drawDangerVignette(ctx, W, H, strength) {
  if (strength <= 0) return;
  const g = ctx.createRadialGradient(W / 2, H / 2, Math.min(W, H) * 0.2, W / 2, H / 2, Math.max(W, H) * 0.7);
  g.addColorStop(0, 'rgba(240,96,92,0)'); g.addColorStop(1, `rgba(240,96,92,${0.55 * strength})`);
  ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);
}

export function createParticles() {
  const ps = [];
  return {
    burst(x, y, opts = {}) {
      const n = opts.count || 16, speed = opts.speed || 90, color = opts.color || COLORS.orange, size = opts.size || 3;
      for (let i = 0; i < n; i++) {
        const a = (i / n) * Math.PI * 2 + Math.random() * 0.5, v = speed * (0.5 + Math.random());
        ps.push({ x, y, vx: Math.cos(a) * v, vy: Math.sin(a) * v, life: 1, decay: 1 / (opts.life || 0.55), color, size: size * (0.6 + Math.random() * 0.8) });
      }
    },
    step(dt) {
      for (let i = ps.length - 1; i >= 0; i--) {
        const p = ps[i]; p.x += p.vx * dt; p.y += p.vy * dt; p.vx *= 0.9; p.vy *= 0.9; p.life -= p.decay * dt;
        if (p.life <= 0) ps.splice(i, 1);
      }
    },
    draw(ctx, scale = 1) {
      for (const p of ps) {
        ctx.globalAlpha = Math.max(0, p.life); ctx.fillStyle = p.color;
        ctx.beginPath(); ctx.arc(p.x, p.y, p.size * scale * (0.4 + 0.6 * p.life), 0, Math.PI * 2); ctx.fill();
      }
      ctx.globalAlpha = 1;
    },
    get count() { return ps.length; },
  };
}

export function createShake() {
  let energy = 0;
  return {
    kick(strength = 1) { energy = Math.max(energy, strength); },
    offset(dt) {
      if (energy <= 0.001) { energy = 0; return [0, 0]; }
      const e = energy; energy *= Math.pow(0.02, dt);          // dies in about a third of a second
      return [(Math.random() * 2 - 1) * 9 * e, (Math.random() * 2 - 1) * 9 * e];
    },
    get active() { return energy > 0.001; },
  };
}

/* Animated count-up for HUD numbers. Pops the element (class "pop") when the value grows. */
const counters = new WeakMap();
export function countUp(el, to, opts = {}) {
  const st = counters.get(el) || { value: +el.textContent || 0, raf: 0 };
  counters.set(el, st);
  const from = st.value, dur = opts.duration == null ? 420 : opts.duration;
  if (from === to) return;
  if (to > from && opts.pop !== false) { el.classList.remove('pop'); void el.offsetWidth; el.classList.add('pop'); }
  if (dur === 0 || matchMedia('(prefers-reduced-motion: reduce)').matches) { st.value = to; el.textContent = String(to); return; }
  cancelAnimationFrame(st.raf);
  const t0 = performance.now();
  const tick = now => {
    const k = Math.min(1, (now - t0) / dur), e = 1 - Math.pow(1 - k, 3);
    st.value = k < 1 ? from + (to - from) * e : to;
    el.textContent = String(Math.round(st.value));
    if (k < 1) st.raf = requestAnimationFrame(tick);
  };
  st.raf = requestAnimationFrame(tick);
}

/* A thin progress ring: the circle's stroke-dasharray follows the fraction. */
export function ringProgress(circle, fraction) {
  const r = +circle.getAttribute('r'), c = 2 * Math.PI * r;
  circle.style.strokeDasharray = `${Math.max(0, Math.min(1, fraction)) * c} ${c}`;
}
