// Pure logic for ClipView: no DOM, no canvas. Unit-tested with `node --test`.
// Times are seconds, positions are frames, pixels are CSS pixels.

export const STEPS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600];

/** Round to 1 ms and clamp to [0, dur], so what you see is what you type. */
export function snap(t, dur) {
  return Math.min(Math.max(Math.round(t * 1000) / 1000, 0), dur);
}

export const fmt = (t) => t.toFixed(3);

/** Clip slice syntax, e.g. "[1.234:2.345]". */
export const sliceText = (sel) => `[${fmt(sel[0])}:${fmt(sel[1])}]`;

/** Smallest ruler step giving at least minPx between ticks. */
export function rulerStep(dur, width, minPx = 70) {
  const step = STEPS.find((s) => (s / dur) * width >= minPx) ?? STEPS[STEPS.length - 1];
  const decimals = Math.max(0, -Math.floor(Math.log10(step) + 1e-9));
  return { step, decimals };
}

/**
 * Reduce (columns, channels, 2) min/max peaks to one (lo, hi) pair per pixel for a channel.
 * Returns {lo, hi} Float32Arrays of length width; NaN where a pixel covers no column.
 */
export function pixelPeaks(peaks, columns, channels, channel, width) {
  const lo = new Float32Array(width).fill(NaN);
  const hi = new Float32Array(width).fill(NaN);
  for (let x = 0; x < width; x++) {
    const c0 = Math.floor((x * columns) / width);
    const c1 = Math.max(c0 + 1, Math.floor(((x + 1) * columns) / width));
    for (let i = c0; i < c1 && i < columns; i++) {
      const o = (i * channels + channel) * 2;
      lo[x] = Number.isNaN(lo[x]) ? peaks[o] : Math.min(lo[x], peaks[o]);
      hi[x] = Number.isNaN(hi[x]) ? peaks[o + 1] : Math.max(hi[x], peaks[o + 1]);
    }
  }
  return { lo, hi };
}

/**
 * Convert the synced `playing` trait (t0 = kernel wall-clock epoch ms) to the local
 * performance.now() timebase. Assumes kernel and browser clocks agree (local kernel).
 */
export function localPlaying(p, perfNow, epochNow) {
  if (!p) return null;
  return { start: p.start, stop: p.stop, loop: p.loop, t0: perfNow - (epochNow - p.t0) };
}

/**
 * Where the kernel playback is now, extrapolated from when it started.
 * playing = {start, stop, loop, t0} (frames, ms). Returns frame, or null once a one-shot ended.
 */
export function playheadFrame(playing, nowMs, sr) {
  if (!playing) return null;
  const len = playing.stop - playing.start;
  const elapsed = ((nowMs - playing.t0) / 1000) * sr;
  if (playing.loop && len > 0) return playing.start + (elapsed % len);
  if (elapsed >= len) return null;
  return playing.start + elapsed;
}

/** The kernel play request for the current view state. */
export function playRequest({ selection, cursor, loop }) {
  const [start, stop] = selection ?? [cursor, null];
  return { cmd: "play", start, stop, loop };
}

// ---- pointer interaction as a pure reducer ----
// state = {cursor, selection: [a, b] | null, drag: {x, t, moved} | null}
// Each handler returns {state, sync, scrub}: sync = selection changed and should be
// sent to the kernel; scrub = playback should jump to the new cursor.

export const DRAG_PX = 3;

const ordered = (a, b) => (a === b ? null : [Math.min(a, b), Math.max(a, b)]);

export function pointerDown(state, t, x, shift) {
  if (shift) {
    const selection = ordered(state.cursor, t);
    return { state: { ...state, selection, drag: null }, sync: true, scrub: false };
  }
  return { state: { ...state, drag: { x, t, moved: false } }, sync: false, scrub: false };
}

export function pointerMove(state, t, x) {
  const d = state.drag;
  if (!d || (!d.moved && Math.abs(x - d.x) <= DRAG_PX)) return { state, sync: false, scrub: false };
  const selection = ordered(d.t, t);
  return { state: { ...state, selection, drag: { ...d, moved: true } }, sync: false, scrub: false };
}

export function pointerUp(state, playing) {
  const d = state.drag;
  if (!d) return { state, sync: false, scrub: false };
  if (d.moved) {
    const cursor = state.selection ? state.selection[0] : d.t;
    return { state: { ...state, cursor, drag: null }, sync: true, scrub: false };
  }
  // plain click: move the cursor, clear the selection, and scrub if playing
  return { state: { cursor: d.t, selection: null, drag: null }, sync: true, scrub: playing };
}
