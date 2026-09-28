import assert from "node:assert/strict";
import { test } from "node:test";

import {
  localPlaying, pixelPeaks, playheadFrame, playRequest, pointerDown, pointerMove, pointerUp,
  rulerStep, sliceText, snap,
} from "../../baddaw/ui/clip_view_core.js";

const fresh = { cursor: 0, selection: null, drag: null };

test("snap rounds to ms and clamps", () => {
  assert.equal(snap(1.23449, 10), 1.234);
  assert.equal(snap(1.2345, 10), 1.235);
  assert.equal(snap(-0.5, 10), 0);
  assert.equal(snap(12, 10), 10);
});

test("sliceText is pasteable clip syntax", () => {
  assert.equal(sliceText([1.5, 2]), "[1.500:2.000]");
});

test("rulerStep keeps ticks apart and picks decimals", () => {
  assert.deepEqual(rulerStep(3, 800), { step: 0.5, decimals: 1 });
  assert.deepEqual(rulerStep(0.05, 800), { step: 0.005, decimals: 3 });
  assert.deepEqual(rulerStep(600, 800), { step: 60, decimals: 0 });
});

test("pixelPeaks reduces columns to pixels per channel", () => {
  // 4 columns, 2 channels: channel 0 min/max = -i/max i, channel 1 = constant 0.5
  const cols = 4;
  const p = new Float32Array(cols * 2 * 2);
  for (let i = 0; i < cols; i++) {
    p.set([-i / 10, i / 10], (i * 2 + 0) * 2);
    p.set([0.5, 0.5], (i * 2 + 1) * 2);
  }
  const a = pixelPeaks(p, cols, 2, 0, 2);
  assert.deepEqual([...a.lo].map((v) => +v.toFixed(2)), [-0.1, -0.3]);
  assert.deepEqual([...a.hi].map((v) => +v.toFixed(2)), [0.1, 0.3]);
  const b = pixelPeaks(p, cols, 2, 1, 8); // more pixels than columns: every pixel still covered
  assert.equal(b.lo.length, 8);
  assert.ok(b.hi.every((v) => v === 0.5));
});

test("playhead extrapolates, loops, and ends", () => {
  const sr = 1000;
  const once = { start: 100, stop: 300, loop: false, t0: 0 };
  assert.equal(playheadFrame(once, 50, sr), 150);
  assert.equal(playheadFrame(once, 250, sr), null);
  const loop = { ...once, loop: true };
  assert.equal(playheadFrame(loop, 250, sr), 150);
  assert.equal(playheadFrame(null, 0, sr), null);
});

test("localPlaying maps kernel wall clock to the local timebase", () => {
  assert.equal(localPlaying(null, 0, 0), null);
  // started 250 ms ago by the wall clock -> t0 is 250 ms before perfNow
  const p = localPlaying({ start: 0, stop: 1000, loop: false, t0: 10_000 }, 500, 10_250);
  assert.deepEqual(p, { start: 0, stop: 1000, loop: false, t0: 250 });
  assert.equal(playheadFrame(p, 500, 1000), 250); // 250 ms in at 1 kHz
});

test("play request: selection wins, else cursor to end", () => {
  assert.deepEqual(playRequest({ selection: [1, 2], cursor: 0.5, loop: true }), { cmd: "play", start: 1, stop: 2, loop: true });
  assert.deepEqual(playRequest({ selection: null, cursor: 0.5, loop: false }), { cmd: "play", start: 0.5, stop: null, loop: false });
});

test("click moves cursor, clears selection, scrubs only while playing", () => {
  let s = { cursor: 0, selection: [1, 2], drag: null };
  s = pointerDown(s, 3, 300, false).state;
  const idle = pointerUp(s, false);
  assert.deepEqual(idle.state, { cursor: 3, selection: null, drag: null });
  assert.equal(idle.sync, true);
  assert.equal(idle.scrub, false);
  assert.equal(pointerUp(s, true).scrub, true);
});

test("jitter under the drag threshold is still a click", () => {
  let s = pointerDown(fresh, 1, 100, false).state;
  s = pointerMove(s, 1.01, 102).state;
  assert.equal(s.selection, null);
  assert.equal(pointerUp(s, false).state.cursor, 1);
});

test("drag selects in either direction and puts the cursor at the start", () => {
  let s = pointerDown(fresh, 2, 200, false).state;
  s = pointerMove(s, 1, 100).state;
  assert.deepEqual(s.selection, [1, 2]);
  const r = pointerUp(s, false);
  assert.deepEqual(r.state.selection, [1, 2]);
  assert.equal(r.state.cursor, 1);
  assert.equal(r.sync, true);
});

test("dragging back onto the start clears the selection", () => {
  let s = pointerDown(fresh, 2, 200, false).state;
  s = pointerMove(s, 3, 300).state;
  s = pointerMove(s, 2, 201).state; // moved stays true once past the threshold
  assert.equal(s.selection, null);
});

test("shift-click selects from the cursor", () => {
  const r = pointerDown({ ...fresh, cursor: 4 }, 1.5, 0, true);
  assert.deepEqual(r.state.selection, [1.5, 4]);
  assert.equal(r.sync, true);
});
