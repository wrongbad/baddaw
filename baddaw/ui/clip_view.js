// ClipView: waveform + ruler + cursor/selection/playhead. Playback runs in the kernel;
// this view only sends play/stop requests and animates the playhead locally.
// All logic lives in clip_view_core.js; this file is DOM and canvas glue.
import {
  fmt, localPlaying, pixelPeaks, playheadFrame, playRequest, pointerDown, pointerMove, pointerUp,
  rulerStep, sliceText, snap,
} from "./clip_view_core.js";

const RULER = 16;
const LANE = 56;

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
}

function button(text, title) {
  const b = el("button", "bd-btn", text);
  b.title = title;
  return b;
}

function render({ model, el: root }) {
  root.classList.add("baddaw-clip");

  // ---- dom ----
  const bar = el("div", "bd-bar");
  const playBtn = button("▶", "play from cursor, or the selection (Enter)");
  const stopBtn = button("■", "stop (Space toggles play/stop)");
  const loopBtn = button("⟳", "loop");
  const copyBtn = button("copy", "copy the selection as a slice, e.g. [1.234:2.345]");
  const label = el("span", "bd-label");
  const readout = el("span", "bd-readout");
  const selText = el("span", "bd-sel");
  const err = el("span", "bd-err");
  bar.append(playBtn, stopBtn, loopBtn, label, readout, selText, copyBtn, err);

  const wrap = el("div", "bd-wave");
  wrap.tabIndex = 0;
  wrap.setAttribute("data-lm-suppress-shortcuts", "true"); // keep JupyterLab shortcuts off our keys
  const wave = el("canvas");
  const over = el("canvas", "bd-over");
  wrap.append(wave, over);
  root.append(bar, wrap);

  // ---- state ----
  const sr = model.get("sr");
  const channels = model.get("channels");
  const columns = model.get("columns");
  const dur = model.get("frames") / sr;
  const dv = model.get("peaks");
  const peaks = new Float32Array(dv.buffer.slice(dv.byteOffset, dv.byteOffset + dv.byteLength));
  const height = RULER + LANE * channels;
  const initial = model.get("selection");

  let view = { cursor: 0, selection: initial.length === 2 ? initial : null, drag: null };
  let hover = null;
  let playing = null; // {start, stop, loop, t0}: frames, frames, bool, performance.now() ms
  let raf = 0;

  const width = () => Math.max(wrap.clientWidth, 100);
  const tToX = (t) => (t / dur) * width();
  const tAt = (e) => snap(((e.clientX - over.getBoundingClientRect().left) / width()) * dur, dur);

  function setupCanvas(c) {
    const dpr = window.devicePixelRatio || 1;
    c.width = Math.round(width() * dpr);
    c.height = Math.round(height * dpr);
    c.style.width = width() + "px";
    c.style.height = height + "px";
    const ctx = c.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return ctx;
  }

  // ---- static layer: ruler + waveform ----
  function drawWave() {
    const ctx = setupCanvas(wave);
    const W = width();
    ctx.clearRect(0, 0, W, height);

    const { step, decimals } = rulerStep(dur, W);
    ctx.fillStyle = getComputedStyle(root).color;
    ctx.globalAlpha = 0.6;
    ctx.font = "10px sans-serif";
    for (let i = 0; i * step <= dur + 1e-9; i++) {
      const x = Math.round(tToX(i * step));
      ctx.fillRect(x, RULER - 5, 1, 5);
      ctx.fillText((i * step).toFixed(decimals), x + 2, 10);
    }
    ctx.globalAlpha = 1;

    for (let c = 0; c < channels; c++) {
      const top = RULER + c * LANE;
      const mid = top + LANE / 2;
      ctx.fillStyle = "rgba(128,128,128,0.35)";
      ctx.fillRect(0, mid, W, 1);
      if (c) ctx.fillRect(0, top, W, 1);
      const { lo, hi } = pixelPeaks(peaks, columns, channels, c, W);
      for (let x = 0; x < W; x++) {
        if (Number.isNaN(lo[x])) continue;
        const y0 = mid - (Math.min(hi[x], 1) * (LANE - 2)) / 2;
        const y1 = mid - (Math.max(lo[x], -1) * (LANE - 2)) / 2;
        ctx.fillStyle = hi[x] >= 1 || lo[x] <= -1 ? "#dc2626" : "#6b7280";
        ctx.fillRect(x, y0, 1, Math.max(1, y1 - y0));
      }
    }
  }

  // ---- dynamic layer: selection, hover, cursor, playhead ----
  function drawOverlay() {
    const ctx = setupCanvas(over);
    ctx.clearRect(0, 0, width(), height);
    const s = view.selection;
    if (s) {
      ctx.fillStyle = "rgba(59,130,246,0.22)";
      ctx.fillRect(tToX(s[0]), 0, Math.max(1, tToX(s[1]) - tToX(s[0])), height);
    }
    if (hover != null) {
      ctx.fillStyle = "rgba(128,128,128,0.5)";
      ctx.fillRect(Math.round(tToX(hover)), 0, 1, height);
    }
    ctx.fillStyle = "#2563eb";
    ctx.fillRect(Math.round(tToX(view.cursor)), 0, 1, height);
    const f = playheadFrame(playing, performance.now(), sr);
    if (f == null && playing) {
      playing = null; // one-shot finished
      updateBar();
    }
    if (f != null) {
      ctx.fillStyle = "#ef4444";
      ctx.fillRect(Math.round(tToX(f / sr)) - 1, 0, 2, height);
    }
  }

  function animate() {
    cancelAnimationFrame(raf);
    const tick = () => {
      drawOverlay();
      if (playing) raf = requestAnimationFrame(tick);
    };
    tick();
  }

  function updateBar() {
    const s = view.selection;
    label.textContent = `${model.get("label")}  ${fmt(dur)} s`;
    readout.textContent = `cursor ${fmt(view.cursor)}` + (hover != null ? `  hover ${fmt(hover)}` : "");
    selText.textContent = s ? `${sliceText(s)}  ${fmt(s[1] - s[0])} s` : "";
    copyBtn.style.display = s ? "" : "none";
    loopBtn.classList.toggle("bd-on", model.get("loop"));
    playBtn.classList.toggle("bd-on", !!playing);
  }

  function refresh() {
    updateBar();
    if (!playing) drawOverlay();
  }

  // ---- kernel transport ----
  function play() {
    err.textContent = "";
    model.send(playRequest({ ...view, loop: model.get("loop") }));
  }

  const stop = () => model.send({ cmd: "stop" });

  // playback state is synced from the kernel transport, so late-rendered views catch up
  function syncPlaying() {
    playing = localPlaying(model.get("playing"), performance.now(), Date.now());
    updateBar();
    animate();
  }
  model.on("change:playing", syncPlaying);

  model.on("msg:custom", (msg) => {
    if (msg.event === "error") err.textContent = msg.message;
  });

  // ---- interaction ----
  function apply({ state, sync, scrub }) {
    view = state;
    if (sync) {
      model.set("selection", view.selection ?? []);
      model.save_changes();
    }
    if (scrub) play();
    refresh();
  }

  over.addEventListener("pointerdown", (e) => {
    wrap.focus();
    over.setPointerCapture(e.pointerId);
    apply(pointerDown(view, tAt(e), e.clientX, e.shiftKey));
  });
  over.addEventListener("pointermove", (e) => {
    hover = tAt(e);
    apply(pointerMove(view, hover, e.clientX));
  });
  over.addEventListener("pointerup", () => apply(pointerUp(view, !!playing)));
  over.addEventListener("pointerleave", () => {
    hover = null;
    refresh();
  });

  wrap.addEventListener("keydown", (e) => {
    const keys = {
      " ": () => (playing ? stop() : play()),
      Enter: play,
      Escape: () => apply({ state: { ...view, selection: null }, sync: true, scrub: false }),
    };
    if (!(e.key in keys)) return;
    e.preventDefault();
    e.stopPropagation();
    keys[e.key]();
  });

  playBtn.addEventListener("click", play);
  stopBtn.addEventListener("click", stop);
  loopBtn.addEventListener("click", () => {
    model.set("loop", !model.get("loop"));
    model.save_changes();
  });
  copyBtn.addEventListener("click", () => {
    if (view.selection) navigator.clipboard?.writeText(sliceText(view.selection));
  });

  // selection or loop set from python
  model.on("change:selection", () => {
    const s = model.get("selection");
    view = { ...view, selection: s.length === 2 ? s : null };
    refresh();
  });
  model.on("change:loop", updateBar);

  const ro = new ResizeObserver(() => {
    drawWave();
    drawOverlay();
  });
  ro.observe(wrap);
  drawWave();
  syncPlaying();

  return () => {
    ro.disconnect();
    cancelAnimationFrame(raf);
  };
}

export default { render };
