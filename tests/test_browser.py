"""Render the real ClipView bundle in Playwright's headless Chromium against a stub model.

No Jupyter, no server, no network: the page is set_content() and the module is inlined.
Opt-in: `pip install playwright && playwright install --only-shell chromium`.
Screenshots land in .baddaw/screenshots/ for eyeballing.
"""

import base64
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("anywidget")
sync_api = pytest.importorskip("playwright.sync_api")

import baddaw as daw  # noqa: E402
from baddaw.ui.clip_view import ClipView, bundle_esm  # noqa: E402

pytestmark = pytest.mark.browser
SHOTS = Path(__file__).parents[1] / ".baddaw" / "screenshots"
CSS = (Path(daw.__file__).parent / "ui" / "clip_view.css").read_text()

# stub of the anywidget model API; records what the view sends and syncs
HARNESS = """
const s = window.__state;
s.peaks = new DataView(Uint8Array.from(atob(s.peaks_b64), c => c.charCodeAt(0)).buffer);
const handlers = {};
window.sent = []; window.synced = [];
const model = {
  get: k => s[k],
  set: (k, v) => { s[k] = v; (handlers['change:' + k] || []).forEach(f => f()); },
  save_changes: () => window.synced.push(JSON.parse(JSON.stringify(s.selection))),
  on: (e, f) => (handlers[e] = handlers[e] || []).push(f),
  send: m => window.sent.push(m),
};
window.kernel = (msg) => (handlers['msg:custom'] || []).forEach(f => f(msg));
window.pySet = (k, v) => model.set(k, v);
// what python's on_transport would sync: t0 is kernel wall-clock ms
window.kernelPlays = (start, stop, loop, ago = 0) => model.set('playing', {start, stop, loop, t0: Date.now() - ago});
render({ model, el: document.getElementById('root') });
window.ready = true;
"""


@pytest.fixture(scope="module")
def browser():
    with sync_api.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as e:  # browser binary not installed
            pytest.skip(f"chromium unavailable: {e}")
        yield b
        b.close()


@pytest.fixture
def page(browser, write_wav):
    sr = 48000
    t = np.arange(2 * sr) / sr
    tone = (0.8 * np.sin(2 * np.pi * 5 * t) * np.exp(-t)).astype(np.float32)
    write_wav("tone.wav", np.stack([tone, -tone], axis=1))
    v = ClipView(daw.load("tone.wav"))
    state = {k: getattr(v, k) for k in ("label", "sr", "frames", "channels", "columns", "selection", "loop", "playing")}
    state["peaks_b64"] = base64.b64encode(v.peaks).decode()

    pg = browser.new_page(viewport={"width": 1000, "height": 260})
    pg.set_content(f"<style>{CSS} body{{margin:10px;font-family:sans-serif}}</style><div id=root style='width:900px'></div>")
    pg.evaluate("s => window.__state = s", state)
    pg.add_script_tag(content=bundle_esm() + HARNESS, type="module")
    pg.wait_for_function("window.ready === true")
    yield pg
    pg.close()


def box(page):
    return page.locator(".bd-over").bounding_box()


def x_at(page, seconds, dur=2.0):
    b = box(page)
    return b["x"] + seconds / dur * b["width"]


def red_pixels(page):
    return page.evaluate("""() => {
        const c = document.querySelector('.bd-over');
        const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
        let n = 0;
        for (let i = 0; i < d.length; i += 4) if (d[i] > 200 && d[i+1] < 100 && d[i+3] > 0) n++;
        return n;
    }""")


def shot(page, name):
    SHOTS.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=SHOTS / f"{name}.png")


def test_renders(page):
    assert "'tone'" in page.text_content(".bd-label") and "2.000 s" in page.text_content(".bd-label")
    assert page.text_content(".bd-readout") == "cursor 0.000"
    assert box(page)["height"] == 16 + 2 * 56
    # waveform actually drawn: non-transparent pixels on the static canvas
    drawn = page.evaluate("""() => {
        const c = document.querySelector('.bd-wave canvas');
        return c.getContext('2d').getImageData(0, 0, c.width, c.height).data.filter((v, i) => i % 4 === 3 && v).length;
    }""")
    assert drawn > 1000
    shot(page, "renders")


def test_drag_selects_and_syncs(page):
    y = box(page)["y"] + 60
    page.mouse.move(x_at(page, 1.5), y)
    page.mouse.down()
    page.mouse.move(x_at(page, 0.5), y, steps=5)
    page.mouse.up()
    sel = page.evaluate("window.synced.at(-1)")
    assert sel == pytest.approx([0.5, 1.5], abs=0.003)
    assert all(round(t * 1000) == t * 1000 for t in sel)  # snapped to ms
    text = page.text_content(".bd-sel")
    assert text.startswith(f"[{sel[0]:.3f}:{sel[1]:.3f}]")
    assert page.text_content(".bd-readout").startswith(f"cursor {sel[0]:.3f}")
    shot(page, "selection")


def test_play_selection_and_playhead(page):
    page.evaluate("pySet('selection', [0.5, 1.5])")
    page.click("text=▶")
    assert page.evaluate("window.sent.at(-1)") == {"cmd": "play", "start": 0.5, "stop": 1.5, "loop": False}
    # the kernel answers with the frame range it is playing
    page.evaluate("kernelPlays(24000, 72000, true)")
    page.wait_for_timeout(150)
    assert red_pixels(page) > 0
    shot(page, "playing")
    page.evaluate("pySet('playing', null)")
    page.wait_for_timeout(50)
    assert red_pixels(page) == 0


def test_click_scrubs_while_playing(page):
    page.evaluate("kernelPlays(0, 96000, false)")
    page.mouse.click(x_at(page, 1.25), box(page)["y"] + 60)
    msg = page.evaluate("window.sent.at(-1)")
    assert msg["cmd"] == "play" and msg["stop"] is None
    assert msg["start"] == pytest.approx(1.25, abs=0.003)


def test_click_when_idle_only_moves_cursor(page):
    page.mouse.click(x_at(page, 0.75), box(page)["y"] + 60)
    assert page.evaluate("window.sent.length") == 0
    assert page.text_content(".bd-readout").startswith("cursor 0.75")


def test_keyboard(page):
    page.evaluate("pySet('selection', [0.25, 0.5])")
    page.focus(".bd-wave")
    page.keyboard.press("Enter")
    assert page.evaluate("window.sent.at(-1)")["start"] == 0.25
    page.evaluate("kernelPlays(12000, 24000, true)")
    page.keyboard.press(" ")
    assert page.evaluate("window.sent.at(-1)") == {"cmd": "stop"}
    page.keyboard.press("Escape")
    assert page.evaluate("window.synced.at(-1)") == []
    assert page.text_content(".bd-sel") == ""


def test_one_shot_playhead_ends(page):
    page.evaluate("kernelPlays(0, 4800, false)")  # 0.1 s
    page.wait_for_timeout(250)
    assert red_pixels(page) == 0
    assert "bd-on" not in page.get_attribute("text=▶", "class")


def playhead_x(page):
    return page.evaluate("""() => {
        const c = document.querySelector('.bd-over');
        const d = c.getContext('2d').getImageData(0, 60, c.width, 1).data;
        for (let i = 0; i < d.length; i += 4) if (d[i] > 200 && d[i+1] < 100 && d[i+3] > 0) return i / 4 / devicePixelRatio;
        return null;
    }""")


def test_late_view_picks_up_running_playback(page):
    # e.g. daw.play() started 1 s ago, then the view rendered: playhead should be ~halfway
    page.evaluate("kernelPlays(0, 96000, false, 1000)")
    page.wait_for_timeout(30)
    assert playhead_x(page) == pytest.approx(450, abs=15)


def test_loop_toggle_and_error(page):
    page.click("text=⟳")
    assert page.evaluate("window.__state.loop") is True
    assert "bd-on" in page.get_attribute("text=⟳", "class")
    page.evaluate("kernel({event: 'error', message: 'ImportError: no sounddevice'})")
    assert page.text_content(".bd-err") == "ImportError: no sounddevice"


def test_state_is_json_safe(page):
    # everything python syncs must survive the widget protocol's JSON
    assert json.loads(json.dumps(page.evaluate("window.__state.selection"))) == []
