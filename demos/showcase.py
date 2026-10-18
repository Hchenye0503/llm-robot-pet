"""Showcase video: English, few words, one ability per segment.

    title  An LLM-Driven Robot Pet (Microduck) (MuJoCo simulation)
    (1)    Follows instructions                         owner speaks -> LLM -> validator -> scheduler -> policies
    (2)    Different situations, different reactions    same owner, different words -> different ethogram behaviors
    (3)    Finds its own way                            A* on the floor map around furniture; minimap bottom left
    (4)    Recognizes the owner and what they're doing  camera only: DINO -> VLM -> events -> LLM (came home; holding a toy)
    (5)    Notices how the owner feels                  owner curled up for 30 s -> emotion cue -> goes to keep them company

Rules: every sentence is really sent to the LLM and whatever it picks is recorded; if the camera misses, that is
recorded too (no picking good takes). Everything plays at real speed except the 30 s wait in (5), which is fast-forwarded
and labeled. One segment per process (a second VisionWorld renderer in the same process records black frames).

Frame: 1280x720; main view = observer camera (or the owner's view); top right = the robot's eye (detector box + label);
top left = segment title; bottom = caption bar.

    python -m demos.showcase still                  # one frame per segment to check camera placement (no cost)
    python -m demos.showcase seg1                   # seg1 seg2 seg3 "seg4 a" seg5 (needs OPENAI_API_KEY; "seg4 b" is not in the video)
    python -m demos.showcase assemble               # join the segments with a title card
Output: out/showcase_seg<N>.mp4 (+ .json with the playback speed of each frame) -> out/showcase.mp4
"""

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pet import config
from pet.body import sim as S
from pet.perception import geometry as G
from pet.perception import vision_world as V
from pet.planner.schema import Behavior, Gesture, SitStand, Walk
from pet.world import stage as T

OUT = Path(os.environ.get("PET_OUT_DIR", config.ROOT / "out"))
AUTHOR = "Chen-Ye Hong"
W, H = 1280, 720
PIP_W, PIP_H = 384, 216


def _font(size, style=""):
    """Segoe UI on Windows, DejaVu Sans elsewhere, Pillow's default as a last resort."""
    names = {"": ["segoeui.ttf", "DejaVuSans.ttf", "Arial.ttf"], "b": ["segoeuib.ttf", "DejaVuSans-Bold.ttf", "Arial Bold.ttf"],
             "i": ["segoeuii.ttf", "DejaVuSans-Oblique.ttf", "Arial Italic.ttf"]}[style]
    for n in names:
        try:
            return ImageFont.truetype(n, size)
        except OSError:
            pass
    return ImageFont.load_default(size)


FONT_T, FONT_S, FONT_SI = _font(30, "b"), _font(32), _font(34, "i")
FONT_TAG, FONT_BIG, FONT_NAME = _font(22, "b"), _font(60, "b"), _font(36)

TITLES = {1: "Follows instructions", 2: "Different situations, different reactions", 3: "Finds its own way",
          4: "Recognizes the owner and what they're doing", 5: "Notices how the owner feels"}
TRIM_S = 1.0                         # drop the first 1 s of each segment (before settling and mounting the camera)
WAIT_SPEED = 8                       # fast-forward only while waiting (labeled 8x)

CAP = {"seg": 1, "say": "", "do": "", "label": "", "map": None, "view": 0, "speed": 1}
REF = {"speeds": []}


# ═══════════════════════ frame ═══════════════════════

def _box_on_pip(box):
    """VisionWorld box (640x480, fovy 90) -> picture-in-picture coordinates. The main robot view is 1280x720 with the
    same fovy -> focal length x1.5."""
    k = H / G.H_PX
    x1, y1, x2, y2 = ((box[0] - G.W / 2) * k + W / 2, box[1] * k, (box[2] - G.W / 2) * k + W / 2, box[3] * k)
    s = PIP_W / W
    return [x1 * s, y1 * s, x2 * s, y2 * s]


def _bar(d, xy, alpha=150):
    d.rounded_rectangle(xy, radius=10, fill=(0, 0, 0, alpha))


def overlay(img, sim):
    """img = [main camera | robot's eye | owner's view (optional)] side by side -> one 1280x720 frame."""
    REF["speeds"].append(CAP["speed"])
    panels = [img[:, k * W:(k + 1) * W] for k in range(img.shape[1] // W)]
    main = Image.fromarray(panels[2] if CAP["view"] == 2 and len(panels) > 2 else panels[0]).convert("RGBA")
    eye = Image.fromarray(panels[1]).resize((PIP_W, PIP_H))
    layer = Image.new("RGBA", main.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    t = f"({CAP['seg']})  {TITLES[CAP['seg']]}"
    tw = d.textlength(t, font=FONT_T)
    _bar(d, [20, 20, 20 + tw + 32, 68])
    d.text((36, 25), t, font=FONT_T, fill=(255, 255, 255))
    px, py = W - PIP_W - 20, 20
    d.rounded_rectangle([px - 3, py - 3, px + PIP_W + 3, py + PIP_H + 3], radius=8, fill=(255, 255, 255, 230))
    if CAP["view"] == 2:
        d.text((24, 80), "Owner's view", font=FONT_TAG, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
    if CAP["map"] is not None:
        mimg = CAP["map"](sim)
        mx, my = 20, H - mimg.height - 110
        d.rounded_rectangle([mx - 3, my - 3, mx + mimg.width + 3, my + mimg.height + 3], radius=8, fill=(255, 255, 255, 230))
    if CAP["say"] or CAP["do"]:
        _bar(d, [20, H - 96, W - 20, H - 20], alpha=170)
        if CAP["say"]:
            d.text((44, H - 88), CAP["say"], font=FONT_SI, fill=(255, 255, 255))
        if CAP["do"]:
            dw = d.textlength(CAP["do"], font=FONT_S)
            d.text((W - 44 - dw, H - 86), CAP["do"], font=FONT_S, fill=(255, 210, 90))
    out = Image.alpha_composite(main, layer)
    out.paste(eye, (px, py))
    if CAP["map"] is not None:
        out.paste(mimg, (mx, my))
    d2 = ImageDraw.Draw(out)
    d2.text((px + 8, py + 4), "Robot view", font=FONT_TAG, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
    if CAP["speed"] != 1:
        d2.text((px + PIP_W - 52, py + PIP_H + 10), f"{CAP['speed']}×", font=FONT_T,
                fill=(255, 255, 255), stroke_width=3, stroke_fill=(0, 0, 0))
    vw = REF.get("vw")
    if vw is not None and vw.last.get("box") and vw.last.get("status") == "ok":
        b = _box_on_pip(vw.last["box"])
        b = [b[0] + px, b[1] + py, b[2] + px, b[3] + py]
        d2.rectangle(b, outline=(80, 230, 80), width=3)
        if CAP["label"]:
            # label under the picture-in-picture, wrapped at commas when wider than it
            lines, cur = [], ""
            for w in CAP["label"].split(", "):
                if cur and d2.textlength(cur + ", " + w, font=FONT_TAG) > PIP_W:
                    lines.append(cur + ",")
                    cur = "  " + w
                else:
                    cur = f"{cur}, {w}" if cur else w
            for i, ln in enumerate(lines + [cur]):
                d2.text((px, py + PIP_H + 10 + 26 * i), ln, font=FONT_TAG, fill=(80, 230, 80), stroke_width=3, stroke_fill=(0, 0, 0))
    return np.asarray(out.convert("RGB"))


def does(steps) -> str:
    """Plan -> plain words for the caption (what it did, no numbers)."""
    words = []
    for s in steps:
        if isinstance(s, Walk):
            if s.forward_mps > 0:
                words.append("walk forward")
            if s.turn_left_deg_s:
                words.append("turn " + ("left" if s.turn_left_deg_s > 0 else "right"))
            if s.forward_mps == 0 and not s.turn_left_deg_s and s.head and s.head.yaw_left_deg:
                words.append("look " + ("left" if s.head.yaw_left_deg > 0 else "right"))
        elif isinstance(s, SitStand):
            words.append("sit down" if s.posture == "sit" else "stand up")
        elif isinstance(s, (Behavior, Gesture)):
            words.append(s.name.replace("_", " "))
    out = []
    for w in words:                                  # "turn left, turn left" -> once
        if not out or out[-1] != w:
            out.append(w)
    return "→ " + ", ".join(out) if out else ""


def label(obs) -> str:
    """The VLM's owner state -> the label under the robot view, e.g. "person (sitting, holding toy)"."""
    if not obs.get("present"):
        return "person"
    bits = [obs["posture"]] if obs["posture"] != "unknown" else []
    if obs["arms"] not in ("resting", "unknown"):
        bits.append(obs["arms"].replace("_", " "))
    if obs.get("head") == "down":
        bits.append("head down")
    return f"person ({', '.join(bits)})" if bits else "person"


# ═══════════════════════ shared ═══════════════════════

def make_sim(seg, scene, spec_edit, start, cams, video=True, tag=""):
    CAP["seg"] = seg
    REF["out"] = OUT / f"showcase_seg{seg}{tag}.mp4"
    OUT.mkdir(parents=True, exist_ok=True)
    sim = S.DuckSim(scene, str(REF["out"]) if video else None, size=(W, H), cam=[cams[0], "head_camera", *cams[1:]],
                    overlay=overlay, spec_edit=spec_edit, start=start)
    for _ in range(50):
        sim.step("walk", S.zeros())
    vw = V.VisionWorld(sim)
    REF["vw"] = vw
    return sim, vw


def say_and_do(sim, vw, line, view=0):
    """The owner says one line -> LLM -> validator -> scheduler -> policies; the caption shows the line and what was done.
    view = 2: this line is shown from the owner's eyes."""
    from pet.body.scheduler import hold, run_steps, sense_posture
    from pet.planner.llm import plan_with_retry
    p = sense_posture(sim)
    CAP["say"], CAP["do"], CAP["view"] = f"“{line}”", "", view
    idle = ("walk", S.zeros()) if p != "sitting" else ("sitstand", [1.0] + [0.0] * 12)
    hold(sim, idle, 0.8)
    res = plan_with_retry(line, p if p in ("standing", "sitting") else "fallen")
    CAP["do"] = does(res.steps)
    log = list(res.log)
    r = run_steps(sim, res.steps, log, world=vw)
    hold(sim, r["idle"], 1.0)
    CAP["view"] = 0
    print(f"  {line!r:40} -> {CAP['do']}   log={log[-3:]}")
    return r


def finish(sim, vw, falls):
    sim.close()
    vw.r.close()
    REF["out"].with_suffix(".json").write_text(json.dumps(REF["speeds"]))
    print(f"{REF['out'].stem}: {len(falls)} falls; {sim.t:.0f} s -> {REF['out']}")


def watch_falls(sim):
    from pet.body.scheduler import is_fallen
    falls, prev, orig = [], [False], sim.step

    def step(*a):
        orig(*a)
        f = is_fallen(sim)
        if f and not prev[0]:
            falls.append(round(sim.t, 1))
        prev[0] = f
    sim.step = step
    return falls


def eyes(owner_xy, duck_xy, z):
    """Owner's view: from the owner's head toward the duck (moved 12 cm toward the duck so the head itself is not in shot)."""
    dx, dy = duck_xy[0] - owner_xy[0], duck_xy[1] - owner_xy[1]
    n = math.hypot(dx, dy)
    return {"pos": [owner_xy[0] + 0.12 * dx / n, owner_xy[1] + 0.12 * dy / n, z], "lookat": "duck"}


def face(src, dst):
    """Heading (deg) of src looking at dst."""
    return round(math.degrees(math.atan2(dst[1] - src[1], dst[0] - src[0])), 1)


# ═══════════════════════ segments ═══════════════════════
# cameras: {"pos", "lookat": "duck"} = fixed position aimed at the duck; (distance, elevation, azimuth) = tracking

SHOULDER = {"pos": [-3.85, -2.6, 1.75], "lookat": "duck"}       # over the owner's shoulder
SEG1 = dict(start=(-1.3, -1.6, 200.0), cams=[(1.3, -25, 60)])
SEG2 = dict(start=(T.START[0], T.START[1], T.START[2] + 100), cams=[SHOULDER, T.CAM_OWNER])
SEG3 = dict(start=(-3.3, -0.45, -60.0), goal=(0.05, 1.6), cams=[(2.8, -62, 120)])
SEG5 = dict(duck=(-3.565, -1.9, 0.0), owner=(-1.4, -1.9), cams=[{"pos": [-2.5, -0.05, 1.8], "lookat": [-2.45, -1.9, 0.3]}])
# segment 4: the owner comes in at the entrance; or holds up a toy facing the duck
ENTRY_IN, ENTRY_STAND = (0.05, 2.75), (0.05, 2.2)
HOME_DUCK = (-0.67, -0.35)                                      # next to the north end of the stairwell, looking at the entrance
TOY_DUCK, TOY_OWNER = (-1.1, -1.9), (-3.0, -2.2)
SEG4 = {
    "a": dict(duck=(*HOME_DUCK, face(HOME_DUCK, ENTRY_STAND)), owner=dict(pose="stand", yaw=-90), home=False,
              act=lambda o: o.do("appear", at=ENTRY_IN, yaw=-100, pose="stand").do("walk", path=[(0.05, 1.8)], then="stand"),
              cams=[{"pos": [-0.2, -2.6, 1.6], "lookat": "duck"}]),
    "b": dict(duck=(*TOY_DUCK, face(TOY_DUCK, TOY_OWNER)), owner=dict(pose="stand", at=TOY_OWNER, yaw=face(TOY_OWNER, TOY_DUCK)),
              home=True, act=lambda o: o.do("pose", name="hold_toy", blend=0.8),
              cams=[{"pos": [-1.1, 0.35, 1.7], "lookat": "duck"}, eyes(TOY_OWNER, TOY_DUCK, 1.47)]),
}


def seg1(video=True):
    from pet.body.scheduler import hold
    sim, vw = make_sim(1, T.SCENE, T.stage, SEG1["start"], SEG1["cams"], video)
    if not video:
        return sim, vw
    falls = watch_falls(sim)
    CAP["label"] = "person"
    for line in ["Walk forward a little.", "Look to your left.", "Back up!", "Sit down."]:
        say_and_do(sim, vw, line)
    CAP["say"] = CAP["do"] = ""
    hold(sim, ("sitstand", [1.0] + [0.0] * 12), 1.0)
    finish(sim, vw, falls)


def seg2(video=True):
    from pet.body.scheduler import hold, run_steps
    sim, vw = make_sim(2, T.SCENE, T.stage, SEG2["start"], SEG2["cams"], video)
    if not video:
        return sim, vw
    falls = watch_falls(sim)
    run_steps(sim, [SitStand(action="sitstand", posture="sit")], [])
    hold(sim, ("sitstand", [1.0] + [0.0] * 12), 1.0)
    CAP["label"] = "person"
    for line in ["I'm home!", "Good boy!", "Hey! You knocked over my cup!"]:
        say_and_do(sim, vw, line)
    say_and_do(sim, vw, "Wanna play?", view=2)                      # the play invitation from the owner's eyes
    CAP["say"] = CAP["do"] = ""
    finish(sim, vw, falls)


def minimap_fn(sim, goal, path):
    """Minimap: furniture (gray), A* waypoints (orange), the path walked (blue), the duck (red), the goal (green)."""
    from pet.nav import grid as M
    from pet.nav.route import occ_for
    occ = occ_for(sim.model)
    nx, ny = occ.shape
    px = 4                                              # one 5 cm cell -> 4 px
    a = np.full((ny * px, nx * px, 3), 250, np.uint8)
    for i, j in np.argwhere(occ):
        a[(ny - 1 - j) * px:(ny - j) * px, i * px:(i + 1) * px] = (120, 120, 120)
    base = Image.fromarray(a)
    trail = []

    def to_px(x, y):
        return (x - M.X0) / M.RES * px, (ny - (y - M.Y0) / M.RES) * px

    def draw(sim):
        x, y, _ = sim.xy_yaw()
        trail.append(to_px(x, y))
        im = base.copy()
        d = ImageDraw.Draw(im)
        if len(path) > 1:
            d.line([to_px(*p) for p in path], fill=(255, 150, 30), width=4)
        if len(trail) > 1:
            d.line(trail, fill=(40, 110, 230), width=4)
        gx, gy = to_px(*goal)
        d.ellipse([gx - 8, gy - 8, gx + 8, gy + 8], fill=(40, 190, 70))
        cx, cy = trail[-1]
        d.ellipse([cx - 8, cy - 8, cx + 8, cy + 8], fill=(230, 50, 50))
        return im.resize((im.width * 3 // 4, im.height * 3 // 4))
    return draw


def stage_dim_tiles(spec):
    """The upstream kitchen tiles (floor_tile, 0.92, almost white) blow out under the lights -> darkened for segment 3 only."""
    T.stage(spec)
    for t in spec.textures:
        if t.name == "floor_tile":
            t.rgb1, t.rgb2 = [0.66, 0.70, 0.74], [0.56, 0.60, 0.64]


def seg3(video=True):
    from pet.body.scheduler import hold
    from pet.nav import route as N
    sim, vw = make_sim(3, T.SCENE, stage_dim_tiles, SEG3["start"], SEG3["cams"], video)
    if not video:
        return sim, vw
    falls = watch_falls(sim)
    goal = SEG3["goal"]
    x, y, _ = sim.xy_yaw()
    pts = N.plan(N.grid_for(sim.model), (x, y), goal) or []
    CAP["map"] = minimap_fn(sim, goal, [(x, y)] + pts)
    CAP["label"] = ""
    CAP["say"], CAP["do"] = "Goal: the front door", "→ plan a path on the floor map (A*)"
    hold(sim, ("walk", S.zeros()), 1.5)
    log = []
    r = N.goto_planned(sim, goal, stop_dist=0.3, log=log)
    CAP["do"] = "→ arrived" if r.get("reached") else "→ stopped"
    hold(sim, ("walk", S.zeros()), 1.5)
    print(f"  A* waypoints {pts}; {log}; result {r}")
    finish(sim, vw, falls)


def _owner_scene(seg, tag, duck, owner_kw, cams, video):
    from pet.world import owner as O
    CAP["seg"] = seg
    REF["out"] = OUT / f"showcase_seg{seg}{tag}.mp4"
    OUT.mkdir(parents=True, exist_ok=True)
    sim = S.DuckSim(O.T.SCENE, str(REF["out"]) if video else None, size=(W, H), cam=[cams[0], "head_camera", *cams[1:]],
                    overlay=overlay, spec_edit=O.stage_live, start=duck)
    owner = O.Owner(sim, **owner_kw)
    for _ in range(50):
        sim.step("walk", S.zeros())
    x, y, _ = sim.xy_yaw()
    owner.look = (x, y)
    vw = V.VisionWorld(sim)
    REF["vw"] = vw
    return sim, vw, owner


def seg4(video=True, shot=None):
    """Two shots, the owner says nothing: a) the owner comes home (camera -> came_home -> LLM -> greet);
    b) the owner holds up a toy (toy_out -> look up to check they are looking -> LLM)."""
    from pet.body.scheduler import hold, run_steps, sense_posture
    from pet.mind.inner import Inner
    from pet.perception.async_loop import AsyncPerception
    from pet.perception.loop import WATCH_DT, compose, confirm_facing, needs_look
    from pet.planner.llm import plan_with_retry
    from pet.planner.prompts import SYSTEM_PET
    from pet.world import owner as O
    shot = shot or (sys.argv[2] if len(sys.argv) > 2 else "a")
    sc = SEG4[shot]
    owner_kw = dict(sc["owner"])
    owner_kw.setdefault("at", O.AWAY)
    sim, vw, owner = _owner_scene(4, shot, sc["duck"], owner_kw, sc["cams"], video)
    if not video:
        return sim, vw
    falls = watch_falls(sim)
    per = AsyncPerception(vw)
    per.ev.known_home = sc["home"]
    inner = Inner()                     # as in the full system: the sentence carries [self], prompt SYSTEM_PET
    inner.d.check()
    idle = ("walk", S.zeros())
    owner.do("wait", secs=2.0)
    sc["act"](owner)
    t0, fired = sim.t, None
    CAP["say"] = CAP["do"] = ""
    while sim.t - t0 < 30 and not fired:            # the system keeps watching; 30 s covers a missed confirmation + patrol
        obs, ev = per.watch_step()
        CAP["label"] = label(per.last)
        if ev:
            fired = ev
            CAP["say"] = {"came_home": "Owner came home", "toy_out": "Owner is holding a toy"}.get(ev[0], ev[0].replace("_", " "))
            if needs_look(ev, obs):
                CAP["do"] = "→ looks up: is the owner looking at me?"
            obs = confirm_facing(vw, ev, obs, per.log)             # interactive event and facing unknown -> look up first
            sentence = compose(ev, obs, inner.text(obs))
            p = sense_posture(sim)
            res = plan_with_retry(sentence, p if p in ("standing", "sitting") else "fallen", system=SYSTEM_PET)
            CAP["do"] = does(res.steps)
            log = list(res.log)
            idle = run_steps(sim, res.steps, log, world=vw)["idle"]   # keep the posture the plan ended in (sitting stays sitting)
            print(f"  events {ev}: {sentence} -> {CAP['do']}  {log[-3:]}")
            break
        hold(sim, idle, WATCH_DT)
    hold(sim, idle, 1.5)
    CAP["view"] = 0
    per.close()
    print("\n".join(per.log[-12:]))
    if not fired:
        print("  no event fired")
    finish(sim, vw, falls)


def seg5(video=True):
    """Emotion cue: the owner sits -> hugs their knees -> held for 30 s before it counts -> Inner tells the LLM "may be
    sad" -> the duck goes to keep them company. The 30 s wait plays at 8x (labeled); everything else at 1x."""
    from pet.body.scheduler import hold, run_steps, sense_posture
    from pet.mind.inner import Inner
    from pet.perception.async_loop import AsyncPerception
    from pet.perception.loop import WATCH_DT, compose
    from pet.planner.llm import plan_with_retry
    from pet.planner.prompts import SYSTEM_PET
    sim, vw, owner = _owner_scene(5, "", SEG5["duck"], dict(pose="sit", at=SEG5["owner"], yaw=180), SEG5["cams"], video)
    if not video:
        return sim, vw
    falls = watch_falls(sim)
    per, inner = AsyncPerception(vw), Inner()
    per.ev.known_home = True
    idle = ("walk", S.zeros())
    owner.do("wait", secs=3.0).do("pose", name="hug_knees", blend=0.8)
    t0 = t_prev = sim.t
    fired = None
    CAP["say"] = CAP["do"] = ""
    while sim.t - t0 < 50 and not fired:
        if sim.t - t0 > 6.0:
            CAP["speed"] = WAIT_SPEED
            CAP["say"] = "Owner stays curled up..."
        obs, ev = per.watch_step()
        CAP["label"] = label(per.last)
        reasons, extra = inner.update(sim.t, obs, ev, sim.t - t_prev)
        t_prev = sim.t
        if reasons:
            fired = reasons
            CAP["speed"] = 1
            CAP["say"] = f"Owner may be sad  ({sim.t - t0 - 3.0:.0f} s later)"
            sentence = compose(ev, obs, extra)
            p = sense_posture(sim)
            res = plan_with_retry(sentence, p if p in ("standing", "sitting") else "fallen", system=SYSTEM_PET)
            CAP["do"] = does(res.steps)
            log = list(res.log)
            run_steps(sim, res.steps, log, world=vw)
            print(f"  cue {reasons}: {sentence} -> {CAP['do']}  {log[-3:]}")
            break
        hold(sim, idle, WATCH_DT)
    CAP["speed"] = 1
    hold(sim, ("sitstand", [1.0] + [0.0] * 12) if sense_posture(sim) == "sitting" else idle, 2.0)
    per.close()
    print("\n".join(per.log[-10:]))
    if not fired:
        print("  no cue fired")
    finish(sim, vw, falls)


# ═══════════════════════ camera check, assembly ═══════════════════════

def still():
    """One frame at the start of each segment (no cost): is the duck in shot, is a wall in the way?"""
    import mujoco
    OUT.mkdir(parents=True, exist_ok=True)
    shots = {1: (seg1, SEG1["cams"][0]), 2: (seg2, SEG2["cams"][1]), 3: (seg3, SEG3["cams"][0]), 5: (seg5, SEG5["cams"][0])}
    for k, (fn, spec) in shots.items():
        sim, vw = fn(video=False)
        r = mujoco.Renderer(sim.model, H // 2, W // 2)       # no video -> the offscreen buffer is only 640 wide; half size is enough
        cam = mujoco.MjvCamera()
        x, y, _ = sim.xy_yaw()
        if isinstance(spec, dict):
            sim._aim(cam, spec["pos"], [x, y, 0.15] if spec["lookat"] == "duck" else spec["lookat"])
        else:
            cam.lookat[:] = [x, y, 0.15]
            cam.distance, cam.elevation, cam.azimuth = spec
        r.update_scene(sim.data, cam)
        Image.fromarray(r.render()).save(OUT / f"showcase_still{k}.png")
        r.close()
        vw.r.close()
        sim.close()
        print("saved", OUT / f"showcase_still{k}.png")


def _card(lines, frames):
    im = Image.new("RGB", (W, H), (18, 20, 26))
    d = ImageDraw.Draw(im)
    hs = [font.size + 26 for _, font, _ in lines]
    y = (H - sum(hs)) / 2
    for (text, font, color), h in zip(lines, hs):
        w = d.textlength(text, font=font)
        d.text(((W - w) / 2, y), text, font=font, fill=color)
        y += h
    a = np.asarray(im)
    return [a] * frames


def assemble(fps=25):
    """Resample each segment by the speeds recorded with it (.json, one per frame): 1x keeps 1 of 2 frames
    (50 -> 25 fps), 8x keeps 1 of 16."""
    import imageio
    import imageio.v3 as iio
    src_fps = int(round(1 / S.CONTROL_DT))
    out = OUT / "showcase.mp4"
    w = imageio.get_writer(out, fps=fps, quality=8)
    head = [("An LLM-Driven Robot Pet (Microduck)", FONT_BIG, (255, 255, 255)),
            ("(MuJoCo simulation)", FONT_S, (190, 200, 215))]
    if AUTHOR:
        head.append((AUTHOR, FONT_NAME, (255, 210, 90)))
    for f in _card(head, int(3 * fps)):
        w.append_data(f)
    total = 3.0
    # seg4b (holding up a toy) is left out: from ~2 m the camera sees a standing owner only up to the waist, so the toy
    # held at the chest is not in the photo and the VLM honestly answers "unknown" (see README, Limitations)
    for name in ["seg1", "seg2", "seg3", "seg4a", "seg5"]:
        path = OUT / f"showcase_{name}.mp4"
        speeds = json.loads(path.with_suffix(".json").read_text())
        n_in = n_out = 0
        acc = 0.0
        for i, frame in enumerate(iio.imiter(path)):       # frame by frame (reading a whole segment at once used ~8 GB)
            n_in += 1
            if i < TRIM_S * src_fps:
                continue
            acc += fps / (src_fps * (speeds[i] if i < len(speeds) else 1))
            if acc >= 1.0:
                acc -= 1.0
                w.append_data(frame)
                n_out += 1
        total += n_out / fps
        print(f"  {name}: {n_in / src_fps:.0f} s recorded -> {n_out / fps:.1f} s")
    w.close()
    print(f"-> {out} ({total:.0f} s)")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "still"
    {"still": still, "seg1": seg1, "seg2": seg2, "seg3": seg3, "seg4": seg4, "seg5": seg5, "assemble": assemble}[cmd]()
