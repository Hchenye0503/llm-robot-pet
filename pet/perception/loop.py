"""Perception loop: photo -> is anyone there -> owner state -> which region -> events -> one sentence for the LLM.

While the duck is idle it watches:
    1. photo (head camera, vision_world.py)
    2. Grounding DINO looks for "a person" (0.3 s, free); nobody -> no VLM call (saves money; not seen is not absent)
    3. the VLM fills in OwnerState (owner_state.py, ~1.5 s)
    4. where the owner is: box -> landing point (floor plane cross-checked with known size) -> nav.waypoints.region
    5. events.EventDetector.update -> events
    6. describe() -> "[camera] The owner is holding up a toy (...)" -> the planning LLM

Why the VLM fills in states and the program detects events: one photo cannot tell "just came home" (events.py).
"""

import base64
import io
import math
import time

from PIL import Image
from pydantic import ValidationError

from . import geometry as G
from .events import EventDetector, classify_motion
from .owner_state import CUES, OwnerState, openai_schema
from .vision_world import MIN_SCORE, SIZE_K, held_toy, landing, shoot
from ..nav.waypoints import region
from ..planner.llm import MODEL

OBS_DT = 2.0           # how often to look while idle (s)
SYSTEM_STATE = ("You look at one photo from a small robot dog's head camera (about 24 cm above the floor) in a simulated home "
                "rendered with simple 3D graphics. People are simplified 3D human models made of rounded shapes; count them as people. "
                "Describe ONLY what you can see about the person, following each field's definition exactly. "
                "If a body part is out of the frame or unclear, answer unknown instead of guessing. "
                # seen from a sitting duck looking up, perspective is extreme (things held at the chest look above the head)
                "The camera is low and may be tilted up at the person, so hands and objects held in front of the chest "
                "can look higher than they really are.")
EVENT_TEXT = {"came_home": "The owner just came home", "stood_up": "The owner just stood up",
              "leaving": "The owner is walking away", "toy_out": "The owner is holding up a toy"}
ABSENT = {"present": False, "posture": "unknown", "arms": "unknown", "head": "unknown", "facing_duck": "unknown", "region": "other",
          "motion": "still"}
MOTION_WIN = 1.5   # compare the box with ~1.5 s ago (events.classify_motion)


CROP_MARGIN, CROP_MAX = 0.25, 384     # crop: box expanded by 25%; shrink only if the long side exceeds 384 px
SEND_MODE = "crop"                    # what the perception loop sends to the VLM (see observe)


def prepare(img: Image.Image, mode: str = "full", box=None) -> Image.Image:
    """The image sent to the VLM: full = whole image; small = whole image at half size; crop = only the owner box
    found by the detector (expanded by CROP_MARGIN).
    Shrinking the whole image also shrinks the owner (hands and toy harder to see); cropping drops only background and
    keeps the owner's resolution -> fewer pixels (faster, cheaper) without losing detail (detector finds, VLM looks closely)."""
    if mode == "small":
        return img.resize((img.width // 2, img.height // 2))
    if mode == "crop" and box is not None:
        x1, y1, x2, y2 = box
        mx, my = CROP_MARGIN * (x2 - x1), CROP_MARGIN * (y2 - y1)
        c = img.crop((max(0, x1 - mx), max(0, y1 - my), min(img.width, x2 + mx), min(img.height, y2 + my)))
        k = CROP_MAX / max(c.size)
        return c.resize((round(c.width * k), round(c.height * k))) if k < 1 else c
    return img


def ask_state(img: Image.Image, mode: str = "full", box=None, usage: list | None = None, _tries: int = 2):
    """The VLM fills in OwnerState from a photo (strict schema). Returns (state, seconds).
    mode / box: see prepare(); usage: a list to which this call's input tokens are appended (cost measurement).
    An answer breaking OwnerState's rules -> ask once more; still invalid -> raise UnclearAnswer."""
    import openai
    img = prepare(img, mode, box)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    t0 = time.perf_counter()
    r = openai.OpenAI(timeout=30, max_retries=0).responses.create(
        model=MODEL,
        input=[{"role": "system", "content": SYSTEM_STATE},
               {"role": "user", "content": [{"type": "input_image", "image_url": url},
                                            {"type": "input_text", "text": "Fill in the owner's state."}]}],
        text={"format": {"type": "json_schema", "name": "OwnerState", "schema": openai_schema(), "strict": True}},
    )
    secs = time.perf_counter() - t0
    if usage is not None:
        usage.append(r.usage.input_tokens)
    try:
        return OwnerState.model_validate_json(r.output_text), secs
    except ValidationError as e:
        # The VLM answered "hugging knees but not sitting"; the validator rejected it (correctly) but nobody caught the
        # error and the whole perception loop crashed. Same rule as the planner: invalid -> ask again; still invalid ->
        # degrade (observe treats it as "not seen clearly")
        if _tries <= 1:
            raise UnclearAnswer(e.errors()[0]["msg"]) from e
        st, s2 = ask_state(img, "full", None, usage, _tries - 1)       # img is already prepared -> send it as is
        return st, secs + s2


class UnclearAnswer(ValueError):
    """The VLM still gives an invalid state after asking again (e.g. lying with hands on hips) -> the caller treats
    this observation as not seen clearly."""


def owner_xy(sim, cid, box):
    """Box -> roughly where the owner is (world). Floor plane cross-checked with known size (as in VisionWorld.observe);
    only needs to be good enough for the region."""
    p = landing(sim, cid, box)
    cx, cy = sim.data.cam_xpos[cid][:2]
    d_size = SIZE_K["owner"] / max(box[2] - box[0], 1)
    if p is None or math.hypot(p[0] - cx, p[1] - cy) > 1.5 * d_size + 0.3:
        b = math.radians(sim.xy_yaw()[2] + G.pixel_to_ground((box[0] + box[2]) / 2, G.CY)[0])
        p = (cx + d_size * math.cos(b), cy + d_size * math.sin(b))
    return p


# "Unknown" is said as unknown. Writing every answer other than "yes" as "not looking at you" meant an owner facing the
# duck with an "unknown" VLM answer was reported as not looking, and holding up a toy got 0/3 play invitations
# (3/3 when reported as looking). Not knowing must not be written as "no".
FACING_TEXT = {"yes": "looking at you", "no": "not looking at you", "unknown": "can't tell if looking at you"}


def describe(events: list, obs: dict) -> str:
    """Events + observed state -> one sentence for the planning LLM. Emotions are only "may be" (cues, not facts; owner_state.CUES)."""
    what = "; ".join(EVENT_TEXT[e] for e in events)
    bits = [f"{obs['posture']}", f"arms {obs['arms'].replace('_', ' ')}",
            FACING_TEXT.get(obs["facing_duck"], FACING_TEXT["unknown"]), f"in the {obs['region'].replace('_', ' ')}"]
    cue = CUES.get(obs["posture"]) or CUES.get(obs["arms"]) or (CUES["head_down"] if obs["head"] == "down" else None)
    return f"[camera] {what} ({', '.join(bits)}{', ' + cue if cue else ''})."


def describe_now(obs: dict) -> str:
    """How the owner looks *now*, without a new event (for re-asking: a re-ask sentence with only [self] left the LLM
    unaware that the owner was walking away)."""
    if not obs.get("present"):
        return "[camera] You cannot see the owner right now."
    bits = [f"{obs['posture']}", f"arms {obs['arms'].replace('_', ' ')}",
            FACING_TEXT.get(obs.get("facing_duck"), FACING_TEXT["unknown"]), f"in the {obs['region'].replace('_', ' ')}"]
    cue = CUES.get(obs["posture"]) or CUES.get(obs["arms"]) or (CUES["head_down"] if obs.get("head") == "down" else None)
    return f"[camera] The owner is now {', '.join(bits)}{', ' + cue if cue else ''}."


def compose(events: list, obs: dict, inner_text: str | None) -> str:
    """The whole sentence for the planning LLM: events (describe) + what mind.inner says (long cues, wants, drives)."""
    return " ".join(p for p in (describe(events, obs) if events else None, inner_text) if p)


WATCH_DT = 0.5         # how often the detector looks (free)
REFRESH_S = 6.0        # ask the VLM at least this often even if nothing changed (box unchanged, gesture changed)
VLM_COOLDOWN = 2.0     # after a shape-triggered VLM call, wait at least this long before the next
SHAPE_TOL = 0.25       # box height / width ratio differs by > 25% from the last VLM call = posture / gesture may have changed


# ─────────────────────────────────────────────────────────────
# Look up to confirm. Some events need "is the owner looking at me", but the VLM answers unknown: a standing duck's
# camera sees only below the owner's chest. "I saw the owner bring out a toy, of course I first check whether they
# are looking at me" -> confirming a fact (interpretation layer), not a behavior the LLM chooses.
# How: stand_look (official alpha_stand: lean back 15 deg + head 60 deg, camera 66-71 deg up, measured 10/10 without falling).
# Rules: only events about interacting with the owner (not came_home / leaving); still unseen after looking up ->
# say so; at most one look per event.
# ─────────────────────────────────────────────────────────────
def needs_look(events: list, obs: dict) -> bool:
    """Look up first to check whether the owner is looking at me? Only for interactive events, and only when facing
    is unknown ("no" is an answer: no need to look)."""
    return bool(set(events) & INTERACTIVE) and obs["facing_duck"] == "unknown"


INTERACTIVE = {"toy_out"}     # events that need to know whether the owner is looking at me
LOOK_S = 1.0                  # how long to hold the look (photo at the middle of stand_look)
LOOK_YAW_MAX = 60.0           # at most this much head turn toward the owner (no body turn: sit / stand / turn
                              # switches are riskier than missing one look)


def confirm_facing(vw, events: list, obs: dict, log: list) -> dict:
    """needs_look -> look up (head toward the owner) -> photo at the peak -> detector boxes the owner -> VLM -> update
    facing_duck only. Returns a new obs (the original is not modified). Other fields stay: the camera angle while looking
    up is not the calibrated one (the region would be wrong); posture and arms were already confirmed."""
    if not needs_look(events, obs):
        return obs
    from ..body.behaviors import bearing_to, stand_look
    sim = vw.sim
    arr, _ = shoot(sim, vw.r)
    found = vw.det(Image.fromarray(arr))["owner"]
    yaw = 0.0
    if found and found[1] >= MIN_SCORE["owner"]:
        yaw = max(-LOOK_YAW_MAX, min(LOOK_YAW_MAX, bearing_to(sim, owner_xy(sim, vw.cid, found[0]))))
    got = {}

    def snap(sim):
        img = Image.fromarray(shoot(sim, vw.r)[0])
        f = vw.det(img)["owner"]
        if f is None or f[1] < MIN_SCORE["owner"]:
            got["why"] = "detector found no owner after looking up"
            return
        try:
            st, secs = ask_state(img, SEND_MODE, f[0])
        except UnclearAnswer as e:
            got["why"] = f"invalid VLM answer ({e})"
            return
        got["facing"], got["why"] = st.facing_duck, f"VLM {secs:.1f}s"

    stand_look(sim, LOOK_S, yaw=yaw, at_peak=snap)
    new = dict(obs, facing_duck=got.get("facing", obs["facing_duck"]))
    log.append(f"t={sim.t:5.1f} looked up to confirm (head {yaw:+.0f} deg): facing {obs['facing_duck']} -> {new['facing_duck']} ({got.get('why')})")
    return new


def selftest() -> int:
    """needs_look (no simulator, no cost)."""
    O = lambda f: {"facing_duck": f}
    cases = [(["toy_out"], "unknown", True), (["toy_out"], "yes", False), (["toy_out"], "no", False),
             (["came_home"], "unknown", False), (["leaving"], "unknown", False), ([], "unknown", False),
             (["came_home", "toy_out"], "unknown", True)]
    ok = 0
    for ev, f, want in cases:
        got = needs_look(ev, O(f))
        ok += got == want
        print(f"  {'ok ' if got == want else 'BAD'} needs_look({ev}, facing={f!r}) -> {got} (want {want})")
    print(f"{ok}/{len(cases)}")
    return len(cases) - ok


class Perception:
    """The duck's continuous watching: observe() = one photo, one judgment, one update of the event detector."""

    def __init__(self, vw, detector: EventDetector | None = None):
        self.vw, self.ev = vw, detector or EventDetector()
        self.log, self.last = [], dict(ABSENT)
        self.vlm_calls = []          # [(sim time, seconds)]: the simulator pauses while waiting for the API -> add back for real latency
        self._vlm_box, self._vlm_toy, self._vlm_t, self._seen, self._vlm_aspect = None, False, -1e9, False, None
        self._hist, self.motion = [], "still"      # history of the owner box height (one entry per watch_step, 0.5 s)
        self.notices, self.on_notice = [], None    # small reactions: the detector noticed a change (before the VLM answers)

    def watch_step(self, at=None, hold_fn=None):
        """The detector looks every WATCH_DT s (free); the VLM is asked only when the image changed, and then twice in
        a row (debouncing without waiting another 2 s).

        Change = the owner appears / disappears, the box shape differs from the last VLM call, a yellow box appears at
        the owner's hands, or the last VLM call is too old (REFRESH_S).
        hold_fn(seconds): keeps the simulator running between the two calls (current posture). Returns (state, events)."""
        sim, vw = self.vw.sim, self.vw
        arr, box, why, toy_near, aspect = self._dino_look()
        if box is None:
            return self.last, self.ev.update(dict(ABSENT))
        if why is None:
            # shape unchanged -> posture and arms from the last VLM answer, only the detector's motion is new
            # (free, confirmed within 0.5 s)
            vw.last.update(box=box, status="ok", verdict=f"watching (DINO only): {self.motion}")
            if not self.last["present"]:
                return self.last, []
            obs = dict(self.last, motion=self.motion)
            self.last = obs
            ev = self.ev.update(obs)
            if ev:
                self.log.append(f"t={sim.t:5.1f} events {ev} (detector only: motion={self.motion})")
            return obs, ev
        self.log.append(f"t={sim.t:5.1f} DINO: {why} -> ask the VLM")
        obs, events = self.observe(at)
        if obs["present"] and self.ev.pending is not None and hold_fn is not None:
            hold_fn(WATCH_DT)                     # differs from what is believed -> look again right away to confirm
            obs, ev2 = self.observe(at)
            events += ev2
        self._vlm_box, self._vlm_toy, self._vlm_t, self._vlm_aspect = box, toy_near, sim.t, aspect
        return obs, events

    def _dino_look(self):
        """One free detector look: update motion, record small reactions, decide whether to ask the VLM
        (shared by the synchronous and background versions).
        Returns (photo, owner box or None, why to ask the VLM or None, yellow box at the hands, box aspect ratio)."""
        sim, vw = self.vw.sim, self.vw
        arr, _ = shoot(sim, vw.r)
        vw.last_img = arr
        found = vw.det(Image.fromarray(arr))
        person = found["owner"] if found["owner"] and found["owner"][1] >= MIN_SCORE["owner"] else None
        if person is None:
            self._hist, self.motion = [], "still"
            if self._seen:
                self.log.append(f"t={sim.t:5.1f} DINO: owner gone from view")
            self._seen = False
            vw.last.update(box=None, status="none", verdict="watching: no person")
            return arr, None, None, False, None
        box = person[0]
        # motion: compare the box height with ~1.5 s ago. A person's height does not change, so a shorter box = farther;
        # no distance in meters needed. The bottom (feet) is recorded too, for classify_motion.
        h = box[3] - box[1]
        self._hist = [e for e in self._hist if sim.t - e[0] <= MOTION_WIN + 0.01] + [(sim.t, h, box[3])]
        t_old, h_old, b_old = self._hist[0]
        if sim.t - t_old < MOTION_WIN - 0.6:
            self.motion = "still"                 # history too short to judge
        else:
            # shorter + feet up = walking away; shorter but feet not up = sitting down (sitting was reported as leaving)
            self.motion = classify_motion(h / max(h_old, 1), box[3] - b_old)
        # Holding a toy: the first version checked only the best yellow box, which was the big toy on the floor (0.69),
        # not the one in the owner's hands (0.24), and toy_out never fired in 30 s -> check *every* yellow box on the owner.
        # Score >= 0.2: being on the owner is evidence in itself (measured: holding a toy 0.24-0.25, only 0.04 of margin).
        # "On the owner" = at least 80% of the yellow box inside the owner's box (vision_world.held_toy): "any overlap"
        # accepted a green object behind the owner whose box only touched the edge of the owner's box (26% inside).
        toy_near = held_toy(getattr(vw.det, "toys", []), box) is not None
        # Trigger on *shape* (aspect ratio) only: walking toward or away makes the box bigger / smaller and moves it, but
        # keeps the shape; posture and gesture changes change the shape. Triggering on IoU asked the VLM every 0.5 s while
        # the owner walked (9 calls, 9-10 s real latency to report leaving).
        aspect = (box[3] - box[1]) / max(box[2] - box[0], 1)
        why = ("appeared" if not self._seen else
               # no close look while walking (turning and stepping keep changing the shape -> the VLM is asked again and
               # again -> "leaving" took 6-8 s) -> look once the owner is still; plus a cooldown (three calls in the 1.5 s
               # before motion was known). "Moving" means walking (receding / approaching) only: sitting down and standing
               # up (lowering / rising) are exactly the posture changes the VLM should look at.
               "shape changed" if self.motion not in ("receding", "approaching") and sim.t - self._vlm_t >= VLM_COOLDOWN
                                  and (self._vlm_aspect is None or abs(aspect / self._vlm_aspect - 1) > SHAPE_TOL) else
               "yellow box at the hands" if toy_near and not self._vlm_toy else
               "periodic" if sim.t - self._vlm_t > REFRESH_S else None)
        moved = self.motion != "still" and self.motion != getattr(self, "_noticed_motion", "still")
        self._noticed_motion = self.motion
        if why in ("appeared", "shape changed", "yellow box at the hands") or moved:
            # small reaction: the detector knows within 0.5 s that something happened -> look up and tilt the head first
            # (like a dog pricking its ears); the full behavior comes after the VLM confirms
            self.notices.append((sim.t, why or f"motion {self.motion}"))
            if self.on_notice:
                self.on_notice(why or f"motion {self.motion}")
        self._seen = True
        return arr, box, why, toy_near, aspect

    def observe(self, at=None):
        """Returns (observed state dict, event list).
        at: the owner's position if already known (e.g. just found while the duck is looking up) -> region from it.
            While looking up the camera tilt is not the calibrated one, so the floor-plane position cannot be used."""
        sim, vw = self.vw.sim, self.vw
        arr, _ = shoot(sim, vw.r)
        vw.last_img = arr
        img = Image.fromarray(arr)
        found = vw.det(img)["owner"]
        if found is None or found[1] < MIN_SCORE["owner"]:
            obs = dict(ABSENT)
            vw.last.update(box=None, status="none", verdict="no person")
        else:
            box = found[0]
            # Cropped: hugging knees 2 m away (owner only 74 x 113 px), the same photo sent whole got arms 0/3, head 0/3;
            # cropped 2/3, 3/3
            try:
                st, secs = ask_state(img, SEND_MODE, box)
            except UnclearAnswer as e:
                # degrade = not seen clearly this time (like "not seen is not absent": the event detector and the inner
                # state skip present=False and keep their memory)
                self.log.append(f"t={sim.t:5.1f} VLM answer still invalid after asking again ({e}) -> not seen clearly")
                vw.last.update(box=box, status="none", verdict="unclear")
                return dict(ABSENT), []
            self.vlm_calls.append((sim.t, secs))
            obs = st.model_dump()
            obs["region"] = (region(*(at or owner_xy(sim, vw.cid, box)))) if st.present else "other"
            obs["motion"] = self.motion
            vw.last.update(box=box, status="ok" if st.present else "none",
                           verdict=f"{obs['posture']}, {obs['arms']}, {obs['region']}")
            self.log.append(f"t={sim.t:5.1f} VLM {secs:.1f}s: {obs}")
        events = self.ev.update(obs)
        if events:
            self.log.append(f"t={sim.t:5.1f} events {events}")
        self.last = obs
        return obs, events


if __name__ == "__main__":
    raise SystemExit(selftest())
