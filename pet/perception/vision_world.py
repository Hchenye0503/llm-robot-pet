"""Connect the camera to the behavior library: world() stops reading ground truth and instead does
"stop -> photo -> detect -> verify -> geometry".

    owner: "I'm home!" -> LLM -> [auto] stand -> behavior(greet)
      -> run_steps reaches greet -> VisionWorld(sim, "greet")
           greet needs the owner -> nav.search.look_for
             observe: stand still 1 s -> photo -> Grounding DINO -> VLM check -> box touching an edge? -> geometry -> landing point
             turn   : turn in place
           -> {"owner": landing point, "owner_head": a point behind it}; not found -> {} -> head tilt ("where are you?")
      -> run_behavior is unchanged (it does not know whether the coordinates came from the camera or ground truth)

Assumptions to remember
    camera height and tilt are calibrated while standing -> photos taken sitting are off -> locate requires standing
    (behaviors that need a target stand up first); gaze cannot turn the body while sitting -> it uses the current view only
Detector (where) and VLM (whether it really is the target) are separate stages: a cheap model finds the box, an
expensive one looks closely at the crop.
"""

import base64
import io
import math
import time

import mujoco
import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict

from . import geometry as G
from ..body import sim as S
from ..body.behaviors import walk_for
from ..nav.search import look_for
from ..planner.llm import MODEL


# ═══════════════════════════════════════════════════════════════
# camera, detector, VLM verifier
# ═══════════════════════════════════════════════════════════════

def mount_camera(sim):
    """Align head_camera with the CAD camera mount (site head_camera), field of view fovy = G.FOVY_DEG.

    The camera angle follows the robot's own design, not a value tuned here. Upstream:
      site head_camera (the camera mount exported from CAD): x axis = optical axis, forward and 8.5 deg down when standing
      the upstream <camera> looks exactly along -x of the site (backward, 8.5 deg up): a sign error, not another design
      -> camera frame looks along -z with +y up: camera z = -site x, camera y = site z, camera x = -site y
    Field of view: the CAD has an M12 lens mount (lenses are interchangeable), so it cannot be read from the design;
    upstream fovy 45 is the MuJoCo default -> G.FOVY_DEG 90 is an assumption (a wide M12 lens)."""
    m, d = sim.model, sim.data
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "head_camera")
    q = m.site_quat[sid]
    R_bs = np.zeros(9)
    mujoco.mju_quat2Mat(R_bs, q)
    R_bs = R_bs.reshape(3, 3)
    R_bc = R_bs @ np.column_stack([[0, -1, 0], [0, 0, 1], [-1, 0, 0]])      # camera axes in the site frame: x = -y, y = z, z = -x
    qc = np.zeros(4)
    mujoco.mju_mat2Quat(qc, R_bc.flatten())
    m.cam_quat[cid], m.cam_pos[cid], m.cam_fovy[cid] = qc, m.site_pos[sid], G.FOVY_DEG
    mujoco.mj_forward(m, d)
    return cid


def sim_trunk(sim):
    m = sim.model
    return m.jnt_bodyid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")]


def shoot(sim, r):
    """One head-camera photo + perfect boxes from the segmentation image {"owner": [x1, y1, x2, y2] or None, "toy": ...}
    (the perfect boxes are only used for scoring)."""
    r.update_scene(sim.data, camera="head_camera")
    img = r.render().copy()
    r.enable_segmentation_rendering()
    r.update_scene(sim.data, camera="head_camera")
    seg = r.render().copy()
    r.disable_segmentation_rendering()
    m, boxes = sim.model, {}
    for name in ("owner", "toy"):
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        gids = [g for g in range(m.ngeom) if m.geom_bodyid[g] == bid]
        ys, xs = np.nonzero(np.isin(seg[..., 0], gids) & (seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM))
        boxes[name] = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) else None
    return img, boxes


def pose_ground(sim, cid, u, v):
    """Pixel ray -> floor using the camera's true pose right now. Returns (bearing deg relative to the trunk heading,
    horizontal distance or None)."""
    p, R = sim.data.cam_xpos[cid], sim.data.cam_xmat[cid].reshape(3, 3)
    ray = R @ np.array([(u - G.CX) / G.F, (G.CY - v) / G.F, -1.0])
    yaw = sim.xy_yaw()[2]
    b = math.degrees(math.atan2(ray[1], ray[0])) - yaw
    b = (b + 180) % 360 - 180
    if ray[2] >= 0:
        return b, None
    s = p[2] / -ray[2]
    return b, math.hypot(s * ray[0], s * ray[1])


def cam_state(sim, cid):
    """The camera's actual height and downward tilt (deg) right now."""
    fwd = -sim.data.cam_xmat[cid].reshape(3, 3)[:, 2]
    return float(sim.data.cam_xpos[cid][2]), math.degrees(math.asin(-fwd[2]))


# name -> (prompt for the detector, description for the VLM). Fixed in code.
# Missing a toy that is right there is worse for a pet than sniffing a yellow ball, so the description does not say
# "a cube": the check is looser, and shape is no longer a condition (a yellow ball decoy may pass as the toy).
TARGETS = {"owner": ("a person", "a person"), "toy": ("a yellow box", "a yellow toy block")}


# ─── 1. detector ───
class Detector:
    def __init__(self, mid="IDEA-Research/grounding-dino-tiny"):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        self.proc = AutoProcessor.from_pretrained(mid)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(mid).to("cuda").eval()
        self.text = " ".join(p + "." for p, _ in TARGETS.values())

    def __call__(self, img: Image.Image) -> dict:
        """Returns {"owner": (box, score) or None, "toy": ...}, the highest-scoring box per target."""
        import torch
        inp = self.proc(images=img, text=self.text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = self.model(**inp)
        # threshold 0.20: a toy in the owner's hands scores only 0.24; per-target thresholds are applied later (MIN_SCORE)
        res = self.proc.post_process_grounded_object_detection(
            out, inp.input_ids, threshold=0.20, text_threshold=0.20, target_sizes=[img.size[::-1]])[0]
        best = {k: None for k in TARGETS}
        self.toys = []               # *all* yellow boxes, not only the best one -> the perception loop checks if one is on the owner
        for b, s, lab in zip(res["boxes"], res["scores"], res["text_labels"]):
            k = "owner" if "person" in lab else "toy" if ("yellow" in lab or "box" in lab) else None
            if k == "toy":
                self.toys.append(([round(x) for x in b.tolist()], float(s)))
            if k and (best[k] is None or s > best[k][1]):
                best[k] = ([round(x) for x in b.tolist()], float(s))
        return best


# ─── 2. VLM verifier ───
class Verdict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    is_target: bool
    what_is_it: str


VLM_SYSTEM = ("You check object detections for a small robot dog's camera. You see one cropped region. "
              "Answer whether it really shows the named object. If not, say briefly what it is instead. "
              # the environment must be stated: the first version judged the owner "not a real person" 10/10
              # (literally true: it is made of capsules)
              "This is a simulated home rendered with simple 3D graphics: people are simplified 3D human models "
              "made of rounded shapes, and should count as a person. "
              # a red cube in front of the real toy made the VLM answer "not a single yellow block" -> state the rule
              "Answer yes if the named object is the main object in the crop, even if other objects are nearby "
              "or partly in front of it.")


def verify(img: Image.Image, box, desc: str) -> tuple[Verdict, dict]:
    """Crop the box (expanded by 20%) and ask the VLM. Returns (verdict, {seconds, input / output tokens})."""
    import openai
    x1, y1, x2, y2 = box
    mx, my = 0.2 * (x2 - x1), 0.2 * (y2 - y1)
    crop = img.crop((max(0, x1 - mx), max(0, y1 - my), min(G.W, x2 + mx), min(G.H_PX, y2 + my)))
    buf = io.BytesIO()
    crop.save(buf, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    schema = Verdict.model_json_schema()
    schema["additionalProperties"] = False
    t0 = time.perf_counter()
    r = openai.OpenAI(timeout=20, max_retries=0).responses.create(
        model=MODEL,
        input=[{"role": "system", "content": VLM_SYSTEM},
               {"role": "user", "content": [{"type": "input_image", "image_url": url},
                                            {"type": "input_text", "text": f"Is this {desc}?"}]}],
        text={"format": {"type": "json_schema", "name": "Verdict", "schema": schema, "strict": True}},
    )
    return Verdict.model_validate_json(r.output_text), {
        "s": time.perf_counter() - t0, "in": r.usage.input_tokens, "out": r.usage.output_tokens}


def landing(sim, cid, box):
    """Bottom center of the box -> geometry.pixel_to_ground -> landing point in world coordinates.
    The ray starts at the *camera*, about 8 cm in front of the trunk center (starting at the trunk shifted every
    landing point by 8 cm). On hardware: trunk position + the fixed camera offset from CAD."""
    b, d = G.pixel_to_ground((box[0] + box[2]) / 2, box[3])
    if d is None:
        return None
    _, _, yaw = sim.xy_yaw()
    x, y = sim.data.cam_xpos[cid][:2]
    a = math.radians(yaw + b)
    return x + d * math.cos(a), y + d * math.sin(a)


# ═══════════════════════════════════════════════════════════════
# vision for the behavior library
# ═══════════════════════════════════════════════════════════════

NEEDS = {"greet": "owner", "stay_close": "owner", "gaze": "owner", "play_bow": "owner", "guilty": "owner",
         "investigate": "toy"}
EDGE = 2            # box within 2 px of the left / right image edge = cut off
# Turn lead (deg): the duck keeps sliding after the stop command. Measured with the pet gait (no lead, 9 turns each
# way): left turns overshoot +0.2 to +1.6 deg, right turns +5.6 to +6.4 deg. Lead 6: right turns land exactly, left
# turns fall ~5 deg short (overlap 39 deg, still no blind spot). The left / right difference (5 deg, much larger than the
# 1-2 deg spread) is real, but separate values buy nothing (5 deg short does no harm) -> one number to re-measure on hardware.
TURN_LEAD = 6.0
# Detection score thresholds per target. Standing in front of the owner, the owner's legs filled the image and the
# detector called them "yellow box" at 0.29-0.38 -> "cut off" -> pointless turns -> 6 photos used, toy not found.
# Data so far: real toy 9/9 at 0.70-0.78; false toys 0.29-0.78 -> toy 0.4 cheaply drops weak junk, strong fakes
# (a green book at 0.7x) are left to the VLM. Owner: no real threshold; very close, a real owner's legs score only 0.30.
# These numbers were set while looking at the development and held-out sets: new positions are needed to claim they hold.
MIN_SCORE = {"owner": 0.25, "toy": 0.4}
HINT_SPOT = {"entrance": (0.05, 0.6), "living_room": (-2.0, -1.5)}   # sound came from a region -> first walk to where it can be seen
FAR_M, APPROACH_M = 1.3, 0.9   # estimated distance > 1.3 m -> walk to 0.9 m first, then look again (locate)
OFF_AXIS = 30.0     # box center more than this off the image center -> turn toward it before computing the landing point
NEAR_W = 0.45       # box width >= 45% of the image -> the box bottom cannot be trusted (observe)
LOOK_YAW_MAX = 60.0  # looking up for a held toy: at most this much head turn toward the owner (no body turn)
HEAD_BEHIND = 0.45  # sitting owner: head about 0.45 m behind the toes (the point to face, __call__)
SIZE_K = {"owner": 90.0, "toy": 41.0}   # distance (m) ~ K / box width (px). Owner: 85 px at 1.06 m, front view;
                                         # toy: 14 cm cube, 0.14 m front, 0.20 m diagonal -> 0.17 x F 240 ~ 41 (computed, not measured)
HELD_MIN = 0.2      # a yellow box overlapping the owner's box counts as held from this score (the overlap is evidence itself;
                    # measured: a held toy scores 0.24-0.25)
HELD_INSIDE = 0.8   # at least this fraction of the toy's box must lie inside the owner's box: a green object behind the owner,
                    # called "yellow box" at 0.24, only touched the edge of the owner's box (26% inside) and passed "any overlap"


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter, 1)

def _inside(toy_box, owner_box):
    ix = max(0, min(toy_box[2], owner_box[2]) - max(toy_box[0], owner_box[0]))
    iy = max(0, min(toy_box[3], owner_box[3]) - max(toy_box[1], owner_box[1]))
    inter = ix * iy

    toy_area = (toy_box[2] - toy_box[0]) * (toy_box[3] - toy_box[1])
    
    if toy_area <= 0:
        return False
        
    return (inter / toy_area) >= HELD_INSIDE

def held_toy(toys, owner_box, min_score=HELD_MIN):
    """Which toy is in the owner's hands? Only the best-scoring yellow box used to be checked, and the big toy on the
    floor (0.69) always beat the smaller, partly covered one in the owner's hands (0.24): the duck walked away from an
    owner holding up a toy to sniff the floor toy.

    toys: [(box, score), ...] with box = [x1, y1, x2, y2]; owner_box: the owner's box, or None (owner not seen)
    Returns the highest-scoring (box, score) among the toys with score >= min_score and at least HELD_INSIDE of their own
    area inside the owner's box (overlap area / toy area); None if there is none."""
    if owner_box is None:
        return None 
    best = None
    for box, score in toys:
        if score < min_score:
            continue
        if _inside(box, owner_box):
            if best is None or score > best[1]:
                best = (box, score)
    return best


class VisionWorld:
    """A callable world: VisionWorld(sim, behavior name) -> dict. Also remembers the last photo and box (for video overlays)."""

    def __init__(self, sim, use_vlm=True, det=None, servo=False):
        # With the pet gait the camera bobs 13.5-31.6 deg while walking and tracking on the move lost the target ->
        # once the position is known there is no need to keep photographing: coarse to fine (stop, head level, look)
        # is the default; servo=True is kept for comparison.
        self.sim, self.use_vlm, self.servo = sim, use_vlm, servo
        self.cid = mount_camera(sim)
        self.r = mujoco.Renderer(sim.model, G.H_PX, G.W)
        self.det = det or Detector()                              # can be shared (loading the model takes seconds)
        self.last = {"box": None, "status": "", "verdict": ""}    # drawn on videos
        self.found = {}              # last landing point per target (to face it without peeking at ground truth)
        self.hint = None             # which region a sound came from (microphone): locate walks there first
        self.held = None             # set by observe("toy"): the owner box when the toy is in their hands
        self.log = []

    def observe(self, target):
        sim = self.sim
        from ..body.behaviors import _idle
        from ..body.scheduler import hold
        hold(sim, _idle(sim), 1.0)                                  # stand still 1 s before the photo (keep the current posture)
        arr, _ = shoot(sim, self.r)
        # Use the camera height and tilt at the moment of the photo: with the pet gait the standing camera tilt is
        # 10.0-20.4 deg (median 12.7); a fixed 9.6 put distances too far (2.7 and 5.0 m for two start positions).
        # Hardware: IMU body tilt + neck / head servo encoders -> forward kinematics; the simulator reads the camera.
        G.CAM_H, G.TILT_DEG = cam_state(sim, self.cid)
        self.last_img = arr
        seen = self.det(Image.fromarray(arr))
        found = seen[target]
        self.held = None                                            # the owner's box when the toy is in their hands
        if target == "toy":
            o = seen["owner"]
            h = held_toy(getattr(self.det, "toys", []), o[0] if o and o[1] >= MIN_SCORE["owner"] else None)
            if h is not None:
                found, self.held = h, o[0]
        self.last = {"box": None, "status": "none", "verdict": ""}
        if found is None:
            self.log.append(f"  photo: no {target}")
            return "none", None
        box, score = found
        if self.held is not None:
            self.log.append(f"  photo: toy {box} score {score:.2f} in the owner's hands (beats the best-scoring yellow box)")
        elif score < MIN_SCORE[target]:
            self.log.append(f"  photo: {target} but score {score:.2f} < {MIN_SCORE[target]} -> treat as none")
            self.last["status"] = "none"
            return "none", None
        self.last["box"] = box
        self.log.append(f"  photo: {target} {box} score {score:.2f}")
        # Edges before the VLM: a small piece of leg was judged "not a person" -> "none" -> the duck turned past the owner.
        # A small piece is unrecognizable anyway (for people too): turn toward it first, verify once it is fully visible.
        if target == "toy" and box[2] - box[0] >= NEAR_W * G.W:
            # In front of the owner the detector boxed the owner's body as "yellow box" (0.46-0.74, 80% wide) -> touching
            # the left edge -> kept turning left -> 6 photos used. Known size again: a 14 cm toy cannot fill half the image
            # unless the camera is touching it -> junk (must come before the edge check)
            self.log.append(f"  toy box {box[2] - box[0]} px wide: a 14 cm toy cannot be that big -> junk box")
            self.last["status"] = "none"
            return "none", None
        if box[0] <= EDGE and box[2] >= G.W - 1 - EDGE:
            # the detector sometimes returns a box spanning the whole image (once on an all-black photo): not cut off, junk
            self.log.append("  box touches both edges -> junk box, treat as none")
            self.last["status"] = "none"
            return "none", None
        if box[0] <= EDGE:
            self.last["status"] = "cut_left"
            return "cut_left", None
        if box[2] >= G.W - 1 - EDGE:
            self.last["status"] = "cut_right"
            return "cut_right", None
        if self.use_vlm:
            v, st = verify(Image.fromarray(arr), box, TARGETS[target][1])
            self.last["verdict"] = f"VLM: {'yes' if v.is_target else 'no'} ({v.what_is_it[:40]})"
            self.log.append(f"VLM {st['s']:.1f} s: {v.is_target} {v.what_is_it}")
            if not v.is_target:
                return "none", None
        b_center = G.pixel_to_ground((box[0] + box[2]) / 2, G.CY)[0]
        if abs(b_center) > OFF_AXIS:
            # With the shallow CAD camera tilt the rays are flatter and landing errors grow toward the image edges
            # (a 34 cm miss with the owner at the edge) -> turn toward it first and compute when centered
            # (active perception, reusing look_for's cut status: turn 30 deg that way)
            self.log.append(f"  box center {b_center:+.0f} deg > {OFF_AXIS:g} deg -> center it first")
            self.last["status"] = "cut_left" if b_center > 0 else "cut_right"
            return self.last["status"], None
        wide, low = box[2] - box[0] >= NEAR_W * G.W, box[3] >= G.H_PX - 1 - EDGE
        if wide or low:
            # Too close for "box bottom = feet": the bottom touching the lower edge means the feet are out of frame; a
            # wide box ending at the knees would put the landing point at the owner's hips -> walked into them and fell.
            # "Treat as arrived" was wrong too: 45% width triggers at 0.5 m (legs apart) and the play bow never came closer.
            # -> estimate the owner's distance from known size: 85 px wide at 1.06 m -> distance ~ SIZE_K / width.
            # Seen at an angle the box is wider -> estimated too close -> walks a little less (errs on the safe side).
            u, w = (box[0] + box[2]) / 2, box[2] - box[0]
            cx, cy = sim.data.cam_xpos[self.cid][:2]
            if target == "owner":
                d = SIZE_K["owner"] / w
                b = math.radians(sim.xy_yaw()[2] + G.pixel_to_ground(u, G.H_PX - 1)[0])
                p = (cx + d * math.cos(b), cy + d * math.sin(b))
                self.log.append(f"  box {'wide' if wide else 'touches the bottom'} ({w} px) -> bottom is not the feet -> size estimate {d:.2f} m")
            else:
                p = (cx, cy)                                        # toy size not calibrated -> treat as already there
                self.log.append("  box too big / touches the bottom -> already very close")
            self.last["status"] = "ok"
            return "ok", p
        if self.held is not None:
            # A held toy is in the air: its box bottom is not on the floor, and the floor-plane point would land far behind
            # the owner. Where the toy is on the floor = where the owner stands -> estimate from the owner's box.
            p = self.estimate_xy(self.held, "owner")
            self.log.append(f"  toy in the owner's hands -> go to the owner ({p[0]:+.2f}, {p[1]:+.2f})")
            self.last["status"] = "ok"
            return "ok", p
        p = landing(sim, self.cid, box)
        cx, cy = sim.data.cam_xpos[self.cid][:2]
        d_size = SIZE_K[target] / max(box[2] - box[0], 1)
        d_ground = math.hypot(p[0] - cx, p[1] - cy) if p else float("inf")
        from ..nav.route import feet_hidden
        if d_ground > 1.5 * d_size + 0.3 and not feet_hidden(sim.model, (cx, cy), p):
            # The two estimates disagree, but no furniture lies on that line of sight on the map -> the feet are visible
            # -> the floor plane is right (an owner seen at an angle with legs stretched out has a wide box, so the size
            # estimate is too close; the duck stopped 0.8-1 m short)
            self.log.append(f"  floor plane {d_ground:.1f} m >> size {d_size:.1f} m, but the line of sight is clear on the map -> trust the floor plane")
        elif d_ground > 1.5 * d_size + 0.3:
            # The coffee table hid the owner's feet: the box bottom stopped at the table top near the horizon and the
            # floor plane said 5.6 m (outside the apartment). Two independent estimates cross-checked (the verifier idea):
            # floor plane much farther than known size = the contact point is hidden -> trust the size.
            # With a map this branch is reached only after the line of sight was found blocked (or the point is outside).
            b = math.radians(sim.xy_yaw()[2] + G.pixel_to_ground((box[0] + box[2]) / 2, G.CY)[0])
            p = (cx + d_size * math.cos(b), cy + d_size * math.sin(b))
            self.log.append(f"  floor plane {d_ground:.1f} m >> size {d_size:.1f} m -> contact point may be hidden, use size")
        self.last["status"] = "ok" if p else "none"
        return ("ok", p) if p else ("none", None)

    def turn(self, deg):
        """Turn in place by deg (positive = left), *closed loop on the IMU heading*: stop when turned.
        Open loop ("43 deg/s x time") turned 120 deg for a 72 deg command, more than the 106 deg field of view,
        leaving a blind spot in the scan. Command is not outcome: measure. On hardware the heading comes from the IMU."""
        sim = self.sim
        y0, t0 = sim.xy_yaw()[2], sim.t
        while sim.t - t0 < 6.0:
            done = (sim.xy_yaw()[2] - y0 + 180) % 360 - 180
            if abs(done) >= abs(deg) - TURN_LEAD:
                break
            walk_for(sim, 0, math.copysign(43, deg), S.CONTROL_DT)
        walk_for(sim, 0, 0, 0.3)

    def locate(self, target):
        """Find target's landing point, coarse to fine.

        Far estimates were wrong (1.8 m off by 36 cm, 2.0 m by 62 cm): a geometric limit. The camera is 24 cm above the
        floor, so at 2 m the ray grazes the floor at 6.8 deg and 1 px of box bottom is ~7 cm of distance
        (error ~ distance^2 / camera height). -> when far, trust the direction, walk to ~0.9 m, then look again."""
        from ..nav.route import clear, goto_planned as goto, grid_for   # follow A* when furniture is in the way
        if self.hint:
            # The microphone knows the sound came from the entrance, but the camera cannot see through the wall ->
            # walk along the waypoints to where that region is visible, then look (a dog hears, runs over, then looks)
            from ..nav.waypoints import navigate, region
            x, y, _ = self.sim.xy_yaw()
            if region(x, y) != self.hint and self.hint in HINT_SPOT:
                self.log.append(f"  sound from {self.hint} -> walk the waypoints to {HINT_SPOT[self.hint]}")
                navigate(self.sim, HINT_SPOT[self.hint], self.log, stop_dist=0.2)
            self.hint = None
        p = look_for(self.observe, self.turn, target, self.log)
        if p is None:
            return None
        cx, cy = self.sim.data.cam_xpos[self.cid][:2]
        d = math.hypot(p[0] - cx, p[1] - cy)
        grid = grid_for(self.sim.model)
        blocked = grid is not None and not clear(grid, (cx, cy), p)
        if blocked:
            self.log.append("  straight line blocked by furniture -> no visual servoing (it only chases straight), A* closer then look")
        if d > FAR_M and self.servo and self.last.get("box") and not blocked:
            # visual servoing: the VLM already confirmed; track the same detector box while walking, no stop
            from ..nav.servo import servo
            self.log.append(f"  estimated {d:.1f} m > {FAR_M} m: track while walking (visual servoing)")
            r = servo(self.sim, self, target, self.last["box"], self.log)
            if r["reached"] and r.get("box"):
                p = self.estimate_xy(r["box"], target)          # position from the last tracked box (no VLM)
            else:                                               # lost or bumped -> back to the old way: stop and search
                p2 = look_for(self.observe, self.turn, target, self.log)
                p = p2 if p2 is not None else p
        elif d > FAR_M:
            self.log.append(f"  estimated {d:.1f} m > {FAR_M} m: walk to ~{APPROACH_M} m first, then look again")
            goto(self.sim, p, stop_dist=APPROACH_M, log=self.log)
            p2 = look_for(self.observe, self.turn, target, self.log)
            p = p2 if p2 is not None else p
        self.found[target] = p
        return p

    def estimate_xy(self, box, target):
        """Box -> roughly where the target is (world): floor plane; too close (wide / touching the bottom) or far from
        the known-size estimate -> use the size."""
        sim = self.sim
        cx, cy = sim.data.cam_xpos[self.cid][:2]
        w = max(box[2] - box[0], 1)
        d_size = SIZE_K[target] / w
        p = landing(sim, self.cid, box)
        near = w >= NEAR_W * G.W or box[3] >= G.H_PX - 1 - EDGE
        from ..nav.route import feet_hidden                     # when the two disagree, check the line of sight on the map first
        if p is None or near or (math.hypot(p[0] - cx, p[1] - cy) > 1.5 * d_size + 0.3 and feet_hidden(sim.model, (cx, cy), p)):
            b = math.radians(sim.xy_yaw()[2] + G.pixel_to_ground((box[0] + box[2]) / 2, G.CY)[0])
            p = (cx + d_size * math.cos(b), cy + d_size * math.sin(b))
        return p

    def look_up_for_held_toy(self):
        """Is the toy in the owner's hands? The camera is 24 cm above the floor: from ~2 m it sees a standing owner only up
        to the waist, and a toy held at the chest (~1 m) is not in the photo at all, so the floor toy was the only one
        found. Same answer as for "is the owner looking at me": look up first.
          1. level photo (calibrated camera) -> where the owner stands
          2. lean back and look up toward the owner (stand_look) -> photo at the peak -> held_toy on that photo
             (only box positions are used: the camera angle while looking up is not calibrated, so no distances)
        Returns the owner's position if a toy is in their hands, else None (then search the floor as before)."""
        from ..body.behaviors import _idle, bearing_to, stand_look
        from ..body.scheduler import hold
        sim = self.sim
        hold(sim, _idle(sim), 1.0)
        arr, _ = shoot(sim, self.r)
        G.CAM_H, G.TILT_DEG = cam_state(sim, self.cid)
        seen = self.det(Image.fromarray(arr))
        held = held_toy(getattr(self.det, "toys", []), seen["owner"][0] if seen["owner"] and seen["owner"][1] >= MIN_SCORE["owner"] else None)
        if seen["owner"] is None or seen["owner"][1] < MIN_SCORE["owner"]:
            return None                                             # no owner in view: nobody can be holding it
        p_owner = self.estimate_xy(seen["owner"][0], "owner")
        if held is None:
            yaw = max(-LOOK_YAW_MAX, min(LOOK_YAW_MAX, bearing_to(sim, p_owner)))
            got = {}

            def snap(sim):
                img = Image.fromarray(shoot(sim, self.r)[0])
                self.last_img = np.asarray(img)
                s2 = self.det(img)
                o = s2["owner"][0] if s2["owner"] and s2["owner"][1] >= MIN_SCORE["owner"] else None
                got["held"] = held_toy(getattr(self.det, "toys", []), o)
                if got["held"] is not None:
                    self.last.update(box=got["held"][0], status="ok", verdict="toy in the owner's hands")

            stand_look(sim, 1.0, yaw=yaw, at_peak=snap)
            held = got.get("held")
            self.log.append(f"  looked up toward the owner (head {yaw:+.0f} deg): "
                            + (f"toy in their hands, score {held[1]:.2f}" if held else "no toy in their hands"))
        else:
            self.log.append(f"  toy in the owner's hands in the level photo, score {held[1]:.2f}")
        return p_owner if held is not None else None

    def __call__(self, sim, name):
        target = NEEDS.get(name)
        if target is None:
            return {}
        from ..body.scheduler import sense_posture
        if sense_posture(sim) == "sitting":
            # Sitting cannot turn in place (gaze turning while sitting fell over) -> only the current photo; the camera was
            # calibrated standing -> distance is off, direction is fine
            st, p = self.observe(target)
            p = p if st == "ok" else None
            self.log.append(f"sitting: one photo only -> {st}")
        else:
            p = None
            if target == "toy":
                p = self.look_up_for_held_toy()
                self.held = True if p is not None else None
            if p is None:
                p = self.locate(target)
        self.last["box"] = None                                     # video: remove the old box once walking starts
        if p is None:
            from ..body.scheduler import run_steps
            from ..planner.schema import Gesture
            run_steps(sim, [Gesture(action="gesture", name="head_tilt")], self.log)   # "huh? where are you?"
            return {}
        # The point to *face* cannot be the toes: stay_close stops 0.25 m from the toes and turns toward them -> the target
        # is so close that the bearing jumps with every turn -> turns forever -> sits before settling -> falls.
        # A sitting owner's head is about 0.45 m behind the toes -> push 0.45 m further along camera -> toes.
        cx, cy = sim.data.cam_xpos[self.cid][:2]
        d = math.hypot(p[0] - cx, p[1] - cy) or 1.0
        head = (p[0] + HEAD_BEHIND * (p[0] - cx) / d, p[1] + HEAD_BEHIND * (p[1] - cy) / d)
        if target == "toy":
            if getattr(self, "held", None) is not None:
                return {"toy": p, "toy_held": True, "owner": p, "owner_head": head}
            return {"toy": p}
        return {"owner": p, "owner_head": head}


# ─────────────────────────────────────────────────────────────
# selftest (no simulator, no detector, no cost)
# ─────────────────────────────────────────────────────────────
def selftest() -> int:
    OWNER = [300, 100, 400, 460]                     # a standing owner in the middle of the image
    IN_HANDS = ([330, 200, 370, 240], 0.24)          # small, overlaps the owner
    FLOOR = ([100, 380, 220, 470], 0.69)             # big, on the floor, far from the owner
    WEAK = ([340, 300, 360, 320], 0.12)              # overlaps the owner but too weak (e.g. a knee called "yellow box")
    cases = [
        ("toy in hands + bigger toy on the floor -> the one in hands", [FLOOR, IN_HANDS], OWNER, IN_HANDS),
        ("only the floor toy -> None (nothing held; the caller falls back to the best score)", [FLOOR], OWNER, None),
        ("overlapping but below the threshold -> None", [WEAK, FLOOR], OWNER, None),
        ("two toys on the owner -> the higher score", [IN_HANDS, ([310, 150, 350, 190], 0.31)], OWNER, ([310, 150, 350, 190], 0.31)),
        ("no toys at all -> None", [], OWNER, None),
        ("owner not seen -> None", [FLOOR, IN_HANDS], None, None),
        ("exactly at the threshold -> counts", [([330, 200, 370, 240], 0.2)], OWNER, ([330, 200, 370, 240], 0.2)),
        ("background object touching the edge of the owner's box (26% inside) -> None",
         [([229, 152, 260, 193], 0.24)], [252, 1, 349, 211], None),
        ("toy mostly inside the owner's box (90%) -> counts", [([296, 200, 336, 240], 0.24)], OWNER, ([296, 200, 336, 240], 0.24)),
    ]
    ok = 0
    for desc, toys, owner, want in cases:
        try:
            got = held_toy(toys, owner)
        except NotImplementedError:
            got = "not written"
        good = got == want
        ok += good
        print(f"{'ok ' if good else 'BAD'} {desc}\n     got {got}" + ("" if good else f" (want {want})"))
    print(f"\n{ok}/{len(cases)}")
    return len(cases) - ok


if __name__ == "__main__":
    raise SystemExit(selftest())
