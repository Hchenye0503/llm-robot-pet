"""A moving owner: walks in, sits, lies down, stands, waves, hands on hips, points, opens arms, hugs knees,
hangs the head, holds up a toy.

This is a stage prop, not part of the duck's brain. The owner's actions are scripted, so the true state is
known at every moment (truth()) and is used to score what the duck perceives.

How it moves
    the owner is one mocap body (positioned by code, not by physics, but the duck still collides with it)
    the body is capsules + spheres; each pose is a set of joint points (relative to the pelvis, facing +x),
    and every 20 ms the capsules are placed on those points
    changing pose = interpolating between two sets of points (the keyframe idea); walking = moving the root along
    a path while swinging arms and legs

Poses (operational definitions: what can be seen; the emotion is only the observer's guess, so it is not in the name)
    sit        on the floor, hands on the knees         sit_head_down  sitting, head hanging forward
    hug_knees  sitting, arms around the knees, head low  lie            lying on the floor, arms along the body
    stand      standing, arms down                       wave           standing, right hand high, waving
    hips       standing, hands on hips                   point          standing, right arm straight, pointing down-forward
    open_arms  standing, arms spread                     hold_toy       standing, a yellow toy held in both hands at the chest

Usage
    sim = DuckSim(stage.SCENE, ..., spec_edit=stage_live)
    owner = Owner(sim)                          # hooks itself onto sim.step: updates every step
    owner.do("appear", at=ENTRY, yaw=-90)       # actions are queued and run in order
    owner.do("walk", path=[...]); owner.do("pose", name="sit"); owner.do("wait", secs=2)
"""

import math

import mujoco
import numpy as np

from . import stage as T

SKIN, SHIRT, JEANS, SOCK = T.SKIN, T.SHIRT, T.JEANS, T.SOCK
HAIR, EYE = [0.20, 0.14, 0.10, 1], [0.08, 0.06, 0.05, 1]
TOY_RGBA = [0.98, 0.80, 0.10, 1]
AWAY = (0.05, 4.5)                     # "not home" = outside the north wall (the duck cannot see it)
SIT_SPOT = (T.OWNER[0], T.OWNER[1])     # where the owner sits in the living room (same as the static owner in stage.py)

# segments: (name, from joint, to joint, radius, color)
SEGS = [("thigh_L", "hip_L", "knee_L", 0.065, JEANS), ("thigh_R", "hip_R", "knee_R", 0.065, JEANS),
        ("shin_L", "knee_L", "ankle_L", 0.055, JEANS), ("shin_R", "knee_R", "ankle_R", 0.055, JEANS),
        ("foot_L", "ankle_L", "toe_L", 0.045, SOCK), ("foot_R", "ankle_R", "toe_R", 0.045, SOCK),
        ("torso", "pelvis", "chest", 0.14, SHIRT), ("neck", "neck0", "neck1", 0.05, SKIN),
        ("uarm_L", "shoulder_L", "elbow_L", 0.042, SHIRT), ("uarm_R", "shoulder_R", "elbow_R", 0.042, SHIRT),
        ("farm_L", "elbow_L", "hand_L", 0.036, SHIRT), ("farm_R", "elbow_R", "hand_R", 0.036, SHIRT)]
BALLS = [("hand_L", 0.045, SKIN), ("hand_R", 0.045, SKIN)]


def _mirror(pts):
    """Only the left side (y > 0) and the midline are written; the right side is mirrored."""
    out = dict(pts)
    for k, v in pts.items():
        if k.endswith("_L") and k[:-2] + "_R" not in pts:
            out[k[:-2] + "_R"] = (v[0], -v[1], v[2])
    return out


def _sym(legs, torso, arms_L, arms_R=None, head=(0.0, 0.0), toy=False):
    p = dict(legs)
    p.update(torso)
    p.update(arms_L)
    if arms_R:
        p.update(arms_R)
    p = _mirror(p)
    p["_head"] = head            # (yaw left positive, pitch down positive), degrees
    p["_toy"] = toy
    return p


# ── sitting family (pelvis near (0, 0, 0.1)) ──
_SIT_LEGS = {"hip_L": (0.0, 0.10, 0.10), "knee_L": (0.28, 0.13, 0.30), "ankle_L": (0.42, 0.13, 0.05), "toe_L": (0.52, 0.13, 0.04)}
_SIT_TORSO = {"pelvis": (0.0, 0.0, 0.14), "chest": (0.04, 0.0, 0.52), "neck0": (0.05, 0.0, 0.55), "neck1": (0.06, 0.0, 0.60), "head": (0.08, 0.0, 0.70)}
_SIT_ARMS = {"shoulder_L": (0.02, 0.20, 0.50), "elbow_L": (0.07, 0.21, 0.37), "hand_L": (0.24, 0.15, 0.33)}
# ── standing family (pelvis 0.86 m) ──
_ST_LEGS = {"hip_L": (0.0, 0.10, 0.84), "knee_L": (0.02, 0.10, 0.46), "ankle_L": (0.0, 0.10, 0.08), "toe_L": (0.13, 0.10, 0.04)}
_ST_TORSO = {"pelvis": (0.0, 0.0, 0.88), "chest": (0.0, 0.0, 1.24), "neck0": (0.0, 0.0, 1.30), "neck1": (0.0, 0.0, 1.36), "head": (0.01, 0.0, 1.47)}
_ST_ARMS = {"shoulder_L": (0.0, 0.20, 1.27), "elbow_L": (0.0, 0.23, 1.02), "hand_L": (0.03, 0.22, 0.80)}


def _pose(name, t):
    """Joint points of pose `name` at time t (owner frame: origin on the floor under the pelvis, facing +x)."""
    if name == "sit":
        return _sym(_SIT_LEGS, _SIT_TORSO, _SIT_ARMS)
    if name == "sit_head_down":
        tor = dict(_SIT_TORSO, chest=(0.10, 0.0, 0.48), neck0=(0.12, 0, 0.51), neck1=(0.16, 0, 0.55), head=(0.22, 0.0, 0.60))
        return _sym(_SIT_LEGS, tor, {"shoulder_L": (0.07, 0.20, 0.46), "elbow_L": (0.12, 0.22, 0.33), "hand_L": (0.26, 0.15, 0.31)}, head=(0, 45))
    if name == "hug_knees":
        legs = {"hip_L": (0.0, 0.09, 0.10), "knee_L": (0.22, 0.10, 0.40), "ankle_L": (0.34, 0.10, 0.05), "toe_L": (0.44, 0.10, 0.04)}
        tor = dict(_SIT_TORSO, chest=(0.08, 0.0, 0.50), neck0=(0.10, 0, 0.53), neck1=(0.13, 0, 0.57), head=(0.18, 0.0, 0.62))
        arms = {"shoulder_L": (0.06, 0.20, 0.48), "elbow_L": (0.22, 0.22, 0.44), "hand_L": (0.32, 0.03, 0.40)}
        return _sym(legs, tor, arms, head=(0, 40))
    if name == "lie":
        legs = {"hip_L": (0.0, 0.10, 0.12), "knee_L": (0.42, 0.11, 0.09), "ankle_L": (0.84, 0.12, 0.08), "toe_L": (0.87, 0.12, 0.19)}
        tor = {"pelvis": (-0.02, 0.0, 0.14), "chest": (-0.40, 0.0, 0.14), "neck0": (-0.46, 0, 0.13), "neck1": (-0.52, 0, 0.12), "head": (-0.62, 0.0, 0.11)}
        arms = {"shoulder_L": (-0.38, 0.22, 0.12), "elbow_L": (-0.14, 0.25, 0.07), "hand_L": (0.08, 0.25, 0.06)}
        return _sym(legs, tor, arms, head=(0, -90))
    if name == "stand":
        return _sym(_ST_LEGS, _ST_TORSO, _ST_ARMS)
    if name == "wave":
        sw = 0.12 * math.sin(2 * math.pi * 1.5 * t)
        right = {"shoulder_R": (0.0, -0.20, 1.27), "elbow_R": (0.05, -0.36, 1.42), "hand_R": (0.06, -0.33 + sw, 1.68)}
        return _sym(_ST_LEGS, _ST_TORSO, _ST_ARMS, right, head=(0, -5))
    if name == "hips":
        return _sym(_ST_LEGS, _ST_TORSO, {"shoulder_L": (0.0, 0.20, 1.27), "elbow_L": (-0.06, 0.38, 1.06), "hand_L": (0.0, 0.17, 0.92)}, head=(0, 15))
    if name == "point":
        right = {"shoulder_R": (0.0, -0.20, 1.27), "elbow_R": (0.26, -0.20, 1.17), "hand_R": (0.54, -0.18, 1.02)}
        return _sym(_ST_LEGS, _ST_TORSO, _ST_ARMS, right, head=(0, 25))
    if name == "open_arms":
        return _sym(_ST_LEGS, _ST_TORSO, {"shoulder_L": (0.0, 0.20, 1.27), "elbow_L": (0.10, 0.42, 1.16), "hand_L": (0.24, 0.58, 1.08)})
    if name == "hold_toy":
        arms = {"shoulder_L": (0.0, 0.20, 1.27), "elbow_L": (0.16, 0.22, 1.03), "hand_L": (0.34, 0.08, 1.04)}
        return _sym(_ST_LEGS, _ST_TORSO, arms, head=(0, 20), toy=True)
    if name == "walk":
        ph = 2 * math.pi * 1.6 * t                     # 1.6 steps per second
        p = _sym(_ST_LEGS, _ST_TORSO, _ST_ARMS)
        for side, sgn in (("L", 1), ("R", -1)):
            s = sgn * math.sin(ph)
            y = p[f"hip_{side}"][1]
            p[f"knee_{side}"] = (0.02 + 0.14 * s, y, 0.47)
            p[f"ankle_{side}"] = (0.26 * s, y, 0.08 + 0.05 * max(0.0, s))
            p[f"toe_{side}"] = (0.26 * s + 0.13, y, 0.04 + 0.05 * max(0.0, s))
            p[f"elbow_{side}"] = (-0.10 * s, p[f"elbow_{side}"][1], 1.02)
            p[f"hand_{side}"] = (-0.18 * s + 0.03, p[f"hand_{side}"][1], 0.81)
        return p
    raise KeyError(name)


POSES = ("sit", "sit_head_down", "hug_knees", "lie", "stand", "wave", "hips", "point", "open_arms", "hold_toy")
SITTING, LYING = ("sit", "sit_head_down", "hug_knees"), ("lie",)


def _blend(a, b, w):
    out = {}
    for k in a:
        if k == "_toy":
            out[k] = b[k] if w > 0.5 else a[k]
        else:
            out[k] = tuple(x + (y - x) * w for x, y in zip(a[k], b[k]))
    return out


def stage_live(spec):
    """Stage without the static owner, plus one mocap owner."""
    T.stage(spec, owner=False)
    o = spec.worldbody.add_body(name="owner", pos=[AWAY[0], AWAY[1], 0], mocap=True)
    for name, _, _, r, rgba in SEGS:
        g = o.add_geom(name=f"owner_{name}")
        g.type, g.size, g.rgba = mujoco.mjtGeom.mjGEOM_CAPSULE, [r, 0.05, 0], rgba
    for name, r, rgba in BALLS + [("head", 0.10, SKIN), ("hair", 0.095, HAIR), ("eye_L", 0.016, EYE), ("eye_R", 0.016, EYE),
                                  ("nose", 0.022, [0.85, 0.62, 0.50, 1])]:
        g = o.add_geom(name=f"owner_{name}")
        g.type, g.size, g.rgba = mujoco.mjtGeom.mjGEOM_SPHERE, [r, 0, 0], rgba
    g = o.add_geom(name="owner_heldtoy")                     # toy in the hands: transparent and non-colliding when not held
    g.type, g.size, g.rgba = mujoco.mjtGeom.mjGEOM_BOX, [0.06, 0.06, 0.06], TOY_RGBA
    g.contype = g.conaffinity = 0


def _quat_z_to(v):
    """Quaternion that rotates the z axis onto v (capsules lie along their own z axis)."""
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    if n < 1e-9:
        return np.array([1.0, 0, 0, 0])
    v = v / n
    z = np.array([0, 0, 1.0])
    c = float(np.dot(z, v))
    if c < -0.999999:
        return np.array([0.0, 1, 0, 0])
    axis = np.cross(z, v)
    q = np.array([1 + c, *axis])
    return q / np.linalg.norm(q)


class Owner:
    """The moving owner. Actions are queued and run in order; updated automatically after every sim.step."""

    def __init__(self, sim, pose="sit", at=AWAY, yaw=0.0):
        self.sim, m = sim, sim.model
        self.mid = m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "owner")]
        g = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"owner_{n}")
        self.g = {n: g(n) for n, *_ in SEGS}
        self.g.update({n: g(n) for n in ("hand_L", "hand_R", "head", "hair", "eye_L", "eye_R", "nose", "heldtoy")})
        # All parts are at the origin when compiled, so MuJoCo marks them as sharing the body frame (geom_sameframe)
        # and later geom_pos changes are ignored (the first posed owner collapsed into one capsule) -> turn it off
        for gi in self.g.values():
            m.geom_sameframe[gi] = 0
        self.xy, self.yaw = np.array(at, float), float(yaw)
        self.pose, self.prev, self.w, self.blend_s, self.t = pose, pose, 1.0, 0.6, 0.0
        self.look = None                 # None = head faces the body's front; (x, y) = turn the head toward it (+-70 deg)
        self.queue, self.cur = [], None
        self.walking = False
        self.apply()
        orig = sim.step

        def step(*a):
            orig(*a)
            self.update(S_DT)
        sim.step = step

    # ── script ──
    def do(self, kind, **kw):
        self.queue.append((kind, kw))
        return self

    @property
    def busy(self):
        return self.cur is not None or bool(self.queue)

    def update(self, dt):
        self.t += dt
        if self.w < 1.0:
            self.w = min(1.0, self.w + dt / self.blend_s)
        if self.cur is None and self.queue:
            self._start(*self.queue.pop(0))
        if self.cur is not None:
            self._run(dt)
        self.apply()

    def _start(self, kind, kw):
        self.cur = [kind, kw, 0.0]
        if kind == "appear":
            self.xy, self.yaw = np.array(kw["at"], float), float(kw.get("yaw", self.yaw))
            self._set_pose(kw.get("pose", "stand"), 0.01)
        elif kind == "leave":
            self.xy = np.array(AWAY, float)
        elif kind == "pose":
            self._set_pose(kw["name"], kw.get("blend", 0.8))
        elif kind == "face":
            self.yaw = float(kw["yaw"])
        elif kind == "look":
            self.look = kw.get("at")
        elif kind == "walk":
            kw["path"] = [np.array(p, float) for p in kw["path"]]
            self._set_pose("walk", 0.3)
            self.walking = True

    def _run(self, dt):
        kind, kw, el = self.cur
        self.cur[2] = el + dt
        if kind == "walk":
            if not kw["path"]:
                self.walking = False
                self._set_pose(kw.get("then", "stand"), 0.4)
                self.cur = None
                return
            tgt = kw["path"][0]
            d = tgt - self.xy
            dist = float(np.linalg.norm(d))
            step = kw.get("speed", 0.6) * dt
            if dist <= step:
                self.xy = tgt
                kw["path"].pop(0)
            else:
                self.xy = self.xy + d / dist * step
                want = math.degrees(math.atan2(d[1], d[0]))
                err = (want - self.yaw + 180) % 360 - 180
                self.yaw += max(-240 * dt, min(240 * dt, err))
            return
        if kind == "wait" and el + dt < kw["secs"]:
            return
        if kind == "pose" and self.w < 1.0:
            return
        self.cur = None

    def _set_pose(self, name, blend):
        self.prev_pts = self._points()
        self._prev_name = self.pose
        self.pose, self.w, self.blend_s = name, 0.0, max(blend, 1e-3)

    def _points(self):
        cur = _pose(self.pose, self.t)
        if self.w >= 1.0 or not hasattr(self, "prev_pts"):
            return cur
        return _blend(self.prev_pts, cur, self.w)

    # ── place the joint points in MuJoCo ──
    def apply(self):
        m, d = self.sim.model, self.sim.data
        P = self._points()
        # "not home" is hidden under the floor: outside the walls the owner could still be seen over them
        away = self.xy[1] >= 3.0
        d.mocap_pos[self.mid] = [self.xy[0], self.xy[1], -10.0 if away else 0.0]
        h = math.radians(self.yaw) / 2
        d.mocap_quat[self.mid] = [math.cos(h), 0, 0, math.sin(h)]
        for name, a, b, r, _ in SEGS:
            pa, pb = np.array(P[a]), np.array(P[b])
            gi = self.g[name]
            m.geom_pos[gi] = (pa + pb) / 2
            m.geom_quat[gi] = _quat_z_to(pb - pa)
            hl = max(np.linalg.norm(pb - pa) / 2, 1e-3)
            m.geom_size[gi][:2] = [r, hl]
            m.geom_rbound[gi] = r + hl                       # broad-phase bounding sphere / box must follow the length, or no collision
            m.geom_aabb[gi] = [0, 0, 0, r, r, r + hl]
        for n in ("hand_L", "hand_R"):
            m.geom_pos[self.g[n]] = P[n]
        # head: turn toward the `look` target (+-70 deg relative to the body) + the pose's own head pitch
        hy, hp = P["_head"]
        head = np.array(P["head"])
        if self.look is not None:
            dx, dy = self.look[0] - self.xy[0], self.look[1] - self.xy[1]
            rel = (math.degrees(math.atan2(dy, dx)) - self.yaw + 180) % 360 - 180
            hy = max(-70, min(70, rel))
            if -60 < hp < 40:            # head hanging (sit_head_down, hug_knees) or lying face up: keep the pose's head
                # looking at a duck on the floor means looking down (turning only left / right left a standing
                # owner's gaze level, over the duck)
                hp = max(0.0, min(60.0, math.degrees(math.atan2(head[2] - 0.24, math.hypot(dx, dy)))))
        cy, sy, cp, sp = math.cos(math.radians(hy)), math.sin(math.radians(hy)), math.cos(math.radians(hp)), math.sin(math.radians(hp))
        R = np.array([[cy * cp, -sy, cy * sp], [sy * cp, cy, sy * sp], [-sp, 0, cp]])     # yaw(z) * pitch(y), pitch positive = down
        m.geom_pos[self.g["head"]] = head
        m.geom_pos[self.g["hair"]] = head + R @ [-0.03, 0, 0.05]
        m.geom_pos[self.g["eye_L"]] = head + R @ [0.088, 0.035, 0.025]
        m.geom_pos[self.g["eye_R"]] = head + R @ [0.088, -0.035, 0.025]
        m.geom_pos[self.g["nose"]] = head + R @ [0.10, 0, -0.01]
        self.head_dir = (self.yaw + hy, hp)
        # toy in the hands
        gt = self.g["heldtoy"]
        if P["_toy"]:
            hl, hr = np.array(P["hand_L"]), np.array(P["hand_R"])
            m.geom_pos[gt] = (hl + hr) / 2 + [0.06, 0, 0.03]
            m.geom_rgba[gt][3] = 1.0
        else:
            m.geom_pos[gt] = [0, 0, -1.0]
            m.geom_rgba[gt][3] = 0.0
        mujoco.mj_kinematics(m, d)

    # ── ground truth (for scoring) ──
    def truth(self, duck_xy=None) -> dict:
        """The owner's true visible state right now (same fields as perception.owner_state.OwnerState)."""
        present = self.xy[1] < 3.0
        if not present:
            return {"present": False, "posture": "unknown", "arms": "unknown", "head": "unknown", "facing_duck": "unknown"}
        pose = self.pose if self.w >= 0.5 else getattr(self, "_prev_name", self.pose)
        posture = "walking" if self.walking else "lying" if pose in LYING else "sitting" if pose in SITTING else "standing"
        arms = {"wave": "waving", "hips": "on_hips", "point": "pointing", "open_arms": "open", "hug_knees": "hugging_knees",
                "hold_toy": "holding_toy"}.get(pose, "resting")
        head = "down" if pose in ("sit_head_down", "hug_knees") else "up"
        facing = "unknown"
        if duck_xy is not None:
            # 3D: looking = the angle between the gaze (head yaw, pitch) and "eyes -> the duck's head (0.24 m up)" < 30 deg.
            # Comparing only horizontal directions and treating every lowered head as "not looking" was wrong: the duck is
            # on the floor, and an owner looking down and forward sees it.
            yaw, pit = math.radians(self.head_dir[0]), math.radians(self.head_dir[1])
            gaze = np.array([math.cos(yaw) * math.cos(pit), math.sin(yaw) * math.cos(pit), -math.sin(pit)])
            eye = self.sim.model.geom_pos[self.g["head"]].copy()
            c, s_ = math.cos(math.radians(self.yaw)), math.sin(math.radians(self.yaw))
            eye_w = np.array([self.xy[0] + c * eye[0] - s_ * eye[1], self.xy[1] + s_ * eye[0] + c * eye[1], eye[2]])
            to = np.array([duck_xy[0], duck_xy[1], 0.24]) - eye_w
            ang = math.degrees(math.acos(max(-1.0, min(1.0, float(gaze @ to / np.linalg.norm(to))))))
            facing = "yes" if ang < 30 else "no"
        return {"present": True, "posture": posture, "arms": arms, "head": head, "facing_duck": facing}


S_DT = 0.02          # same as body.sim.CONTROL_DT (not imported, to keep this prop independent of the simulator)
