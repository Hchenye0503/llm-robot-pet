"""The stage: the upstream apartment, a seated owner, a big toy, and render quality settings.

Everything is changed at load time through MjSpec; the upstream files are not modified.
    a seated owner (body, head, arms) instead of two legs: head at about 0.7 m
    ball_2 sat at the edge of the stairwell and rolled downstairs every time -> moved to the bathroom
    the red cube was too small to look like something worth exploring -> a 14 cm yellow toy
    anti-aliasing and high-resolution shadows
"""

import mujoco

from ..config import APARTMENT as SCENE  # noqa: F401  (scene path relative to MICRODUCK_RL_DIR)

OWNER = (-3.25, -2.0)                # where the owner sits (hips), facing +x
OWNER_HEAD = (-3.17, -2.0, 0.70)
OWNER_SPOT = (-2.70, -2.0)           # where the duck should go = in front of the owner's feet (feet reach x ~ -2.83)
TOY = (-2.15, -1.05)                 # big toy (middle of the living room, east of the coffee table)
START = (-1.55, -2.0, 180.0)         # duck start, facing the owner (west)
WALL_H = 1.8                         # wall height (m); upstream 0.5 m
SKIN, SHIRT, JEANS, SOCK = [0.93, 0.76, 0.62, 1], [0.86, 0.45, 0.30, 1], [0.20, 0.30, 0.55, 1], [0.95, 0.95, 0.95, 1]

CAM_SIDE = {"pos": [-1.10, -0.55, 1.05], "lookat": [-2.40, -2.05, 0.18]}   # observer: living room north-east corner
CAM_OWNER = {"pos": [OWNER_HEAD[0] + 0.10, OWNER_HEAD[1], OWNER_HEAD[2]], "lookat": "duck"}   # the owner's eyes


def _capsule(body, a, b, r, rgba):
    g = body.add_geom()
    g.type = mujoco.mjtGeom.mjGEOM_CAPSULE
    g.fromto = list(a) + list(b)
    g.size = [r, 0, 0]
    g.rgba = rgba
    return g


def _ball(body, p, r, rgba):
    g = body.add_geom()
    g.type = mujoco.mjtGeom.mjGEOM_SPHERE
    g.pos, g.size, g.rgba = list(p), [r, 0, 0], rgba


def stage(spec, owner: bool = True):
    """MjSpec edit for DuckSim(spec_edit=stage). owner=False -> no static owner (world/owner.py adds a moving one)."""
    spec.visual.quality.offsamples = 8          # anti-aliasing
    spec.visual.quality.shadowsize = 8192       # shadow resolution
    # Near clipping plane = znear x scene extent: the 14 m apartment with the default 0.01 skipped everything within
    # 14 cm, so the head camera saw through the owner's feet when close.
    spec.visual.map.znear = 0.001               # -> 1.4 cm
    # Taller walls: upstream walls are 0.5 m (a dollhouse), so a person standing outside showed above them and the
    # duck really saw an owner who had "left" -> 1.8 m (taller than the owner's 1.62 m, below the ~2 m follow camera)
    for g in spec.worldbody.geoms:
        if g.type == mujoco.mjtGeom.mjGEOM_BOX and g.name.startswith(("wall_", "wA_", "wB_", "wC_", "wF", "wG")):
            g.size = [g.size[0], g.size[1], WALL_H / 2]
            g.pos = [g.pos[0], g.pos[1], WALL_H / 2]
    # pink ball to the bathroom (it sat at the stairwell edge and rolled downstairs)
    for b in spec.bodies:
        if b.name == "ball_2":
            b.pos = [2.0, -2.2, 0]
    # owner sitting on the floor (coordinates relative to the hips, facing +x)
    if owner:
        o = spec.worldbody.add_body(name="owner", pos=[OWNER[0], OWNER[1], 0])
        for s in (-1, 1):
            _capsule(o, (0.00, s * 0.10, 0.10), (0.28, s * 0.13, 0.30), 0.065, JEANS)    # thigh: knees up
            _capsule(o, (0.28, s * 0.13, 0.30), (0.42, s * 0.13, 0.05), 0.055, JEANS)    # shin: down to the foot
            _capsule(o, (0.42, s * 0.13, 0.04), (0.52, s * 0.13, 0.04), 0.045, SOCK)     # foot
            _capsule(o, (0.02, s * 0.20, 0.50), (0.22, s * 0.16, 0.32), 0.040, SHIRT)    # arm: hands on the knees
            _ball(o, (0.24, s * 0.15, 0.33), 0.045, SKIN)                                 # hand
        _capsule(o, (0.00, 0.0, 0.14), (0.04, 0.0, 0.52), 0.14, SHIRT)                    # torso
        _capsule(o, (0.05, 0.0, 0.55), (0.06, 0.0, 0.60), 0.05, SKIN)                     # neck
        _ball(o, (OWNER_HEAD[0] - OWNER[0], 0.0, OWNER_HEAD[2]), 0.10, SKIN)              # head
        _ball(o, (OWNER_HEAD[0] - OWNER[0] - 0.03, 0.0, OWNER_HEAD[2] + 0.05), 0.095, [0.20, 0.14, 0.10, 1])  # hair
    # big toy: a 14 cm yellow cube (the upstream red cube is 7 cm)
    t = spec.worldbody.add_body(name="toy", pos=[TOY[0], TOY[1], 0.07])
    t.add_freejoint()
    g = t.add_geom()
    g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.size, g.rgba, g.mass = [0.07, 0.07, 0.07], [0.98, 0.80, 0.10, 1], 0.2


def world(sim) -> dict:
    """World information for behaviors, from ground truth (perception/vision_world.py replaces this with the camera)."""
    b = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "toy")
    return {"owner": OWNER_SPOT, "owner_head": OWNER_HEAD[:2], "toy": tuple(sim.data.xpos[b][:2])}
