"""Walk to a target: a closed-loop controller (the "legs" of the behavior library).

    1. where is the target?   ground truth in simulation, or the camera (perception/vision_world.py)
           -> target position in the robot frame (forward x, left y)
    2. control (this file): every 20 ms choose "turn in place" or "walk forward" until close enough
           -> walk command
    3. behaviors: greet = walk to the owner -> face the owner -> happy -> wag ...

Why not "compute the angle, turn once, walk straight" (open loop): walking drifts 8-16 deg in 2 s and turns
overshoot by 3-15%. Open loop never arrives; re-measuring where the target is every 20 ms and correcting does.

Gait limits the controller respects (measured on the walking policy this was written for):
    turning dead zone: at least 43 deg/s to turn at all -> no gentle corrections, only "turn" or "don't turn"
    walking and turning together barely moves forward -> turn and walk are done separately
    -> two outputs only: turn in place (vx = 0, wz = +-TURN_RATE) or walk straight (vx = WALK_V, wz = 0)

    python -m pet.body.goto --unit      # controller only, no simulator
    python -m pet.body.goto             # walk to 6 targets in the simulator
"""

import argparse
import math

TURN_TOL = 15.0     # heading error above this -> stop and turn (deg)
TURN_RATE = 50.0    # turning rate (deg/s; must be >= 43, the dead zone)
WALK_V = 0.3        # walking speed (m/s)


def goto_command(rel_x: float, rel_y: float, stop_dist: float) -> tuple[float, float, bool]:
    """Target at (rel_x forward, rel_y left) in meters -> (vx m/s, wz deg/s positive = left, arrived).
    Arrived -> stop; heading error > TURN_TOL -> turn in place toward the target; otherwise walk straight."""
    d = math.hypot(rel_x, rel_y)
    theta = math.degrees(math.atan2(rel_y, rel_x))
    if d <= stop_dist:
        return 0.0, 0.0, True
    if abs(theta) > TURN_TOL:
        return 0.0, math.copysign(TURN_RATE, theta), False
    return WALK_V, 0.0, False


# Hysteresis version. With one 15 deg threshold, a 2 m target switched stop / turn / walk 17 times: walking
# drifts, and every drift past 15 deg stopped the duck. Two thresholds, the same trick as idle.py:
#     walking -> stop to turn only when the error exceeds TURN_START (tolerates walking drift)
#     turning -> start walking only when the error is below TURN_STOP (turn precisely, once)
# It needs to remember whether it is turning or walking -> one more input / output, `turning`.
TURN_START = 25.0
TURN_STOP = 5.0


def goto_command_hyst(rel_x, rel_y, stop_dist, turning):
    d = math.hypot(rel_x, rel_y)
    theta = math.degrees(math.atan2(rel_y, rel_x))
    if d <= stop_dist:
        return 0.0, 0.0, True, False
    if turning:
        turning = abs(theta) > TURN_STOP          # keep turning until precise
    else:
        turning = abs(theta) > TURN_START         # start turning only when far off
    if turning:
        return 0.0, math.copysign(TURN_RATE, theta), False, True
    return WALK_V, 0.0, False, False


# ─────────────────────────────────────────────────────────────
# unit test (no simulator)
# ─────────────────────────────────────────────────────────────
UNIT = [
    # (description, x, y, stop, expected (vx, wz, done))
    ("1 m straight ahead -> walk", 1.0, 0.0, 0.4, (WALK_V, 0.0, False)),
    ("60 deg to the front left -> turn left in place", 0.5, 0.866, 0.4, (0.0, TURN_RATE, False)),
    ("60 deg to the front right -> turn right in place", 0.5, -0.866, 0.4, (0.0, -TURN_RATE, False)),
    ("straight behind -> turn (either side), not forward", -1.0, 0.0, 0.4, None),
    ("10 deg to the front left (within tolerance) -> walk", 1.0, 0.176, 0.4, (WALK_V, 0.0, False)),
    ("0.3 m left -> arrived", 0.3, 0.0, 0.4, (0.0, 0.0, True)),
    ("behind on the left but very close -> also arrived", -0.2, 0.2, 0.4, (0.0, 0.0, True)),
]


def unit() -> int:
    bad = 0
    for desc, x, y, stop, expect in UNIT:
        got = goto_command(x, y, stop)
        if expect is None:
            ok = got[0] == 0 and abs(got[1]) >= 43 and not got[2]
        else:
            ok = all(math.isclose(a, b, abs_tol=1e-9) for a, b in zip(got[:2], expect[:2])) and got[2] == expect[2]
        bad += not ok
        print(f"{'ok ' if ok else 'BAD'} {desc:<52} got {got}  want {expect if expect else 'no forward, |turn| >= 43'}")
    print(f"\nunit: {len(UNIT) - bad}/{len(UNIT)}")
    return bad


# ─────────────────────────────────────────────────────────────
# in the simulator: every 20 ms, target relative to the robot -> goto_command -> walk
# ─────────────────────────────────────────────────────────────
def to_robot_frame(sim, tx, ty):
    x, y, yaw = sim.xy_yaw()
    c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    dx, dy = tx - x, ty - y
    return c * dx + s * dy, -s * dx + c * dy


def _walk(sim, vx, wz_deg, secs):
    from . import sim as S
    c = S.zeros()
    c[0], c[2] = vx, math.radians(wz_deg)
    for _ in range(int(round(secs / S.CONTROL_DT))):
        sim.step("walk", c)


def _turn_by(sim, deg):
    """Turn in place by deg (closed loop on the IMU heading; stops TURN_LEAD early, the same number as vision_world)."""
    from . import sim as S
    from ..perception.vision_world import TURN_LEAD
    y0, t0 = sim.xy_yaw()[2], sim.t
    while sim.t - t0 < 4.0 and abs((sim.xy_yaw()[2] - y0 + 180) % 360 - 180) < abs(deg) - TURN_LEAD:
        _walk(sim, 0.0, math.copysign(43, deg), S.CONTROL_DT)


def _detour_side(sim, hit) -> int:
    """Go around the other side of the obstacle: obstacle on the left -> turn right (-1), on the right -> left (+1).
    Unknown position -> left."""
    import mujoco
    m, d = sim.model, sim.data
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, hit)
    if b > 0:
        xy = d.xpos[b][:2]
    else:
        g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, hit)
        if g < 0:
            return 1
        xy = d.geom_xpos[g][:2]
    _, ry = to_robot_frame(sim, *xy)
    return -1 if ry > 0 else 1


def goto(sim, target, stop_dist=0.4, timeout_s=20.0, hysteresis=True, walk_v=None, head=(0.0, 0.0),
         bump=("owner", "toy"), goal_body=None, max_detours=2):
    """Walk near `target` (world coordinates). Returns a summary: arrived, seconds, mode switches, final distance.
    hysteresis=True uses the two-threshold controller; walk_v temporarily changes the speed (e.g. trotting to greet)."""
    from . import sim as S
    global WALK_V
    saved_v, WALK_V = WALK_V, (walk_v or WALK_V)
    t0, mode, switches, turning, hit, detours = sim.t, None, 0, False, None, 0
    while sim.t - t0 < timeout_s:
        rx, ry = to_robot_frame(sim, *target)
        if hysteresis:
            vx, wz, done, turning = goto_command_hyst(rx, ry, stop_dist, turning)
        else:
            vx, wz, done = goto_command(rx, ry, stop_dist)
        if done:
            break
        hit = sim.touching_what()
        if hit and (hit == goal_body or (goal_body is None and hit in bump)):
            break                                         # touched the target -> stop (hardware: bumper / current spike)
        if hit and detours < max_detours:
            # Walking to the owner, the duck bumped into the toy and stood still 1.1 m away. Something that is not the
            # target -> turn 70 deg away from it, walk 0.25 m, continue (the simplest bug algorithm). The pet gait
            # cannot really walk backward (0.07 m/s at most; and pets rarely back up, they turn), so no step back.
            detours += 1
            side = _detour_side(sim, hit)
            _turn_by(sim, 70 * side)
            _walk(sim, 0.3, 0.0, 0.85)
            turning = False
            continue
        if hit:
            break                                         # still bumping after max_detours -> give up (the caller logs it)
        new_mode = "walk" if vx else "turn"
        switches += mode is not None and new_mode != mode
        mode = new_mode
        c = S.zeros()
        c[0], c[2] = vx, math.radians(wz)
        c[5], c[4] = math.radians(head[0]), math.radians(head[1])   # head while walking (deg): (yaw left positive, pitch down positive)
        sim.step("walk", c)
        if sim.upright() < 0.5:
            WALK_V = saved_v
            return {"reached": False, "fell": True, "secs": sim.t - t0, "switches": switches, "dist": math.hypot(rx, ry)}
    WALK_V = saved_v
    for _ in range(25):                                   # stop and settle 0.5 s (slides forward a bit)
        sim.step("walk", S.zeros())
    rx, ry = to_robot_frame(sim, *target)
    d = math.hypot(rx, ry)
    return {"reached": d <= stop_dist + 0.1, "fell": False, "secs": sim.t - t0, "switches": switches, "dist": d, "bumped": hit,
            "detours": detours, "bearing": math.degrees(math.atan2(ry, rx))}


TARGETS = [
    # (description, distance m, bearing deg relative to the start heading, positive = left)
    ("1 m ahead", 1.0, 0),
    ("1 m, 45 deg front left", 1.0, 45),
    ("1 m, 90 deg right", 1.0, -90),
    ("1 m straight behind", 1.0, 180),
    ("0.7 m, 120 deg back left", 0.7, 120),
    ("2 m, 30 deg front right", 2.0, -30),
]


def sim_test():
    from . import sim as S
    sim = S.DuckSim(f"{S.config.SCENES}/scene_allcollisions.xml")
    print(f"TURN_RATE={TURN_RATE} deg/s, WALK_V={WALK_V} m/s; arrived = within 0.4 m (0.5 m allowed after sliding)")
    for label, hyst in ((f"single threshold {TURN_TOL:g} deg", False), (f"hysteresis {TURN_START:g} / {TURN_STOP:g} deg", True)):
        print(f"\n-- {label} --")
        for desc, dist, ang in TARGETS:
            sim.reset()
            for _ in range(50):
                sim.step("walk", S.zeros())
            x, y, yaw = sim.xy_yaw()
            a = math.radians(yaw + ang)
            r = goto(sim, (x + dist * math.cos(a), y + dist * math.sin(a)), hysteresis=hyst)
            mark = "FELL" if r["fell"] else ("ok  " if r["reached"] else "BAD ")
            print(f"{mark} {desc:<26} {r['secs']:5.1f} s, turn/walk switches {r['switches']:>3}, final distance {r['dist']:.2f} m"
                  + (f", target at {r['bearing']:+.0f} deg" if "bearing" in r else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit", action="store_true")
    args = ap.parse_args()
    if args.unit:
        raise SystemExit(unit())
    if unit():
        raise SystemExit("unit test failed: fix it before running the simulator")
    print()
    sim_test()
