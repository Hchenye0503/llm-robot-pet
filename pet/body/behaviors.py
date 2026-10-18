"""Behavior library: ethogram behaviors the LLM can pick by name, expanded into body motion.

    LLM: {"action": "behavior", "name": "greet"}          <- only the name (a whitelist), like gestures and policies
      -> this file: greet = walk to the owner -> face the owner -> look up -> happy -> wag
         the world information it needs (where the owner and the toy are) comes in as `world`:
         ground truth, or the camera (perception/vision_world.py) -- only that layer changes
      -> building blocks: nav/route.goto_planned (walk to a target, around furniture), scheduler.run_steps
         (sit / stand, gestures), keyframes

Gesture vs. behavior: a gesture moves only the head in the current posture; a behavior moves the body and is
directed at something (the owner, a toy), which is what most ethogram behaviors look like.
Names and descriptions are in planner/catalog.py (ids are rows of docs/ethogram.md).
Imports stay inside functions: the scheduler imports this file and this file uses the scheduler.
"""

import math

from ..planner.catalog import BEHAVIORS, BEH_ENDS, end_posture  # noqa: F401  (re-exported for callers)


def _idle(sim):
    from . import sim as S
    from .scheduler import sense_posture
    return ("sitstand", [1.0] + [0.0] * 12) if sense_posture(sim) == "sitting" else ("walk", S.zeros())


def bearing_to(sim, target) -> float:
    """Direction of the target from the robot (deg, positive = left)."""
    from .goto import to_robot_frame
    rx, ry = to_robot_frame(sim, *target)
    return math.degrees(math.atan2(ry, rx))


def face(sim, target, tol=5.0, timeout_s=6.0):
    """Turn in place until facing target (error < tol)."""
    from . import sim as S
    from .goto import TURN_RATE
    t0 = sim.t
    while sim.t - t0 < timeout_s:
        th = bearing_to(sim, target)
        if abs(th) < tol:
            break
        c = S.zeros()
        c[2] = math.radians(math.copysign(TURN_RATE, th))
        sim.step("walk", c)
    for _ in range(15):
        sim.step("walk", S.zeros())


def head_hold(sim, yaw, pitch, secs, ramp=0.4, back=True):
    """In the current posture, move the head smoothly to (yaw, pitch) degrees and hold it for secs (keyframes)."""
    from .keyframes import interp
    policy, base = _idle(sim)
    keys_y = [(0, 0), (ramp, yaw), (ramp + secs, yaw)] + ([(2 * ramp + secs, 0)] if back else [])
    keys_p = [(0, 0), (ramp, pitch), (ramp + secs, pitch)] + ([(2 * ramp + secs, 0)] if back else [])
    total, t0 = keys_y[-1][0], sim.t
    while sim.t - t0 < total:
        t = sim.t - t0
        c = list(base)
        c[5], c[4] = math.radians(interp(keys_y, t)), math.radians(interp(keys_p, t))
        sim.step(policy, c)


def walk_for(sim, vx, wz_deg, secs, head=(0.0, 0.0)):
    from . import sim as S
    c = S.zeros()
    c[0], c[2] = vx, math.radians(wz_deg)
    c[5], c[4] = math.radians(head[0]), math.radians(head[1])
    for _ in range(int(round(secs / S.CONTROL_DT))):
        if vx > 0 and sim.touching("owner"):                 # walking forward into the owner -> stop
            break
        sim.step("walk", c)


def _steps(sim, steps, log):
    from .scheduler import run_steps
    run_steps(sim, steps, log)


def stand_look(sim, secs, up=60.0, body=15.0, yaw=0.0, roll=0.0, ramp=0.5, at_peak=None):
    """Lean back and look up at the owner while standing (official alpha_stand): body pitched back `body` deg plus
    head up `up` deg, held for secs, then back to walking.

    Measured with the pet gait: walk -> look up -> walk 10/10 without falling, camera 66-71 deg up, drift < 2 cm.
    Looking up while standing reads as wanting to interact; quiet company still sits and looks up.
    at_peak: called once as at_peak(sim) halfway through the hold (used to take a photo at that moment).
    13-number command: head_pitch c[4] (negative = up), head_yaw c[5], head_roll c[6], body pitch c[11] (positive = back)."""
    from . import sim as S
    from .keyframes import interp
    total = 2 * ramp + secs
    t0 = sim.t
    while sim.t - t0 < total:
        t = sim.t - t0
        if at_peak and t >= ramp + secs / 2:
            at_peak(sim)
            at_peak = None
        k = interp([(0, 0.0), (ramp, 1.0), (ramp + secs, 1.0), (total, 0.0)], t)
        c = S.zeros()
        c[4], c[5], c[6], c[11] = math.radians(-up * k), math.radians(yaw * k), math.radians(roll * k), math.radians(body * k)
        sim.step("stand", c)
    for _ in range(15):
        sim.step("walk", S.zeros())                                       # back to walking (measured: 0 falls)


def head_keys(sim, keys_yaw, keys_pitch, total, keys_roll=((0, 0),)):
    """Play head keyframes (deg) in the current posture. roll = head tilt (trained range +-17 deg)."""
    from .keyframes import interp
    policy, base = _idle(sim)
    t0 = sim.t
    while sim.t - t0 < total:
        t = sim.t - t0
        c = list(base)
        c[5], c[4] = math.radians(interp(keys_yaw, t)), math.radians(interp(keys_pitch, t))
        c[6] = math.radians(interp(list(keys_roll), t))
        sim.step(policy, c)


def _sway(amp, hz, secs, t0=0.0):
    """Side-to-side keyframes (start at 0, end at 0)."""
    keys, t, half, side = [(t0, 0.0)], t0 + 0.25 / hz, 0.5 / hz, 1
    while t < t0 + secs:
        keys.append((t, side * amp))
        t, side = t + half, -side
    keys.append((t - half / 2, 0.0))
    return keys


def run_behavior(sim, name: str, world: dict, log: list, nxt=None):
    """Run one behavior. world = {"owner": spot in front of the owner's feet, "owner_head": the owner's head,
    "toy": the toy} (world x, y). nxt (the next plan step) is passed for behaviors whose ending could depend on it;
    none uses it now."""
    from ..nav.route import goto_planned as goto            # follows A* waypoints when furniture blocks the straight line
    from ..planner.schema import Gesture, SitStand
    G = lambda n: Gesture(action="gesture", name=n)
    SIT = SitStand(action="sitstand", posture="sit")
    spot, head, toy = world.get("owner"), world.get("owner_head", world.get("owner")), world.get("toy")
    need = {"greet": spot, "stay_close": spot, "gaze": head, "play_bow": head, "guilty": spot, "investigate": toy}
    if name in need and need[name] is None:
        log.append(f"behavior {name}: the target it needs is not in world -> skipped")
        return

    if name == "greet":
        # trot over with the head up; stop at 0.30 m (0.15 was the trunk center, so the head already hit the feet)
        r = goto(sim, spot, stop_dist=0.30, walk_v=0.4, head=(0, -15), goal_body="owner")
        log.append(f"greet: trotted to the owner's feet ({r['secs']:.1f} s)" + (f", stopped on touching {r['bumped']}" if r.get("bumped") else ""))
        face(sim, head)
        stand_look(sim, 1.2)                                              # look up at the owner first (wants to interact)
        _steps(sim, [G("happy"), G("wag")], log)
    elif name == "stay_close":
        goto(sim, spot, stop_dist=0.25, goal_body="owner")
        face(sim, head)
        _steps(sim, [SIT], log)
        head_hold(sim, 0, -25, 2.5)                                          # sit and look up at the owner
    elif name == "gaze":
        from .scheduler import sense_posture
        yaw = max(-80.0, min(80.0, bearing_to(sim, head)))
        tilt = [(0, 0), (0.6, 0), (1.0, 14), (2.2, 14), (2.8, 0)]          # a slight head tilt
        if sense_posture(sim) == "sitting":
            # sitting: turn only the head, not the body (turning in place with walk while sitting fell over)
            head_keys(sim, [(0, 0), (0.6, yaw), (2.6, yaw), (3.0, 0)],
                      [(0, 0), (0.6, -20), (2.6, -20), (3.0, 0)], 3.0, keys_roll=tilt)
        else:
            head_hold(sim, yaw, -10, 0.8, back=False)                        # 1. turn the head first
            face(sim, head)                                                  # 2. the body follows
            stand_look(sim, 1.8, roll=14)                                     # 3. lean back, look up, tilt the head (walk alone reaches only +29 deg)
        log.append(f"gaze: owner was at {yaw:+.0f} deg")
    elif name == "play_bow":
        # Approach first, so the owner feels the invitation is for them. Walking until touching did not work for a
        # standing owner (feet 20 cm apart: the duck hit a foot, got turned 90 deg, and the lunge was blocked) ->
        # come to 0.25 m (almost touching, room to move); the stop-on-touch guard is still there.
        from .goto import to_robot_frame
        target = spot or head
        if math.hypot(*to_robot_frame(sim, *target)) > 0.35:
            r = goto(sim, target, stop_dist=0.25, head=(0, -15), goal_body="owner")
            log.append("play_bow: came up to the owner" + (f" (stopped on touching {r['bumped']})" if r.get("bumped") else ""))
        face(sim, head)
        head_keys(sim, [(0, 0)], [(0, 0), (0.3, 45), (1.1, 45), (1.3, -20), (1.6, -20)], 1.6)   # deep bow -> spring up, head high
        # The pet gait does not walk backward, so the forward-and-back bounce became one short lunge (no stepping
        # back, or each lunge would end closer to the owner) plus the head bouncing twice.
        walk_for(sim, 0.25, 0, 0.3, head=(0, -15))                            # one small lunge forward
        head_keys(sim, [(0, 0)], [(0, 0), (0.2, -25), (0.4, 10), (0.6, -25), (0.8, 10), (1.0, 0)], 1.0)   # head bounces twice
        _steps(sim, [G("wag")], log)
        head_keys(sim, [(0, 0)], [(0, 0), (0.25, 45), (0.9, 45), (1.2, 0)], 1.2)               # bow again
    elif name == "guilty":
        face(sim, head)
        goto(sim, spot, stop_dist=0.30, walk_v=0.25, head=(0, 30), goal_body="owner")          # approach with the head low (0.2 m/s looked deliberate)
        _steps(sim, [SitStand(action="sitstand", posture="sit", head={"yaw_left_deg": 0, "pitch_deg": 30})], log)
        sway = _sway(20, 0.4, 3.0)                                           # sit, head low, slowly swaying left and right
        head_keys(sim, sway, [(0, 30), (sway[-1][0], 30), (sway[-1][0] + 0.4, 0)], sway[-1][0] + 0.4)
    elif name == "investigate" and world.get("toy_held"):
        # the toy is in the owner's hands: walk up to the owner and look up at it with a head tilt
        # (sniffing the floor there would mean looking at the owner's feet)
        r = goto(sim, toy, stop_dist=0.30, head=(0, -15), goal_body="owner")
        face(sim, head)
        stand_look(sim, 1.5, roll=14)
        _steps(sim, [G("head_tilt")], log)
        log.append(f"investigate: the toy is in the owner's hands -> walked to the owner ({r['secs']:.1f} s) and looked up at it")
    elif name == "investigate":
        # stop distance 0.22: 0.35 was tuned while distances were computed ~15 cm too far (a fixed 9.6 deg camera
        # tilt); with the measured tilt the error is gone, so the stop distance shrank instead of the pass threshold
        # growing (0.35 - 0.15 = 0.20, plus ~4 cm more braking slide with the pet gait)
        r = goto(sim, toy, stop_dist=0.22, goal_body="toy")
        face(sim, toy)
        head_keys(sim, [(0, 0)], [(0, 0), (0.4, 35), (0.7, 25), (1.0, 38), (1.3, 25), (1.6, 38), (2.0, 0)], 2.0)  # sniff
        _steps(sim, [G("look_around"), G("head_tilt")], log)
        log.append(f"investigate: walked to {r['dist']:.2f} m from the toy")
    elif name == "rest":
        _steps(sim, [SIT], log)
        head_hold(sim, 0, 25, 3.0, ramp=1.0)                                 # head slowly lowers
    elif name == "zoomies":
        walk_for(sim, 0.4, 0, 0.8)                                           # dash
        walk_for(sim, 0, 0, 0.3)                                             # sudden stop
        walk_for(sim, 0, 57, 1.0)                                            # sharp turn (left)
        walk_for(sim, 0.4, 0, 0.6)
        walk_for(sim, 0, -57, 0.5)                                           # sharp turn (was a backward dash)
        walk_for(sim, 0, -57, 1.2)                                           # sharp turn (right)
        walk_for(sim, 0.4, 0, 0.6)
        _steps(sim, [G("spin_hop"), SIT], log)                               # spin hop -> sudden sit
    else:
        log.append(f"behavior {name}: unknown -> skipped")
