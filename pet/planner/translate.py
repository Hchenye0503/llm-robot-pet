"""Translation: the LLM's form -> each policy's 13 command numbers.

The 13 numbers (the last 13 dims of the policy observation, in the env cfg's "order matters" order):

    index  0   1   2  |     3           4          5          6      |  7  8  9   10    11     12
          vx  vy  wz  | neck_pitch  head_pitch  head_yaw  head_roll  |  x  y  z  roll pitch  yaw
          └ twist ┘     └──────── head_pose (delta from home, rad) ─┘   └──── body_pose ────┘

Two rules:
    1. every step starts from 13 zeros, so the LLM can never leave a stale value behind
    2. the policy defines what a slot means: for walk, vx is a speed; for sitstand, it is the posture flag

    python -m pet.planner.translate
"""

import math

from .limits import RangeReject, apply_limit, snap_deadzone
from .schema import DEADZONE, LIMITS, Behavior, Gesture, Head, Kick, Plan, Refuse, SitStand, Walk

N_CMD = 13


def head_slots(h: Head, log: list[str]) -> list[float]:
    """Head form -> slots 3-6 (rad). Pitch goes to head_pitch; neck_pitch stays 0 (it barely tracks commands)."""
    yaw = apply_limit("yaw_left_deg", h.yaw_left_deg, LIMITS["yaw_left_deg"], log)
    pitch = apply_limit("pitch_deg", h.pitch_deg, LIMITS["pitch_deg"], log)
    #       neck  head_pitch           head_yaw            head_roll
    return [0.0, math.radians(pitch), math.radians(yaw), 0.0]


def walk_to_cmd(step: Walk, log: list[str]) -> tuple[str, list[float]]:
    cmd = [0.0] * N_CMD
    # upper limit and NaN first, then the dead zone (lower limit)
    vx = apply_limit("forward_mps", step.forward_mps, LIMITS["forward_mps"], log)
    cmd[0] = snap_deadzone("forward_mps", vx, *DEADZONE["forward_mps"], log)
    wz = apply_limit("turn_left_deg_s", step.turn_left_deg_s, LIMITS["turn_left_deg_s"], log)
    cmd[2] = math.radians(snap_deadzone("turn_left_deg_s", wz, *DEADZONE["turn_left_deg_s"], log))
    cmd[3:7] = head_slots(step.head, log)
    return "walk", cmd


def sitstand_to_cmd(step: SitStand, log: list[str]) -> tuple[str, list[float]]:
    cmd = [0.0] * N_CMD
    cmd[0] = 1.0 if step.posture == "sit" else 0.0      # posture flag
    cmd[3:7] = head_slots(step.head, log)
    return "sitstand", cmd


def kick_to_cmd(step: Kick, log: list[str]) -> tuple[str, list[float]]:
    """The kick policy was trained with all-zero commands. The target is for the scheduler, which aims first."""
    return "kick", [0.0] * N_CMD


def placeholder_cmd(step, log: list[str]) -> tuple[str, list[float]]:
    """Gestures and behaviors change every 20 ms and depend on the current posture, so the scheduler
    computes them live. This placeholder keeps the whole plan going through one translation path."""
    return "gesture", [0.0] * N_CMD


TRANSLATORS = {Walk: walk_to_cmd, SitStand: sitstand_to_cmd, Kick: kick_to_cmd,
               Gesture: placeholder_cmd, Behavior: placeholder_cmd}


def translate_plan(plan: Plan) -> tuple[list[tuple[str, list[float]]], list[str]]:
    """Translate the whole plan. Any RangeReject propagates, so the whole plan is rejected:
    everything is translated before anything is handed over, and a bad step 2 also stops step 1."""
    log: list[str] = []
    out = [TRANSLATORS[type(step)](step, log) for step in plan.steps if not isinstance(step, Refuse)]
    log += [f"LLM refused: {step.reason}" for step in plan.steps if isinstance(step, Refuse)]
    return out, log


# ─────────────────────────────────────────────────────────────
# selftest
# ─────────────────────────────────────────────────────────────
R = "reject"
r30 = math.radians(30)
Z = [0.0] * 13
CASES = [
    # (description, LLM output, expected [(policy, 13 numbers), ...] or R, expected log lines)
    ("walk forward 0.2", [{"action": "walk", "forward_mps": 0.2, "duration_s": 2}],
     [("walk", [0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])], 0),
    ("turn left 50 deg/s (deg -> rad)", [{"action": "walk", "turn_left_deg_s": 50, "duration_s": 2}],
     [("walk", [0, 0, math.radians(50), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])], 0),
    ("turn left 20 deg/s -> dead zone -> 29", [{"action": "walk", "turn_left_deg_s": 20, "duration_s": 2}],
     [("walk", [0, 0, math.radians(29), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])], 1),
    ("slow walk 0.1 moves on this gait -> unchanged", [{"action": "walk", "forward_mps": 0.1, "duration_s": 2}],
     [("walk", [0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])], 0),
    ("walk while looking left 30", [{"action": "walk", "forward_mps": 0.25, "head": {"yaw_left_deg": 30}, "duration_s": 2}],
     [("walk", [0.25, 0, 0, 0, 0, r30, 0, 0, 0, 0, 0, 0, 0])], 0),
    ("full sprint -> clamped", [{"action": "walk", "forward_mps": 1.0, "duration_s": 2}],
     [("walk", [0.4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])], 1),
    ("turn 100 deg/s -> clamped to 57", [{"action": "walk", "turn_left_deg_s": 100, "duration_s": 2}],
     [("walk", [0, 0, math.radians(57), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])], 1),
    ("centimeters as meters -> reject", [{"action": "walk", "forward_mps": 30, "duration_s": 2}], R, 0),
    ("speed is NaN -> reject", [{"action": "walk", "forward_mps": float("nan"), "duration_s": 2}], R, 0),
    ("sit", [{"action": "sitstand", "posture": "sit"}],
     [("sitstand", [1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])], 0),
    ("stand looking right 30", [{"action": "sitstand", "posture": "stand", "head": {"yaw_left_deg": -30}}],
     [("sitstand", [0.0, 0, 0, 0, 0, -r30, 0, 0, 0, 0, 0, 0, 0])], 0),
    ("sitting, head 1833 deg -> reject", [{"action": "sitstand", "posture": "sit", "head": {"yaw_left_deg": 1833}}], R, 0),
    ("walk then sit (both translated)",
     [{"action": "walk", "forward_mps": 0.2, "duration_s": 2}, {"action": "sitstand", "posture": "sit"}],
     [("walk", [0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), ("sitstand", [1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])], 0),
    ("step 1 fine, step 2 bad -> whole plan rejected",
     [{"action": "walk", "forward_mps": 0.2, "duration_s": 2},
      {"action": "sitstand", "posture": "sit", "head": {"pitch_deg": float("inf")}}], R, 0),
    ("kick to owner (all zeros)", [{"action": "kick", "target": "owner"}], [("kick", Z)], 0),
]


def _same(got, expect) -> bool:
    if expect == R or got == R:
        return got == expect
    if len(got) != len(expect):
        return False
    for (gp, gc), (ep, ec) in zip(got, expect):
        if gp != ep or len(gc) != N_CMD:
            return False
        if not all(math.isclose(a, b, abs_tol=1e-9) for a, b in zip(gc, ec)):
            return False
    return True


def _fmt(cmds) -> str:
    return " | ".join(f"{p} [{', '.join(f'{v:.3g}' for v in c)}]" for p, c in cmds)


def selftest() -> int:
    n_bad = 0
    for desc, steps, expect, n_log in CASES:
        try:
            got, log = translate_plan(Plan.model_validate({"steps": steps}))
            why = "; ".join(log)
        except RangeReject as e:
            got, log, why = R, [], str(e)
        ok = _same(got, expect) and len(log) == n_log
        n_bad += not ok
        print(f"{'ok ' if ok else 'BAD'} {desc}\n     got:  {got if got == R else _fmt(got)}  {why}")
        if not ok and expect != R:
            print(f"     want: {_fmt(expect)} (log {n_log} lines, got {len(log)})")
    print(f"\n{len(CASES) - n_bad}/{len(CASES)} as expected")
    return n_bad


if __name__ == "__main__":
    raise SystemExit(selftest())
