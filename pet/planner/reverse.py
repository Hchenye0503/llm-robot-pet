"""Walking backward = turn around, walk forward, turn back.

The pet gait cannot walk backward (max -0.07 m/s). The first fix asked the LLM to expand "back up"
itself (turn 180°, walk, never write a negative speed), but in longer sentences it dropped the
backward part entirely: "turn right 90 degrees, then back up for two seconds, then sit down" became
turn -> walk forward -> sit. Now the LLM writes the literal meaning (forward_mps < 0) and this
program expands it, the same split as behaviors: the LLM says what, the program decides how.
Turning back at the end leaves the duck facing its original direction, so it really did back up.

Runs right after the structure check: the expanded steps then go through preconditions (a sitting
duck stands first) and the range check. The expansion only flips the sign, so a unit error such as
-3.0 becomes 3.0 and is still rejected.

    python -m pet.planner.reverse
"""

from .schema import Gesture, SitStand, Walk

TURN_DEG_S, TURN_S = 45.0, 4.0      # 45 deg/s x 4 s = 180 deg


def turn_around() -> Walk:
    """Turn 180 degrees in place (to the left)."""
    return Walk(action="walk", forward_mps=0.0, turn_left_deg_s=TURN_DEG_S, duration_s=TURN_S)


def expand_backward(steps: list) -> list:
    """Replace every backward walk with the turn-around version; other steps are kept as they are.
    Returns a new list (the input is not modified).

      1. backward walk          -> turn 180 -> walk forward (same speed, duration, head) -> turn 180
      2. backward while turning -> first the turn alone (same rate and duration), then rule 1
      3. anything else, including forward-and-turning -> unchanged
    """
    expanded = []
    for s in steps:
        if isinstance(s, Walk):
            if s.turn_left_deg_s != 0:
                if s.forward_mps < 0:
                    expanded.append(s.model_copy(update={"forward_mps": 0.0}))
                    s = s.model_copy(update={"turn_left_deg_s": 0.0})
                else:
                    expanded.append(s)
                    continue
            if s.forward_mps < 0:
                expanded.append(turn_around())
                expanded.append(s.model_copy(update={"forward_mps": abs(s.forward_mps)}))
                expanded.append(turn_around())
            else:
                expanded.append(s)
        else:
            expanded.append(s)
    return expanded


# ─────────────────────────────────────────────────────────────
# selftest (no network, no simulator)
# ─────────────────────────────────────────────────────────────
def _w(v, turn=0.0, secs=2.0, yaw=0.0):
    return Walk(action="walk", forward_mps=v, turn_left_deg_s=turn, duration_s=secs, head={"yaw_left_deg": yaw})


def _short(s) -> str:
    if isinstance(s, Walk):
        return f"walk({s.forward_mps:g}, {s.turn_left_deg_s:g}°/s, {s.duration_s:g}s" + \
               (f", head {s.head.yaw_left_deg:g}°)" if s.head.yaw_left_deg else ")")
    if isinstance(s, SitStand):
        return s.posture
    return s.name


T180 = "walk(0, 45°/s, 4s)"
SIT = SitStand(action="sitstand", posture="sit")
NOD = Gesture(action="gesture", name="nod")
CASES = [
    ("walk backward 2 s", [_w(-0.3)], [T180, "walk(0.3, 0°/s, 2s)", T180]),
    ("walk forward: unchanged", [_w(0.3)], ["walk(0.3, 0°/s, 2s)"]),
    ("turn in place: unchanged", [_w(0.0, 45)], ["walk(0, 45°/s, 2s)"]),
    ("turn right 90, back up 2 s, sit", [_w(0.0, -45), _w(-0.3), SIT],
     ["walk(0, -45°/s, 2s)", T180, "walk(0.3, 0°/s, 2s)", T180, "sit"]),
    ("tiny backward is still expanded (dead zone is translate's job)", [_w(-0.05, secs=1.0)],
     [T180, "walk(0.05, 0°/s, 1s)", T180]),
    ("backward while turning -> turn first, then back up", [_w(-0.3, 30)],
     ["walk(0, 30°/s, 2s)", T180, "walk(0.3, 0°/s, 2s)", T180]),
    ("head stays on the walking step", [_w(-0.2, yaw=40)], [T180, "walk(0.2, 0°/s, 2s, head 40°)", T180]),
    ("forward while turning is not backward -> unchanged", [_w(0.3, 30)], ["walk(0.3, 30°/s, 2s)"]),
    ("other steps around it", [NOD, _w(-0.3), _w(0.3)], ["nod", T180, "walk(0.3, 0°/s, 2s)", T180, "walk(0.3, 0°/s, 2s)"]),
]


def selftest() -> int:
    bad = 0
    for desc, steps, want in CASES:
        before = [_short(s) for s in steps]
        got = [_short(s) for s in expand_backward(steps)]
        untouched = [_short(s) for s in steps] == before
        ok = got == want and untouched
        bad += not ok
        print(f"{'ok ' if ok else 'BAD'} {desc}\n     got:  {got}")
        if not ok:
            print(f"     want: {want}" + ("" if untouched else "  (the input plan was modified)"))
    print(f"\n{len(CASES) - bad}/{len(CASES)}")
    return bad


if __name__ == "__main__":
    raise SystemExit(selftest())
