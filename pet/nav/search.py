"""Search: turn when nothing is found, turn toward a box cut off by the image edge, and say honestly when the
target cannot be found.

    a behavior needs world["owner"]
      -> look_for(observe, turn, "owner", log)
           observe: stop -> photo -> detector -> VLM check -> geometry -> (status, landing point)
           turn: turn in place by some degrees (positive = left)
      -> landing point -> goto; None -> "not found" -> head tilt (never invent a position)

observe and turn are parameters, like ask= in planner.llm.plan_with_retry: the selftest feeds a fake sequence of
observations, with no simulator and no API cost; perception/vision_world.py plugs in the real ones.

observe returns one of four statuses
    "ok"         found, the VLM agrees, the box does not touch the left / right image edge -> second value = (x, y)
    "cut_left"   found, but the box touches the left edge -> part of the target is outside, to the left
    "cut_right"  found, but the box touches the right edge
    "none"       nothing found, or the VLM says it is not the target

    python -m pet.nav.search
"""

SCAN_DEG = 72       # nothing found -> turn left 72 deg: horizontal field of view ~106 deg, 34 deg overlap; 72 x 5 = 360
CUT_DEG = 30        # box cut off -> turn 30 deg toward that side
MAX_SHOTS = 6       # total photo limit: stops "cut left -> cut right -> cut left ..." from turning forever


def look_for(observe, turn, target: str, log: list):
    """Find target. Returns the landing point (x, y), or None if not found.
      each round: status, p = observe(target), one more photo
      "ok" -> log and return p
      MAX_SHOTS photos taken -> give up (no more turning)
      "cut_left" -> turn(+CUT_DEG); "cut_right" -> turn(-CUT_DEG)
      "none": if the degrees already turned because of "none" plus SCAN_DEG would reach 360 -> a full circle has
              been seen -> give up; otherwise turn(+SCAN_DEG) and count it
    """
    shot_count = 0
    total_turn = 0
    while True:
        status, p = observe(target)
        shot_count += 1
        if status == "ok":
            log.append(f"found {target} at {p}")
            return p
        if shot_count >= MAX_SHOTS:
            log.append(f"cannot find {target} ({shot_count} photos)")
            return None
        if status == "cut_left":
            turn(CUT_DEG)
        elif status == "cut_right":
            turn(-CUT_DEG)
        elif status == "none":
            if total_turn + SCAN_DEG >= 360:
                log.append(f"cannot find {target} ({shot_count} photos)")
                return None
            turn(SCAN_DEG)
            total_turn += SCAN_DEG


def selftest() -> int:
    P = (-2.7, -2.0)
    cases = [  # (fake observations, expected turns, expected return)
        (["ok"], [], P),
        (["none", "none", "ok"], [72, 72], P),
        (["cut_right", "ok"], [-30], P),
        (["none", "cut_left", "ok"], [72, 30], P),
        (["none"] * 10, [72, 72, 72, 72], None),                         # 5 photos cover a full circle -> give up, no 5th turn
        (["cut_left", "cut_right"] * 5, [30, -30, 30, -30, 30], None),   # flip-flopping -> 6-photo limit
    ]
    ok = 0
    for seq, want_turns, want in cases:
        it, turns, log = iter(seq), [], []
        observe = lambda target: (s := next(it), P if s == "ok" else None)
        got = look_for(observe, turns.append, "owner", log)
        good = turns == want_turns and got == want
        ok += good
        print(f"{'ok ' if good else 'BAD'} {seq[:6]}{'...' if len(seq) > 6 else ''}\n     turns {turns} (want {want_turns}) -> {got} (want {want})  log={log}")
    print(f"{ok}/{len(cases)}")
    return len(cases) - ok


if __name__ == "__main__":
    raise SystemExit(selftest())
