"""Number rules: for each number the LLM writes, pass, clamp, or reject.

    1. Structure wrong                        -> reject (schema.Plan)
    2. Not a finite number (NaN, +-inf)       -> reject     <- must be checked first, see below
    3. Within range                           -> pass
    4. Out of range but within 5x the limit   -> clamp to the limit + one log line
    5. More than 5x the limit                 -> reject (the size of a unit error: 10x, 57x, 100x)
    6. Any step rejected                      -> the whole plan is rejected (translate.py)

Why NaN first: every comparison with NaN is False. abs(nan) > 2.0 is False (rule 5 misses it) and
abs(nan) <= 0.4 is False too, so it would fall through to "clamp".

    python -m pet.planner.limits
"""

import math

REJECT_RATIO = 5.0  # beyond this many times the limit it is a unit error, not an overshoot


class RangeReject(ValueError):
    """This number cannot be used; the whole plan goes back."""


def apply_limit(name: str, x: float, limit: float, log: list[str]) -> float:
    """Return the value to use for field `name`, legal range [-limit, +limit]; raise RangeReject if unusable.
    Appends one line to `log` only when it clamps."""
    if not math.isfinite(x):
        raise RangeReject(f"{name}={x} is not a finite number")
    if abs(x) <= limit:
        return x
    # rule 5 before rule 4: otherwise every unit error would be quietly clamped
    if abs(x) > limit * REJECT_RATIO:
        raise RangeReject(f"{name}={x} is more than {REJECT_RATIO}x the limit {limit}")
    clamped = math.copysign(limit, x)
    log.append(f"{name} {x} -> {clamped} (clamped)")
    return clamped


def snap_deadzone(name: str, x: float, pos_min: float, neg_min: float, log: list[str]) -> float:
    """Dead zone: a value between "moves" and "does not move" goes to the nearest value that is actually executable.

    executable:  -limit ━━━ -neg_min    0    pos_min ━━━ +limit
                              └ dead ┘     └ dead ┘
      x == 0                 -> 0 (no log)
      0 < x < pos_min        -> 0 if closer to 0, else pos_min; exactly halfway -> pos_min
                                (the user asked to move, so prefer moving)
      -neg_min < x < 0       -> same on the negative side
      otherwise              -> unchanged (no log)
    Called after apply_limit, so x is finite and within range.
    """
    if x == 0:
        return 0.0
    if 0 < x < pos_min:
        if x < pos_min / 2:
            log.append(f"{name} {x} -> 0 (dead zone)")
            return 0.0
        log.append(f"{name} {x} -> {pos_min} (dead zone)")
        return pos_min
    if 0 > x > -neg_min:
        if x > -neg_min / 2:
            log.append(f"{name} {x} -> 0 (dead zone)")
            return 0.0
        log.append(f"{name} {x} -> -{neg_min} (dead zone)")
        return -neg_min
    return x


# ─────────────────────────────────────────────────────────────
# selftest
# ─────────────────────────────────────────────────────────────
R = "reject"
CASES = [
    # (description,                     field,          x,            limit, expected)
    ("in range",                         "forward_mps", 0.2,          0.4,  0.2),
    ("exactly at the limit",             "forward_mps", 0.4,          0.4,  0.4),
    ("slightly over",                    "forward_mps", 0.41,         0.4,  0.4),
    ("full sprint",                      "forward_mps", 1.0,          0.4,  0.4),
    ("over backward: keep the sign",     "forward_mps", -1.0,         0.4,  -0.4),
    ("exactly 5x: still clamped",        "forward_mps", 2.0,          0.4,  0.4),
    ("just over 5x",                     "forward_mps", 2.01,         0.4,  R),
    ("head 1833 deg",                    "yaw_left_deg", 1833.0,      80.0, R),
    ("centimeters as meters",            "forward_mps", 30.0,         0.4,  R),
    ("wrong direction (legal but wrong)", "forward_mps", -0.4,        0.4,  -0.4),
    ("NaN",                              "forward_mps", float("nan"), 0.4,  R),
    ("inf",                              "forward_mps", float("inf"), 0.4,  R),
    ("-inf",                             "forward_mps", float("-inf"), 0.4, R),
]

DZ_CASES = [
    # (description, x, pos_min, neg_min, expected, expected log lines) -- example dead zone (0.2, 0.3)
    ("stop (0)", 0.0, 0.2, 0.3, 0.0, 0),
    ("0.1 is exactly halfway -> move", 0.1, 0.2, 0.3, 0.2, 1),
    ("0.15 is closer to 0.2", 0.15, 0.2, 0.3, 0.2, 1),
    ("0.03 is closer to 0 -> stop", 0.03, 0.2, 0.3, 0.0, 1),
    ("0.25 already moves -> unchanged", 0.25, 0.2, 0.3, 0.25, 0),
    ("exactly 0.2 -> unchanged", 0.2, 0.2, 0.3, 0.2, 0),
    ("-0.2 is closer to -0.3", -0.2, 0.2, 0.3, -0.3, 1),
    ("-0.1 is closer to 0 -> stop", -0.1, 0.2, 0.3, 0.0, 1),
    ("-0.15 is exactly halfway -> move", -0.15, 0.2, 0.3, -0.3, 1),
    ("-0.35 already moves -> unchanged", -0.35, 0.2, 0.3, -0.35, 0),
    ("turn 30 deg/s with a 43 dead zone -> 43", 30.0, 43, 43, 43.0, 1),
]


def selftest_deadzone() -> int:
    n_bad = 0
    for desc, x, pmin, nmin, expect, n_log in DZ_CASES:
        log: list[str] = []
        got = snap_deadzone("x", x, pmin, nmin, log)
        ok = math.isclose(got, expect, abs_tol=1e-12) and len(log) == n_log
        n_bad += not ok
        print(f"{'ok ' if ok else 'BAD'} {desc:<40} got={got:<6g} want={expect:<6g} log lines {len(log)} (want {n_log})")
    print(f"\ndead zone: {len(DZ_CASES) - n_bad}/{len(DZ_CASES)} as expected")
    return n_bad


def selftest() -> int:
    n_bad = 0
    for desc, name, x, limit, expect in CASES:
        log: list[str] = []
        try:
            got = apply_limit(name, x, limit, log)
        except RangeReject as e:
            got, why = R, str(e)
        else:
            why = log[0] if log else ""
        same = (got == R) if expect == R else (got != R and math.isclose(got, expect))
        clamped = expect != R and expect != x          # only a clamp may write a log line
        log_ok = len(log) == (1 if clamped else 0)
        ok = same and log_ok
        n_bad += not ok
        note = "" if log_ok else f"  (log should have {1 if clamped else 0} lines, has {len(log)})"
        print(f"{'ok ' if ok else 'BAD'} {desc:<34} got={got!s:<6} want={expect!s:<6} {why}{note}")
    print(f"\n{len(CASES) - n_bad}/{len(CASES)} as expected")
    return n_bad


if __name__ == "__main__":
    bad = selftest()
    print()
    bad += selftest_deadzone()
    raise SystemExit(bad)
