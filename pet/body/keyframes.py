"""Keyframe player for head gestures (the layer that makes the duck look alive).

A fixed head angle cannot express nodding, shaking or tilting: those change over time. When the duck was
already standing, "stand -> head left -> head right" finished each step instantly and the head never moved.
A gesture is a few keyframes; every 20 ms the angle in between is interpolated. The LLM only picks the
gesture by name (a whitelist, like picking a policy); the keyframes are written in advance, so nothing
the LLM writes can produce an unchecked motion.

    LLM: {"action": "gesture", "name": "wag"}
      -> KEYFRAMES["wag"] = yaw keyframes [(0, 0), (0.21, 25), (0.63, -25), ...]
      -> every 20 ms: interp(keyframes, t) -> head angle -> slots 4-6 of the 13 command numbers
      -> run with the policy of the current posture (standing: walk with zero velocity; sitting: sitstand flag 1);
         the policy moves only the head and keeps the body still

Head signs (verified in simulation): yaw positive = left; pitch positive = head down; roll (tilt) trained range
+-0.31 rad (about +-17 deg). Descriptions shown to the LLM are in planner/catalog.py.

    python -m pet.body.keyframes
"""

import math


def interp(keys: list, t: float) -> float:
    """Linear interpolation between keyframes [(t0, v0), (t1, v1), ...] (times increasing).
    Before the first keyframe -> its value; after the last -> its value."""
    if t <= keys[0][0]:
        return keys[0][1]
    if t >= keys[-1][0]:
        return keys[-1][1]
    for i in range(len(keys) - 1):
        ta, va = keys[i]
        tb, vb = keys[i + 1]
        if ta <= t <= tb:
            alpha = (t - ta) / (tb - ta)
            return va + alpha * (vb - va)
    raise ValueError("keyframe times must be increasing")


def _wag(amp: float, hz: float, secs: float) -> list:
    """Side to side: switch sides every half period, return to center at the end."""
    half = 0.5 / hz
    keys, t, side = [(0.0, 0.0)], half / 2, 1
    while t < secs:
        keys.append((round(t, 3), side * amp))
        t, side = t + half, -side
    keys.append((round(t - half / 2, 3), 0.0))
    return keys


# name -> {channel: keyframes}; channels are yaw / pitch / roll in degrees, missing channels stay at 0.
# Numbers in the ethogram (docs/ethogram.md) are the behaviors each gesture stands in for.
# spin_hop is not a head gesture: it switches to its own policy (scheduler.py).
KEYFRAMES = {
    "nod": {"pitch": [(0, 0), (0.4, 15), (0.8, 0), (1.2, 15), (1.6, 0)]},              # #23 slow blink substitute
    "head_tilt": {"roll": [(0, 0), (0.3, 15), (1.3, 15), (1.6, 0)]},                    # #4 hears its name, curious
    # Measured (head yaw +-25 deg sine, standing): tracks 85% at 0.5 Hz, 77% at 1 Hz, 64% at 1.5 Hz, 52% at 2 Hz.
    # The policy is a low-pass filter (the action-rate penalty made it smooth): swing slow and wide, not fast.
    "wag": {"yaw": _wag(25, 1.2, 2.0)},                                                 # #25 tail wag substitute
    # second version: the first one at 2.2 Hz tracked only 36% and was invisible -> 1 Hz, wider, plus roll
    "happy": {"pitch": [(0, 0), (0.4, -25), (2.6, -25), (3.0, 0)],                      # pitch negative = head up
              "yaw": [(0, 0), (0.4, 0)] + [(t + 0.4, v) for t, v in _wag(25, 1.0, 2.0)[1:]],
              "roll": [(0, 0), (0.4, 0)] + [(t + 0.4, v * 0.5) for t, v in _wag(25, 1.0, 2.0)[1:]]},
    "look_around": {"yaw": [(0, 0), (0.8, 40), (1.6, 40), (2.6, -40), (3.4, -40), (4.0, 0)]},   # #21 exploring
    "shy": {"pitch": [(0, 0), (0.6, 30), (2.6, 30), (3.2, 0)],                          # #18 guilty look
            "yaw": [(0, 0), (0.6, 20), (2.6, 20), (3.2, 0)]},
}
LIMITS_DEG = {"yaw": 80, "pitch": 63, "roll": 17}   # roll: head_roll trained range +-0.31 rad
SLOT = {"pitch": 4, "yaw": 5, "roll": 6}             # index in the 13 command numbers


def duration(name: str) -> float:
    return max(keys[-1][0] for keys in KEYFRAMES[name].values())


def head_at(name: str, t: float) -> dict:
    """Degrees of each head channel of gesture `name` at time t."""
    return {ch: interp(keys, t) for ch, keys in KEYFRAMES[name].items()}


def cmd_at(name: str, t: float, base: list) -> list:
    """Replace the head part of `base` (the posture's 13 numbers) with the gesture's angles at t (radians)."""
    cmd = list(base)
    for ch, deg in head_at(name, t).items():
        cmd[SLOT[ch]] = math.radians(deg)
    return cmd


# ─────────────────────────────────────────────────────────────
# selftest
# ─────────────────────────────────────────────────────────────
HAND = [(0.0, 0.0), (0.3, 15.0), (1.3, 15.0), (1.6, 0.0)]   # worked by hand before this was written
CASES = [
    ("t=0.25 (12.5 by hand)", HAND, 0.25, 12.5),
    ("t=1.5 (5 by hand; forgetting 'va +' gives -10)", HAND, 1.5, 5.0),
    ("t=0.8 holds at 15", HAND, 0.8, 15.0),
    ("exactly on a keyframe t=0.3", HAND, 0.3, 15.0),
    ("exactly at the end t=1.6", HAND, 1.6, 0.0),
    ("before the start t=-1", HAND, -1.0, 0.0),
    ("after the end t=5", HAND, 5.0, 0.0),
    ("a set that does not start at 0", [(0, 10), (2, 30)], 0.5, 15.0),
]


def selftest() -> int:
    n_bad = 0
    for desc, keys, t, expect in CASES:
        got = interp(keys, t)
        ok = math.isclose(got, expect, abs_tol=1e-9)
        n_bad += not ok
        print(f"{'ok ' if ok else 'BAD'} {desc:<48} got={got}  want={expect}")
    print(f"\ninterp: {len(CASES) - n_bad}/{len(CASES)}")
    if n_bad:
        return n_bad
    print("\ngesture library (sampled every 20 ms: within the trained range, ends at 0, at most 3 deg per 20 ms):")
    for name, chans in KEYFRAMES.items():
        secs = duration(name)
        samples = [head_at(name, k * 0.02) for k in range(int(secs / 0.02) + 2)]
        over = any(abs(v) > LIMITS_DEG[ch] for s in samples for ch, v in s.items())
        jump = max(abs(b[ch] - a[ch]) for a, b in zip(samples, samples[1:]) for ch in a)
        back = all(abs(v) < 1e-9 for v in samples[-1].values())
        ok = not over and back and jump <= 3.0
        n_bad += not ok
        print(f"  {'ok ' if ok else 'BAD'} {name:<12} {secs:.1f} s  largest step {jump:.2f} deg")
    return n_bad


if __name__ == "__main__":
    raise SystemExit(selftest())
