"""Sit <-> stand with hysteresis: sit down to rest when nothing happens, stand up when something does.

From the ethogram's posture policy: "dogs lie down when idle, but most behaviors happen standing".
    idle (sitting) --any event--> standing (interacting)
          ^                            |
          +--- T_IDLE s with no event --+

Why hysteresis: with "no event -> sit, event -> stand", an owner who speaks every 3 seconds makes the duck
sit, stand, sit, stand... It looks broken, and each transition takes over a second and wobbles. The two
directions get different thresholds: stand up immediately, sit down only after T_IDLE seconds in a row with
nothing happening. Same idea as a Schmitt trigger or a thermostat.

The pet loop calls update(dt, active) every 20 ms; active = something is happening right now (the owner just
spoke, or the scheduler is running a plan). Returns "stand" / "sit" / None (nothing to do).

    python -m pet.body.idle
"""

T_IDLE = 20.0   # seconds with nothing happening before sitting down (ethogram: 20-30 s)


class IdlePosture:
    def __init__(self, posture: str = "sitting"):
        self.posture = posture     # "sitting" or "standing" (a state, not a command)
        self.idle_s = 0.0          # seconds in a row with nothing happening

    def update(self, dt: float, active: bool):
        if active:
            self.idle_s = 0.0
            if self.posture == "sitting":
                self.posture = "standing"
                return "stand"
            return None
        else:
            self.idle_s += dt
            if self.posture == "standing" and self.idle_s >= T_IDLE:
                self.posture = "sitting"
                return "sit"
            return None


# ─────────────────────────────────────────────────────────────
# selftest: each case is a timeline, one call per second; which seconds are "active", which commands are expected
# ─────────────────────────────────────────────────────────────
CASES = [
    # (description, start posture, total seconds, active seconds, expected [(second, command)])
    ("standing, ignored -> sits at 20 s, only once", "standing", 40, [], [(20, "sit")]),
    ("sitting, owner speaks at 5 s -> stands at once", "sitting", 10, [5], [(5, "stand")]),
    ("standing, a sentence every 15 s -> never sits (hysteresis)", "standing", 60, [15, 30, 45], []),
    ("standing and something happens -> no need to stand", "standing", 5, [2], []),
    ("sitting, nothing happens -> nothing to do", "sitting", 60, [], []),
    ("whole story: speak -> stand -> sit 20 s later -> speak -> stand -> sit 20 s later", "sitting", 60, [0, 25],
     [(0, "stand"), (20, "sit"), (25, "stand"), (45, "sit")]),
    ("something at 19 s -> timer restarts, sits at 39 s", "standing", 45, [19], [(39, "sit")]),
]


def selftest() -> int:
    n_bad = 0
    for desc, start, total, events, expect in CASES:
        sm = IdlePosture(start)
        got = []
        for sec in range(total + 1):
            cmd = sm.update(1.0 if sec > 0 else 0.0, sec in events)
            if cmd:
                got.append((sec, cmd))
        ok = got == expect
        n_bad += not ok
        print(f"{'ok ' if ok else 'BAD'} {desc}\n     got  {got}\n     want {expect}")
    print(f"\n{len(CASES) - n_bad}/{len(CASES)}")
    return n_bad


if __name__ == "__main__":
    raise SystemExit(selftest())
