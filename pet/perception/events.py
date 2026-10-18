"""Event state machine: a sequence of observed states -> events (came home, stood up, leaving, toy out).

Why the program and not the VLM: one photo shows "a person standing at the entrance". Just arrived or
about to leave? Cannot tell. An event is a change of state, so it needs memory of what came before.

    1. Not seen is not gone: present=False only means "not seen this time" (the duck may be looking at
       a toy), so it is skipped and no memory changes.
    2. One photo can be wrong, so a new state must be seen CONFIRM times in a row before it is believed
       (debouncing, the same idea as hysteresis).

Observation (dict):
    {"present": bool, "posture": "standing|walking|sitting|lying|unknown",
     "arms": "resting|waving|on_hips|pointing|open|hugging_knees|holding_toy|unknown",
     "region": "living_room|hallway|entrance|other", "motion": "still|receding|approaching|lowering|rising"}

    python -m pet.perception.events
"""

CONFIRM = 2                                      # a new state must be seen this many times in a row
KEYS = ("posture", "arms", "region", "motion")   # the fields compared for "did it change"
UP, DOWN = ("standing", "walking"), ("sitting", "lying")

# motion comes from the detector's bounding box over ~1.5 s, so it needs no distance estimate.
# The first version used box height alone, but sitting down also makes the box shorter, and a
# standing-then-sitting owner was reported as leaving (0/3 on that held-out scenario). Walking away
# raises the feet in the image (a farther floor point is higher up); sitting down never does.
# So: away = shorter AND feet up; closer = taller AND feet down; feet not moving = sitting / standing up.
MOTION_RATIO = 0.15   # box height vs. ~1.5 s ago: more than 15% counts as a change
BOTTOM_PX = 3.0       # the box bottom (feet) must move more than this many pixels
# Measured: a still owner's box bottom drifts at most 1 px in 1.5 s. Walking away 0.75 m in 1.5 s raises
# the feet 12.6 px from 1.5 m, 7.8 px from 2 m, 3.9 px from 3 m, 2.9 px from 3.5 m. 4 px would miss
# people starting to leave from 3 m; a missed event was judged worse than a false one, so 3 px.
# Real cameras jitter more than simulation: re-measure on hardware.


def classify_motion(h_ratio: float, d_bottom: float) -> str:
    """h_ratio: box height now / ~1.5 s ago. d_bottom: box bottom now - then, in pixels (v grows downward,
    so negative = feet moved up = farther). Returns receding / approaching / lowering / rising / still."""
    if h_ratio < 1 - MOTION_RATIO:
        return "receding" if d_bottom < -BOTTOM_PX else "lowering"
    elif h_ratio > 1 + MOTION_RATIO:
        return "approaching" if d_bottom > BOTTOM_PX else "rising"
    else:
        return "still"


MOTION_CASES = [  # (description, h_ratio, d_bottom, expected)
    ("no change", 1.0, 0, "still"),
    ("walking away: shorter, feet up 8 px", 0.80, -8, "receding"),
    ("sitting on the floor: shorter, feet down 10 px (legs stretched out)", 0.80, 10, "lowering"),
    ("sitting on a chair: shorter, feet still", 0.80, 0, "lowering"),
    ("shorter, feet moved 2 px (box jitter) -> feet did not move", 0.80, -2, "lowering"),
    ("walking closer: taller, feet down 8 px", 1.25, 8, "approaching"),
    ("standing up: taller, feet still", 1.25, 0, "rising"),
    ("standing up from the floor: taller, feet pulled in 10 px", 1.25, -10, "rising"),
    ("height changed only 5%, feet up 8 px (walking sideways) -> still", 0.95, -8, "still"),
]


class EventDetector:
    def __init__(self, confirm: int = CONFIRM):
        self.confirm = confirm
        self.known_home = False      # do I know the owner is home (starts: not home)
        self.confirmed = None        # the believed state (dict over KEYS); None = nothing believed yet
        self.pending = None          # a new state being watched, not believed yet
        self.count = 0               # times `pending` has been seen in a row

    def update(self, obs: dict) -> list[str]:
        """Feed one observation; return the events it produces (usually [])."""
        if not obs["present"]:
            return []                # not seen is not gone
        new = {k: obs[k] for k in KEYS}
        if new == self.confirmed:
            self.pending = None
            self.count = 0
            return []
        if new == self.pending:
            self.count += 1
        else:
            self.pending = new
            self.count = 1
        if self.count < self.confirm:
            return []
        old = self.confirmed
        self.confirmed = new
        self.pending = None
        self.count = 0
        return self._events(old, new)

    def _events(self, old, new) -> list[str]:
        """A new state was just believed: old (None the first time) -> new. Several events may happen at once."""
        events = []
        if not self.known_home:
            events.append("came_home")
            self.known_home = True
        if old is not None and old["posture"] in DOWN and new["posture"] in UP:
            events.append("stood_up")
        # The first version used "region changed from living room to hallway": beyond 3 m the floor-plane
        # depth estimate is off by 0.2-0.4 m and an owner just past the doorway still read as living room
        # (0/3). This version uses only relative cues, reported once when they *start* walking away.
        if (old is not None and new["posture"] in UP and new.get("motion") == "receding" and
                old.get("motion") != "receding"):
            events.append("leaving")
        if (old is None or old["arms"] != "holding_toy") and new["arms"] == "holding_toy":
            events.append("toy_out")
        return events


# ─────────────────────────────────────────────────────────────
# selftest: each case is a sequence of observations and the events it should produce
# ─────────────────────────────────────────────────────────────
def O(posture="sitting", arms="resting", region="living_room", present=True, motion="still"):
    return {"present": present, "posture": posture, "arms": arms, "region": region, "motion": motion}


NO = O(present=False, posture="unknown", arms="unknown", region="other")
CASES = [
    ("owner enters at the entrance (seen twice)", [O("walking", region="entrance")] * 2, ["came_home"]),
    ("seen once, then not seen -> not sure yet", [O("walking", region="entrance"), NO, NO], []),
    ("came home counts once while they stay in view", [O("walking", region="entrance")] * 2 + [O("sitting")] * 4, ["came_home"]),
    ("sitting with one wrong 'standing' frame -> not stood up", [O()] * 2 + [O("standing")] + [O()] * 2, ["came_home"]),
    ("really stands up (seen twice)", [O()] * 2 + [O("standing")] * 2, ["came_home", "stood_up"]),
    ("lying -> standing also counts", [O("lying")] * 2 + [O("standing")] * 2, ["came_home", "stood_up"]),
    ("stands up, turns and walks away -> stood up + leaving",
     [O()] * 2 + [O("standing")] * 2 + [O("walking", motion="receding")] * 2, ["came_home", "stood_up", "leaving"]),
    ("keeps walking away -> leaving once", [O("standing")] * 2 + [O("walking", motion="receding")] * 5, ["came_home", "leaving"]),
    ("walking closer -> not leaving", [O("standing")] * 2 + [O("walking", motion="approaching")] * 2, ["came_home"]),
    ("sitting and the box shrinks (leaning back) -> not leaving", [O()] * 2 + [O("sitting", motion="receding")] * 2, ["came_home"]),
    ("one noisy receding frame -> not leaving",
     [O("standing")] * 2 + [O("standing", motion="receding")] + [O("standing")] * 2, ["came_home"]),
    ("walks away, stops, walks away again -> leaving twice",
     [O("standing")] * 2 + [O("walking", motion="receding")] * 2 + [O("standing")] * 2
     + [O("walking", motion="receding")] * 2, ["came_home", "leaving", "leaving"]),
    ("duck looks away for 3 frames -> memory unchanged, no second came_home", [O()] * 2 + [NO] * 3 + [O()] * 2, ["came_home"]),
    ("brings out a toy", [O("standing")] * 2 + [O("standing", arms="holding_toy")] * 2, ["came_home", "toy_out"]),
    ("standing owner sits down: box shrinks without feet rising (lowering) -> not leaving",
     [O("standing")] * 2 + [O("standing", motion="lowering")] * 2, ["came_home"]),
    ("keeps holding the toy -> toy_out once", [O("standing")] * 2 + [O("standing", arms="holding_toy")] * 5, ["came_home", "toy_out"]),
    ("puts it down and picks it up again -> twice",
     [O("standing")] * 2 + [O("standing", arms="holding_toy")] * 2 + [O("standing")] * 2
     + [O("standing", arms="holding_toy")] * 2, ["came_home", "toy_out", "toy_out"]),
]


def selftest() -> int:
    ok = 0
    for desc, seq, want in CASES:
        det, got = EventDetector(), []
        for obs in seq:
            got += det.update(obs)
        good = got == want
        ok += good
        print(f"{'ok ' if good else 'BAD'} {desc}\n     got {got}" + ("" if good else f" (want {want})"))
    print("-- motion --")
    for desc, r, db, want in MOTION_CASES:
        got = classify_motion(r, db)
        good = got == want
        ok += good
        print(f"{'ok ' if good else 'BAD'} {desc} -> {got}" + ("" if good else f" (want {want})"))
    total = len(CASES) + len(MOTION_CASES)
    print(f"\n{ok}/{total}")
    return total - ok


if __name__ == "__main__":
    raise SystemExit(selftest())
