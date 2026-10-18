"""Preconditions: a sitting duck cannot walk, so the program inserts "stand up" automatically.

Runs before translation because the inserted step is an ordinary SitStand that also has to become
13 numbers.

Design: the start posture is measured (torso height in simulation; joint angles + FK on hardware);
the posture after each step is bookkept here; the scheduler checks again before executing.

    walk, kick, spin_hop and walking behaviors need standing; sitstand can start from either posture
    needs standing while sitting -> insert SitStand(stand) and log it (never silently)
    start is neither standing nor sitting (e.g. fallen) -> reject, the caller degrades

    python -m pet.planner.preconditions
"""

from .catalog import BEHAVIORS, end_posture
from .schema import Behavior, Gesture, Kick, SitStand, Walk

POSTURES = ("standing", "sitting")
INSERTED: list = []   # steps inserted here, so subtitles can mark them [auto] instead of looking like LLM choices


class PreconditionReject(ValueError):
    """The start posture cannot be planned from (e.g. fallen)."""


def needs_standing(step) -> bool:
    return (isinstance(step, (Walk, Kick)) or (isinstance(step, Gesture) and step.name == "spin_hop")
            or (isinstance(step, Behavior) and BEHAVIORS[step.name][2]))


def insert_preconditions(steps: list, start: str, log: list[str]) -> list:
    """Return a new plan with "stand up" inserted where needed; `steps` is not modified.
    start: "standing" or "sitting". Appends one log line per inserted step."""
    if start not in POSTURES:
        raise PreconditionReject(f"cannot plan from start posture {start!r} (not standing / sitting)")

    posture = start
    out = []
    for step in steps:
        if needs_standing(step) and posture == "sitting":
            out.append(SitStand(action="sitstand", posture="stand"))
            INSERTED.append(out[-1])
            log.append("inserted SitStand(stand) (precondition)")
            posture = "standing"
        out.append(step)
        if isinstance(step, SitStand):
            posture = "standing" if step.posture == "stand" else "sitting"
        if isinstance(step, Behavior):
            # behaviors change posture too (stay_close ends sitting). Missing this once made the duck try
            # play_bow while sitting, and it fell.
            end = end_posture(step.name)
            if end is not None:
                posture = "standing" if end == "stand" else "sitting"
    return out


# ─────────────────────────────────────────────────────────────
# selftest
# ─────────────────────────────────────────────────────────────
def short(step) -> str:
    if isinstance(step, SitStand):
        return f"sitstand({step.posture})"
    if isinstance(step, Walk):
        return f"walk {step.forward_mps:g}"
    if isinstance(step, Behavior):
        return f"behavior({step.name})"
    return f"kick({step.target})"


W = Walk(action="walk", forward_mps=0.2, duration_s=2)
SIT = SitStand(action="sitstand", posture="sit")
STAND = SitStand(action="sitstand", posture="stand")
K = Kick(action="kick", target="owner")
BH = lambda n: Behavior(action="behavior", name=n)
R = "reject"

CASES = [
    # (description, start, LLM plan, expected full plan or R, expected insertions)
    ("sitting, walk then sit", "sitting", [W, SIT], ["sitstand(stand)", "walk 0.2", "sitstand(sit)"], 1),
    ("standing, sit, kick, walk", "standing", [SIT, K, W], ["sitstand(sit)", "sitstand(stand)", "kick(owner)", "walk 0.2"], 1),
    ("standing, walk: nothing to insert", "standing", [W], ["walk 0.2"], 0),
    ("sitting, LLM already stands up: no duplicate", "sitting", [STAND, W], ["sitstand(stand)", "walk 0.2"], 0),
    ("sitting, sit again: nothing to insert", "sitting", [SIT], ["sitstand(sit)"], 0),
    ("sit-kick, sit-walk: insert twice", "standing", [SIT, K, SIT, W],
     ["sitstand(sit)", "sitstand(stand)", "kick(owner)", "sitstand(sit)", "sitstand(stand)", "walk 0.2"], 2),
    ("fallen: cannot plan", "fallen", [W], R, 0),
    ("stay_close ends sitting -> stand before play_bow", "standing", [BH("stay_close"), BH("play_bow")],
     ["behavior(stay_close)", "sitstand(stand)", "behavior(play_bow)"], 1),
    ("stay_close, rest, then walk -> stand first", "standing", [BH("stay_close"), BH("rest"), W],
     ["behavior(stay_close)", "behavior(rest)", "sitstand(stand)", "walk 0.2"], 1),
    ("guilty ends sitting -> stand before play_bow", "standing", [BH("guilty"), BH("play_bow")],
     ["behavior(guilty)", "sitstand(stand)", "behavior(play_bow)"], 1),
    ("rest ends sitting -> stand before walking", "standing", [BH("rest"), W], ["behavior(rest)", "sitstand(stand)", "walk 0.2"], 1),
    ("gaze keeps posture: gaze sitting, stand before greet", "sitting", [BH("gaze"), BH("greet")],
     ["behavior(gaze)", "sitstand(stand)", "behavior(greet)"], 1),
]


def selftest() -> int:
    n_bad = 0
    for desc, start, steps, expect, n_ins in CASES:
        before = [short(s) for s in steps]
        log: list[str] = []
        try:
            got = [short(s) for s in insert_preconditions(steps, start, log)]
            why = "; ".join(log)
        except PreconditionReject as e:
            got, why = R, str(e)
        untouched = [short(s) for s in steps] == before
        ok = got == expect and len(log) == n_ins and untouched
        n_bad += not ok
        print(f"{'ok ' if ok else 'BAD'} {desc} (start {start})\n     got:  {got}  {why}")
        if not ok:
            print(f"     want: {expect} ({n_ins} insertions, log has {len(log)})"
                  + ("" if untouched else "  (the input plan was modified)"))
    print(f"\n{len(CASES) - n_bad}/{len(CASES)} as expected")
    return n_bad


if __name__ == "__main__":
    raise SystemExit(selftest())
