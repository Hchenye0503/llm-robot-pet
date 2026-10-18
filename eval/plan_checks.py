"""Checks for planner outputs: each test sentence comes with conditions for a *correct* plan, not only a legal one.

Legal is not correct: "walk forward" -> -0.4 is legal; "turn, then walk" merged into one step is legal too.
So every test is (sentence, start posture, [conditions]). Conditions are built from these blocks; numeric conditions
are strings: ">0", "<0", "==0", ">=0.3", "<=2".

    ok()                          the LLM produced a plan (not degraded)
    walk(forward=, turn=, secs=)  some walk step matches (unset fields are not checked); m/s, deg/s, s
    sit(head_yaw=) / stand(...)   a sit / stand step (optionally with a head condition, deg); behaviors ending
                                  sitting / standing count too
    head_yaw(">0")                any step turns the head left (positive = left), gestures and behaviors included
    order("stand", "turn", "go")  these happen in order (other steps may be in between);
                                  turn = turn in place (forward 0), go = walk forward or backward
    no_merge()                    no step turns and walks at once
    ends("sitting" / "standing")  posture after the plan (bookkept)
    total_turn(">60")             total degrees turned (left positive, rate x seconds)
    distance(">=0.4")             total forward distance (m, backward negative)
    turn_around() / no_backward() / no_walk() / no_move() / no_left_turn() / head_moved() / refused()
    behavior(name) / gesture(name)
Plans are checked *after* preconditions are inserted (a sitting duck told to walk gets a stand step first).
Grade the goal, not the means: when a behavior achieves what the test asks for, the check should accept it.
"""

import operator
import re

from pet.body.keyframes import KEYFRAMES
from pet.planner.catalog import end_posture
from pet.planner.schema import Behavior, Gesture, Kick, Refuse, SitStand, Walk

_OPS = {">=": operator.ge, "<=": operator.le, "==": operator.eq, ">": operator.gt, "<": operator.lt}


def _cond(text):
    """">=0.3" -> f(x) -> bool."""
    m = re.fullmatch(r"\s*(>=|<=|==|>|<)\s*(-?[\d.]+)\s*", text)
    if not m:
        raise ValueError(f"cannot read condition {text!r}; write it like \">0\" or \">=0.3\"")
    op, num = _OPS[m.group(1)], float(m.group(2))
    return lambda x: op(round(x, 9), num)


def _kind(s) -> str:
    if isinstance(s, SitStand):
        return s.posture                       # "sit" / "stand"
    if isinstance(s, Kick):
        return "kick"
    if isinstance(s, Refuse):
        return "refuse"
    if isinstance(s, Gesture):
        return "gesture"
    if isinstance(s, Behavior):
        return "behavior"
    if s.forward_mps == 0 and s.turn_left_deg_s != 0:
        return "turn"
    if s.forward_mps != 0:
        return "go"
    return "still"


# ── blocks: each returns (description, check function) ──
def ok():
    return ("not degraded", lambda r: r.status == "ok")


def walk(forward=None, turn=None, secs=None):
    fs = [(f"forward{forward}", "forward_mps", _cond(forward)) if forward else None,
          (f"turn{turn}", "turn_left_deg_s", _cond(turn)) if turn else None,
          (f"{secs}s", "duration_s", _cond(secs)) if secs else None]
    fs = [f for f in fs if f]
    desc = "walk(" + ", ".join(f[0] for f in fs) + ")"
    return (desc, lambda r: any(isinstance(s, Walk) and all(c(getattr(s, a)) for _, a, c in fs) for s in r.steps))


def _posture_step(p, head_yaw):
    # behaviors that end sitting count as sitting down (rest was once graded "did not sit": the grader did not know the word)
    c = _cond(head_yaw) if head_yaw else (lambda x: True)
    desc = f"{p}" + (f"(head{head_yaw})" if head_yaw else "")
    return (desc, lambda r: any((isinstance(s, SitStand) and s.posture == p and c(s.head.yaw_left_deg))
                                or (head_yaw is None and isinstance(s, Behavior) and end_posture(s.name) == p)
                                for s in r.steps))


def sit(head_yaw=None):
    return _posture_step("sit", head_yaw)


def stand(head_yaw=None):
    return _posture_step("stand", head_yaw)


def head_yaw(cond):
    c = _cond(cond)
    # head yaw moves inside behaviors (body/behaviors.py): only used for "did the head swing", not exact angles
    BEH_YAW = {"greet": ["happy", "wag"], "play_bow": ["wag"], "investigate": ["look_around"]}

    def yaws(s):   # gesture / behavior head angles come from the keyframes
        if isinstance(s, Gesture):
            return [v for _, v in KEYFRAMES.get(s.name, {}).get("yaw", [])]
        if isinstance(s, Behavior):
            vals = [v for g in BEH_YAW.get(s.name, []) for _, v in KEYFRAMES[g].get("yaw", [])]
            return vals + ([20, -20] if s.name == "guilty" else [])
        return [s.head.yaw_left_deg] if hasattr(s, "head") else []
    return (f"head yaw{cond}", lambda r: any(c(v) for s in r.steps for v in yaws(s)))


def behavior(name):
    return (f"behavior {name}", lambda r: any(isinstance(s, Behavior) and s.name == name for s in r.steps))


def gesture(name):
    return (f"gesture {name}", lambda r: any(isinstance(s, Gesture) and s.name == name for s in r.steps))


def order(*kinds):
    def check(r):
        seq, i = [_kind(s) for s in r.steps], 0
        for k in seq:
            if i < len(kinds) and k == kinds[i]:
                i += 1
        return i == len(kinds)
    return ("order " + " -> ".join(kinds), check)


def no_merge():
    return ("no turning while walking", lambda r: not any(isinstance(s, Walk) and s.forward_mps != 0 and s.turn_left_deg_s != 0 for s in r.steps))


def ends(posture):
    def check(r):
        p = r.start
        for s in r.steps:
            if isinstance(s, SitStand):
                p = "sitting" if s.posture == "sit" else "standing"
            elif isinstance(s, Behavior):
                e = end_posture(s.name)
                if e is not None:
                    p = "sitting" if e == "sit" else "standing"
        return p == posture
    return (f"ends {posture}", check)


def total_turn(cond):
    c = _cond(cond)
    return (f"total turn{cond} deg", lambda r: c(sum(s.turn_left_deg_s * s.duration_s for s in r.steps if isinstance(s, Walk))))


def turn_around():
    """"Back up" = turn around (either side): one in-place turn of 150-210 deg (the pet gait cannot walk backward)."""
    def check(r):
        return any(isinstance(s, Walk) and s.forward_mps == 0 and 150 <= abs(s.turn_left_deg_s * s.duration_s) <= 210 for s in r.steps)
    return ("turns around ~180 deg in place", check)


def no_backward():
    return ("no backward walking (forward never negative)", lambda r: all(not isinstance(s, Walk) or s.forward_mps >= 0 for s in r.steps))


def no_walk():
    return ("no walk", lambda r: not any(isinstance(s, Walk) for s in r.steps))


def no_move():
    """No step moves the body (walk(0, 0) in place does not count as moving)."""
    return ("no movement", lambda r: not any(isinstance(s, Walk) and (s.forward_mps or s.turn_left_deg_s) for s in r.steps))


def refused():
    return ("refused", lambda r: any(isinstance(s, Refuse) for s in r.steps))


def distance(cond):
    """Total forward distance sum(forward_mps x duration_s) in meters (backward negative)."""
    c = _cond(cond)
    return (f"distance{cond} m", lambda r: c(sum(s.forward_mps * s.duration_s for s in r.steps if isinstance(s, Walk))))


def no_left_turn():
    return ("no step turns left", lambda r: not any(isinstance(s, Walk) and s.turn_left_deg_s > 0 for s in r.steps))


def head_moved():
    """"Look at me" = the head goes up (pitch < 0) or swings left / right (yaw != 0); looking down does not count.
    Known gap: behaviors (e.g. gaze, which does look up at the owner) are not recognized here."""
    def angles(s):   # all (pitch, yaw) values: head fields for plain steps, keyframes for gestures
        if isinstance(s, Gesture):
            kf = KEYFRAMES.get(s.name, {})
            return [(v, 0) for _, v in kf.get("pitch", [])] + [(0, v) for _, v in kf.get("yaw", [])]
        return [(s.head.pitch_deg, s.head.yaw_left_deg)] if hasattr(s, "head") else []
    return ("head up or swinging", lambda r: any(p < 0 or y != 0 for s in r.steps for p, y in angles(s)))


def not_(block):
    desc, f = block
    return (f"not {desc}", lambda r: not f(r))


# ─────────────────────────────────────────────────────────────
def check_format(tests) -> int:
    bad = 0
    for i, t in enumerate(tests, 1):
        problem = None
        if not (isinstance(t, tuple) and len(t) == 3):
            problem = "must be (sentence, start, [conditions])"
        elif t[1] not in ("standing", "sitting"):
            problem = f"start must be 'standing' or 'sitting', not {t[1]!r}"
        elif not t[2] or not all(isinstance(c, tuple) and callable(c[1]) for c in t[2]):
            problem = "empty condition list, or something that is not a block"
        print(f"{'ok ' if not problem else 'BAD'} {i:>2}. {t[0] if isinstance(t, tuple) else t}"
              + (f"  <- {problem}" if problem else f"  [{len(t[2])} conditions]"))
        bad += bool(problem)
    print(f"\n{len(tests)} tests, {bad} with format problems")
    return bad


def evaluate(tests, n: int, system=None):
    """Ask the LLM n times per test (answers vary) and grade each plan. system: None = typed-command prompt;
    prompts.SYSTEM_PET for perception-triggered sentences."""
    from pet.planner.llm import plan_with_retry
    from pet.planner.prompts import SYSTEM
    legal = total = sem_ok = 0
    rows = []
    for i, (say, start, checks) in enumerate(tests, 1):
        passes = 0
        print(f"\n{i:>2}. \"{say}\" (start {start})")
        for _ in range(n):
            res = plan_with_retry(say, start, system=system or SYSTEM)
            res.start = start
            total += 1
            legal += res.status == "ok"
            failed = [d for d, f in checks if not f(res)]
            passes += not failed
            plan = " -> ".join(_kind(s) + (f"({s.forward_mps:g},{s.turn_left_deg_s:g}deg/s,{s.duration_s:g}s)" if isinstance(s, Walk) else
                                           (f"(head {s.head.yaw_left_deg:g})" if hasattr(s, "head") and s.head.yaw_left_deg else
                                            (f"\"{s.reason}\"" if isinstance(s, Refuse) else
                                             (f"({s.name})" if isinstance(s, (Gesture, Behavior)) else ""))))
                               for s in res.steps) or "(empty)"
            print(f"   {'ok ' if not failed else 'BAD'} {res.status:<8} {plan}" + (f"   failed: {failed}" if failed else ""))
        sem_ok += passes
        rows.append((i, say, passes))
    print("\n==== result ====")
    print(f"legal: {legal}/{total}")
    print(f"correct: {sem_ok}/{total}")
    weak = [(i, s, p) for i, s, p in rows if p < n]
    if weak:
        print("not correct every time:")
        for i, s, p in weak:
            print(f"   {i:>2}. {p}/{n} \"{s}\"")
    return sem_ok, total
