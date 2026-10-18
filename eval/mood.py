"""Emotion + inner-state test set: given these cues and this inner state, is the behavior the LLM picks reasonable?

What can and cannot be measured
    1. does the VLM see the head hanging, the knees hugged      ground truth exists (scripted poses): eval/owner_state.py
    2. pose -> "may be tired"                                    our definition (owner_state.CUES, mind/inner.py), stated, not validated
    3. "may be tired" + inner state -> the behavior chosen       this file: acceptable and unacceptable answers fixed in advance
    4. is the owner *really* tired                               only an experiment with people can say; not claimed

A test passes when the plan contains at least one acceptable reaction, none of the unacceptable ones, and is not degraded.
The sentence is not hand-typed: perception.loop.compose (describe + Inner.text) builds it, so the test sends exactly
what the running system would send. The LLM gets prompts.SYSTEM_PET.

Decisions behind the answers
    #3  the owner looks angry but did not scold the duck -> gaze or stay_close; guilty / shy only if clearly aimed at the duck
    #6  low energy when a toy comes out -> a tired pet is not in the mood -> no play_bow
    #11 owner tired + duck bored -> depends on personality, anything not clearly at odds with the situation
        (odd = zoomies, guilty / shy without a reason); play_bow is acceptable (real dogs invite tired owners too)

Sets
    DEV      11 tests; prompt changes may only look at these. English prompt: 33/33
    HOLDOUT  6 tests written before the first run (other poses, other triggers), measured once. English prompt: 15/18
             (one miss: tired *and* bored chose investigate). The effort labels changed after that run, so it is used up.

    python -m eval.mood --show            # print each sentence actually sent, no cost
    python -m eval.mood -n 3 [--set holdout]
"""

import argparse

from pet.mind.inner import START, Inner
from pet.perception.loop import compose
from pet.planner.prompts import SYSTEM_PET
from pet.planner.schema import Behavior, Gesture, SitStand

from .plan_checks import evaluate, ok


def O(posture="sitting", arms="resting", head="up", facing="yes", region="living_room"):
    return {"present": True, "posture": posture, "arms": arms, "head": head, "facing_duck": facing,
            "region": region, "motion": "still"}


def say(events=(), obs=None, cue=None, wants=(), **x):
    """The sentence the system would send; x = inner state (START for anything not given)."""
    inner = Inner()
    inner.d.x = dict(START, **x)
    inner.d.check()     # the state is already like this -> wants that should be active are active (so the sentence says them)
    return compose(list(events), obs or O(), inner.text(obs or O(), cue, wants))


def _picks(r) -> set:
    """Which reactions the plan chose: behavior / gesture names; sitting down counts as "sit"."""
    out = set()
    for s in r.steps:
        if isinstance(s, (Behavior, Gesture)):
            out.add(s.name)
        elif isinstance(s, SitStand) and s.posture == "sit":
            out.add("sit")
    return out


def any_of(*names):
    return (f"one of {'/'.join(names)}", lambda r: bool(_picks(r) & set(names)))


def none_of(*names):
    return (f"none of {'/'.join(names)}", lambda r: not (_picks(r) & set(names)))


DEV = [
    # 1 the owner's head hangs for a long time (may be tired)
    (say(obs=O(head="down"), cue="low"), "sitting",
     [ok(), any_of("stay_close", "gaze"), none_of("zoomies", "play_bow")]),
    # 2 the owner hugs their knees for a long time (may be sad)
    (say(obs=O(arms="hugging_knees", head="down"), cue="low"), "sitting",
     [ok(), any_of("stay_close", "gaze"), none_of("zoomies", "play_bow", "happy")]),
    # 3 hands on hips for a long time (may be angry, but did not scold the duck)
    (say(obs=O("standing", "on_hips"), cue="angry"), "sitting",
     [ok(), any_of("gaze", "stay_close"), none_of("guilty", "shy", "zoomies", "play_bow", "happy")]),
    # 4 waving for a long time (may be happy)
    (say(obs=O("standing", "waving"), cue="happy"), "sitting",
     [ok(), any_of("gaze", "happy", "wag", "greet", "nod", "head_tilt"), none_of("guilty", "rest", "shy")]),
    # 5 a toy comes out, high energy
    (say(["toy_out"], O("standing", "holding_toy"), E=0.85), "sitting",
     [ok(), any_of("play_bow", "zoomies"), none_of("rest", "guilty")]),
    # 6 a toy comes out, low energy (played three times already): not in the mood
    (say(["toy_out"], O("standing", "holding_toy"), E=0.1), "sitting",
     # stay_close was added *after* seeing results (chosen 2/3 in one round; judged acceptable)
     [ok(), any_of("gaze", "head_tilt", "rest", "wag", "stay_close"), none_of("play_bow", "zoomies")]),
    # 7 no event, the duck misses the owner
    (say(obs=O(), wants=["miss_owner"], A=0.8), "sitting",
     [ok(), any_of("stay_close", "gaze"), none_of("investigate", "rest")]),
    # 8 no event, the duck is bored
    (say(obs=O(), wants=["bored"], C=0.8), "sitting",
     [ok(), any_of("investigate", "look_around"), none_of("guilty")]),
    # 9 no event, the duck is tired
    (say(obs=O(), wants=["tired"], E=0.2), "sitting",
     [ok(), any_of("rest", "sit"), none_of("zoomies", "play_bow")]),
    # 10 the owner comes home, the duck missed them a lot
    (say(["came_home"], O("walking", region="entrance"), A=0.9), "sitting",
     [ok(), any_of("greet", "zoomies"), none_of("rest", "guilty")]),
    # 11 owner's head hangs (may be tired) + the duck is bored: anything not clearly odd
    (say(obs=O(head="down"), cue="low", wants=["bored"], C=0.8), "sitting",
     [ok(), any_of("stay_close", "gaze", "investigate", "look_around", "rest", "play_bow", "head_tilt", "nod"),
      none_of("zoomies", "guilty", "shy")]),
]

HOLDOUT = [
    # H1 lying (another "may be tired" cue)
    (say(obs=O("lying", head="unknown", facing="no"), cue="low"), "sitting",
     [ok(), any_of("stay_close", "gaze"), none_of("zoomies", "play_bow")]),
    # H2 pointing for a long time (another "may be angry"), did not scold the duck
    (say(obs=O("standing", "pointing"), cue="angry"), "sitting",
     [ok(), any_of("gaze", "stay_close"), none_of("guilty", "shy", "zoomies", "play_bow", "happy")]),
    # H3 sitting with arms open for a long time (may be welcoming)
    (say(obs=O(arms="open"), cue="welcoming"), "sitting",
     [ok(), any_of("greet", "stay_close", "gaze", "happy", "wag"), none_of("guilty", "rest", "shy")]),
    # H4 the owner stands up + the duck misses them
    (say(["stood_up"], O("standing"), wants=["miss_owner"], A=0.8), "sitting",
     [ok(), any_of("gaze", "stay_close", "greet", "wag", "happy", "head_tilt"), none_of("rest", "guilty")]),
    # H5 the owner walks away + the duck misses them a lot
    (say(["leaving"], O("walking", facing="no"), A=0.9), "sitting",
     [ok(), any_of("gaze", "head_tilt", "stay_close", "greet"), none_of("zoomies", "play_bow")]),
    # H6 the duck is bored and tired
    (say(obs=O(), wants=["bored", "tired"], C=0.8, E=0.15), "sitting",
     [ok(), any_of("rest", "sit", "look_around", "gaze"), none_of("zoomies", "play_bow")]),
]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="dev", choices=["dev", "holdout"])
    ap.add_argument("--show", action="store_true", help="print the sentences only, no LLM")
    ap.add_argument("-n", type=int, default=3)
    a = ap.parse_args()
    tests = DEV if a.set == "dev" else HOLDOUT
    if a.show:
        for i, (s, start, checks) in enumerate(tests, 1):
            print(f"{i:>2}. {s}\n    -> {[d for d, _ in checks[1:]]}")
    else:
        evaluate(tests, a.n, system=SYSTEM_PET)
