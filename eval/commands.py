"""Typed-command test sets: what the owner says, and what counts as a correct plan (eval/plan_checks.py).

Rules for writing conditions
    write what the user actually cares about, not exact numbers: "walk slowly" cares about "forward" and "not the
    fastest", not 0.2 vs 0.25; too strict a condition fails reasonable answers and tests the wrong thing
    directions are checked by sign (left positive, right negative; forward positive, backward negative)

Sets
    DEV        20 commands used while writing the prompt; 60/60 correct (3 runs each) with the English prompt
    HOLDOUT_1  6 sentences never seen while tuning; measured once: 14/18. Its one backing-up failure led to
               "backward is expanded by the program" (planner/reverse.py), so it is used up
    HOLDOUT_2  6 new sentences the owner would really say, answers fixed before the single run: 11/18
               ("Come here." got stay_close, which reaches the owner but the check expected a plain walk -> 14/18 if
               counted; "Turn left -- no, wait, turn right." is a known limitation)
A held-out set is measured once; after its results influence a change, it becomes part of development.

    python -m eval.commands --check              # format only, no LLM
    python -m eval.commands --set dev -n 3       # needs OPENAI_API_KEY
"""

import argparse

from .plan_checks import (check_format, distance, ends, evaluate, head_moved, head_yaw, no_backward, no_left_turn,
                          no_merge, no_move, no_walk, not_, ok, order, refused, sit, stand, total_turn, turn_around, walk)

DEV = [
    # ── basics: the most common phrasings; direction, duration, posture ──
    ("Walk forward for two seconds.", "standing",
     [ok(), walk(forward=">0", secs="==2"), ends("standing")]),
    # The pet gait cannot walk backward (-0.07 m/s at most) -> "back up" = turn around 180 deg and walk (like a dog)
    ("Back up for two seconds.", "standing",
     [ok(), turn_around(), walk(forward=">0", secs="==2"), no_backward(), no_merge(), ends("standing")]),
    ("Sit down.", "standing",
     [ok(), sit(), no_walk(), ends("sitting")]),                         # no extra walking
    ("Stand up.", "sitting",
     [ok(), stand(), no_walk(), ends("standing")]),

    # ── dead zone: small amounts must still really move (direction checked) ──
    ("Walk forward slowly.", "standing",
     [ok(), walk(forward=">0"), walk(forward="<0.4")]),                  # forward, but not the fastest
    ("Back up just a little.", "standing",
     [ok(), turn_around(), walk(forward=">0"), no_backward()]),
    ("Turn right a little.", "standing",
     [ok(), total_turn("<0"), no_merge()]),                              # right = negative

    # ── multi-step order: "then" must become separate steps, in order, never turning while walking ──
    ("Stand up, turn left, then walk forward for two seconds.", "sitting",
     [ok(), order("stand", "turn", "go"), no_merge(), total_turn(">0")]),
    ("Walk forward for three seconds, then turn right, then walk forward for two more seconds.", "standing",
     [ok(), order("go", "turn", "go"), no_merge(), total_turn("<0")]),
    ("Sit down first, then stand up, then back up.", "standing",
     [ok(), order("sit", "stand", "turn", "go"), turn_around(), walk(forward=">0"), no_backward(), ends("standing")]),
    ("Turn 180 degrees to the left.", "standing",
     [ok(), total_turn(">=150"), total_turn("<=210"), no_merge()]),      # the LLM must compute 180 / 45 = 4 s

    # ── head: left / right signs, head and body together ──
    ("Sit down, and look to the left while sitting.", "standing",
     [ok(), sit(head_yaw=">0"), ends("sitting")]),
    ("Look to your right.", "standing",
     [ok(), head_yaw("<0"), ends("standing")]),
    ("Look left while walking.", "standing",
     [ok(), walk(forward=">0"), head_yaw(">0")]),

    # ── beyond the limits: clamp to the limit, never flip the direction ──
    ("Dash over as fast as you can.", "standing",
     [ok(), walk(forward=">=0.35")]),
    ("Turn your head 150 degrees to the left.", "standing",
     [ok(), head_yaw(">=60")]),                                          # 150 is impossible -> as far left as possible (80)

    # ── sitting start: preconditions must add "stand up" ──
    ("Walk forward.", "sitting",
     [ok(), order("stand", "go"), walk(forward=">0")]),
    ("Turn half a circle to the right.", "sitting",
     [ok(), order("stand", "turn"), total_turn("<=-150"), no_merge()]),

    # ── pet: the owner comes home -> stand up and swing the head both ways (happy) ──
    ("I'm home!", "sitting",
     [ok(), stand(), head_yaw(">0"), head_yaw("<0")]),

    # ── impossible: refuse explicitly, do not move ──
    ("Fly!", "standing",
     [ok(), refused(), no_move()]),
]

HOLDOUT_1 = [
    ("Turn around in a full circle.", "standing",
     [ok(), total_turn(">=300"), total_turn("<=420"), no_merge()]),                 # "a circle" = 360
    ("First turn right 90 degrees, then back up for two seconds, then sit down.", "standing",
     [ok(), order("turn", "go", "sit"), turn_around(), walk(forward=">0"), no_backward(), no_merge()]),
    ("Look behind you to the left.", "standing",
     [ok(), head_yaw(">0")]),
    ("You're so cute.", "sitting",
     [ok(), head_yaw(">0"), head_yaw("<0")]),                                       # praise = happy
    ("I'm so tired today.", "standing",
     [ok(), sit(), ends("sitting")]),
    ("Jump.", "standing",
     [ok(), refused()]),                                                            # no jumping action
]

# Typed mode convention: the owner stands straight in front of the duck -> "face the owner" = no turning, no walking away
HOLDOUT_2 = [
    ("Walk forward half a meter.", "standing",
     [ok(), distance(">=0.4"), distance("<=0.6"), no_backward()]),                 # distance -> time; 0.4-0.6 m accepted
    ("Come here.", "sitting",
     [ok(), walk(forward=">0"), total_turn("==0")]),                                # a pet command, from sitting
    ("Turn left — no, wait, turn right.", "standing",
     [ok(), total_turn("<0"), no_left_turn()]),                                     # correction: do only the last one
    ("Could you sit down for me?", "standing",
     [ok(), sit(), ends("sitting"), not_(refused())]),                              # a question is still a command
    ("Look at me.", "standing",
     [ok(), head_moved(), no_move()]),                                              # head up or swinging, body still
    ("Climb onto the sofa.", "standing",
     [ok(), refused()]),                                                            # impossible -> refuse
]

SETS = {"dev": DEV, "holdout1": HOLDOUT_1, "holdout2": HOLDOUT_2}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="dev", choices=list(SETS))
    ap.add_argument("--check", action="store_true", help="format only, no LLM")
    ap.add_argument("-n", type=int, default=3)
    args = ap.parse_args()
    if args.check:
        raise SystemExit(sum(check_format(t) for t in SETS.values()))
    evaluate(SETS[args.set], args.n)
