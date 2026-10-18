"""Behavior and gesture catalog: the names the LLM may pick, and what each one means.

Behaviors come from the ethogram (docs/ethogram.md; the id is the ethogram row). The body side that
actually performs them lives elsewhere; this file is data only, so the planner can import it without
pulling in the simulator.
"""

# name -> (ethogram id, description shown to the LLM, must be standing first)
BEHAVIORS = {
    "greet": ("#5", "greeting: trot up to the owner, look up and wiggle the head happily (when the owner comes home)", True),
    "stay_close": ("#7", "stay close: walk next to the owner, sit down and look up at them (when the owner is tired, sad, or you want company)", True),
    "gaze": ("#1", "gaze at the owner: turn the head toward the owner and look up slightly (when called by name, when the owner is talking)", False),
    "play_bow": ("#10", "play invitation: face the owner, bow deeply and bounce back and forth (when the owner asks to play or brings out a toy)", True),
    "zoomies": ("#14", "zoomies: run around fast, spin hops, then suddenly sit down (when super excited)", True),
    "guilty": ("#18", "guilty: lower the head and look away, slowly approach the owner and sit down (when scolded)", True),
    "investigate": ("#21", "investigate: walk to the toy on the floor, look down at it, then look left and right (when curious or seeing something new)", True),
    "rest": ("#22", "rest: sit down and slowly lower the head (when tired or there is nothing to do)", False),
}

# Head-only gestures, played from keyframes in the current posture. spin_hop is the exception:
# it is a separate trained policy and needs standing.
GESTURES = {
    "nod": "nod twice (substitute for a slow blink: agreement / affection)",
    "head_tilt": "tilt the head (hearing its name, curious)",
    "wag": "head wag (substitute for a tail wag): swing left and right",
    "happy": "happy: look up 25° + wiggle the head (swinging left and right while tilting to the same side)",
    "look_around": "look left and right (exploring: curious)",
    "shy": "lower the head and look away (guilty look: scolded)",
    "spin_hop": "very excited: spin hop in place for 1.5 seconds (must be standing)",
}

# Actions that cost energy. The label goes into the description the LLM sees, and the pet prompt
# says "low energy -> pick a (low effort) one", so adding a new behavior only needs a label here.
# Walking behaviors (greet, stay_close) are low effort on purpose: a tired pet still greets its owner.
HIGH_EFFORT = {"play_bow", "zoomies", "spin_hop"}


def effort(name: str) -> str:
    # "(light)" was ambiguous (easy vs. brisk) and the LLM sent a tired pet walking to a toy; "effort" is not.
    return "(high effort) " if name in HIGH_EFFORT else "(low effort) "


# Posture at the end of a behavior; missing = unchanged (gaze while sitting stays sitting).
# Precondition bookkeeping and the evaluators both read this, so they always agree.
BEH_ENDS = {"stay_close": "sit", "guilty": "sit", "rest": "sit", "zoomies": "sit",
            "greet": "stand", "play_bow": "stand", "investigate": "stand"}


def end_posture(name: str) -> str | None:
    """"sit" / "stand" after behavior `name`, or None if it does not change posture."""
    return BEH_ENDS.get(name)
