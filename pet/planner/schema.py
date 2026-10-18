"""The form the LLM fills in: a plan is a list of typed steps (a discriminated union on "action").

This schema is written for the LLM, not for the policies: units it understands (degrees, m/s,
"sit"/"stand"), with the direction in the field name (yaw_left_deg). Turning it into the 13 numbers a
policy reads is translate.py's job.

Note: pydantic puts class docstrings and Field descriptions into the JSON Schema, so everything in
quotes below is text the LLM reads. Development notes are kept in comments.

    python -m pet.planner.schema            selftest
    python -m pet.planner.schema --schema   print the JSON Schema
"""

import argparse
import json
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .catalog import BEHAVIORS, GESTURES, effort

# Command ranges seen in training (not keyboard limits), rounded down after converting to degrees.
#   vx ±0.4 m/s, wz ±1.0 rad/s, head_yaw ±1.40 rad, head_pitch ±1.10 rad  (microduck velocity / sitstand env cfg)
# These go into the schema so the LLM sees them, but pydantic does NOT enforce them: limits.apply_limit
# decides clamp vs. reject. With Field(ge=, le=) a slightly-too-large 0.41 would be rejected outright and
# the "small overshoot -> clamp" rule could never run.
LIMITS = {
    "forward_mps": 0.4,
    "turn_left_deg_s": 57,
    "yaw_left_deg": 80,
    "pitch_deg": 63,
}

# Dead zone: below these the duck does not move. (positive-direction minimum, |negative-direction minimum|)
# Measured on the deployed pet gait (seed 42): forward moves from 0.05; turning starts at +0.5 / -0.3 rad/s,
# i.e. 29 deg/s left and 17 deg/s right (asymmetric, as measured). The gait cannot really walk backward
# (max -0.07 m/s), which is why reverse.py turns "back up" into turn-walk-turn.
DEADZONE = {
    "forward_mps": (0.05, 0.125),
    "turn_left_deg_s": (29, 17),
}


def limited(name: str, description: str):
    """A symmetric-range field: the LLM sees the range; apply_limit handles overshoot."""
    lim = LIMITS[name]
    return Field(0.0, description=description, json_schema_extra={"minimum": -lim, "maximum": lim})


class _Strict(BaseModel):
    # extra="forbid": a sit step carrying a leftover forward_mps is an error, not silently ignored.
    model_config = ConfigDict(extra="forbid")


class Head(_Strict):
    """Head channel. Deltas from the home pose, in degrees."""

    # Signs checked in simulation: +yaw turns left; +pitch (head_pitch joint) tilts the head DOWN.
    yaw_left_deg: float = limited("yaw_left_deg", "positive = look left, negative = look right")
    pitch_deg: float = limited("pitch_deg", "positive = head down (look down), negative = head up (look up)")


class Walk(_Strict):
    """Walking (velocity policy)."""

    action: Literal["walk"]
    # The dead zone is written into the description so the LLM picks values that move;
    # limits.snap_deadzone is the safety net when it does not.
    forward_mps: float = limited("forward_mps", "Forward speed in m/s, negative = back up (the program turns around "
                                 "and walks back). To move, at least 0.05 (smaller values do not move the duck); 0 if not walking")
    turn_left_deg_s: float = limited("turn_left_deg_s", "Turning rate in place, degrees/s, positive = turn left. To turn, "
                                     "at least 29 to the left or 17 to the right in absolute value (smaller values do not "
                                     "turn); 0 if not turning")
    head: Head = Head()
    # 60 s is a product decision, not a training limit: after a minute, come back and ask.
    duration_s: float = Field(gt=0, le=60, description="How many seconds this action lasts")


class SitStand(_Strict):
    action: Literal["sitstand"]
    posture: Literal["sit", "stand"]
    head: Head = Head()


class Kick(_Strict):
    # The LLM only names the target; aiming is done by the program.
    action: Literal["kick"]
    target: Literal["owner", "forward"]


class Refuse(_Strict):
    """For things that cannot be done (flying, jumping, picking things up...), so the LLM can clearly say "I can't" instead of forcing a plan."""

    action: Literal["refuse"]
    reason: str = Field(description="Tell the user in one English sentence why it cannot be done")


class Gesture(_Strict):
    """Pet gesture. The LLM only picks the name (whitelist); the keyframes are predefined.
    Done in the current posture (standing if standing, sitting if sitting), so there are no preconditions and no head field."""

    action: Literal["gesture"]
    name: Literal[tuple(GESTURES)] = Field(description="; ".join(f"{k}={effort(k)}{v}" for k, v in GESTURES.items()))


class Behavior(_Strict):
    """Pet behavior (from the ethogram). The body moves and faces the owner or the toy; the LLM only picks the name and the program expands it."""

    action: Literal["behavior"]
    name: Literal[tuple(BEHAVIORS)] = Field(description="; ".join(f"{k}={effort(k)}{v[1]}" for k, v in BEHAVIORS.items()))


Action = Annotated[Union[Walk, SitStand, Kick, Refuse, Gesture, Behavior], Field(discriminator="action")]


class Plan(_Strict):
    """The whole plan the LLM returns at once."""

    steps: list[Action] = Field(min_length=1, max_length=5)

    @model_validator(mode="after")
    def refuse_alone(self):
        """Refusing plus doing something else is contradictory: refuse must be the only step."""
        if any(isinstance(s, Refuse) for s in self.steps) and len(self.steps) > 1:
            raise ValueError("refuse must be the only step in the plan")
        return self


# ─────────────────────────────────────────────────────────────
# selftest: each case states whether it should pass and why
# ─────────────────────────────────────────────────────────────
CASES = [
    # (name, LLM output, should pass)
    ("walk + look left", {"steps": [{"action": "walk", "forward_mps": 0.2, "head": {"yaw_left_deg": 30}, "duration_s": 2}]}, True),
    # ranges are not this layer's job (limits.apply_limit) -> structurally fine
    ("too fast 1.0 (structure ok)", {"steps": [{"action": "walk", "forward_mps": 1.0, "duration_s": 2}]}, True),
    ("head 1833 deg (structure ok)", {"steps": [{"action": "walk", "head": {"yaw_left_deg": 1833}, "duration_s": 2}]}, True),
    ("no duration", {"steps": [{"action": "walk", "forward_mps": 0.2}]}, False),
    ("unknown action", {"steps": [{"action": "fly", "duration_s": 2}]}, False),
    ("sit", {"steps": [{"action": "sitstand", "posture": "sit"}]}, True),
    ("sit with leftover speed", {"steps": [{"action": "sitstand", "posture": "sit", "forward_mps": 0.3}]}, False),
    ("posture typo", {"steps": [{"action": "sitstand", "posture": "sitt"}]}, False),
    ("walk 2 s then sit", {"steps": [
        {"action": "walk", "forward_mps": 0.2, "duration_s": 2},
        {"action": "sitstand", "posture": "sit"}]}, True),
    ("empty plan", {"steps": []}, False),
    ("kick to owner", {"steps": [{"action": "kick", "target": "owner"}]}, True),
    ("kick forward", {"steps": [{"action": "kick", "target": "forward"}]}, True),
    ("kick with head", {"steps": [{"action": "kick", "target": "owner", "head": {"yaw_left_deg": 30}}]}, False),
    ("kick at the ceiling", {"steps": [{"action": "kick", "target": "ceiling"}]}, False),
    ("LLM computes the angle", {"steps": [{"action": "kick", "target": "owner", "turn_left_deg": 37}]}, False),
    ("refuse", {"steps": [{"action": "refuse", "reason": "I can't fly"}]}, True),
    ("refuse and walk", {"steps": [{"action": "refuse", "reason": "I can't fly"},
                                   {"action": "walk", "forward_mps": 0.3, "duration_s": 2}]}, False),
    ("refuse without reason", {"steps": [{"action": "refuse"}]}, False),
    ("wag", {"steps": [{"action": "gesture", "name": "wag"}]}, True),
    ("stand then tilt head", {"steps": [{"action": "sitstand", "posture": "stand"}, {"action": "gesture", "name": "head_tilt"}]}, True),
    ("unknown gesture", {"steps": [{"action": "gesture", "name": "dance"}]}, False),
    ("greet", {"steps": [{"action": "behavior", "name": "greet"}]}, True),
    ("unknown behavior", {"steps": [{"action": "behavior", "name": "fetch"}]}, False),
]


def selftest() -> int:
    n_bad = 0
    for name, payload, should_pass in CASES:
        try:
            Plan.model_validate(payload)
            passed, why = True, ""
        except ValidationError as e:
            passed = False
            err = e.errors()[0]
            why = f"{'.'.join(map(str, err['loc']))}: {err['msg']}"
        ok = passed == should_pass
        n_bad += not ok
        print(f"{'ok ' if ok else 'BAD'} {name:<28} got={'pass' if passed else 'reject'} "
              f"want={'pass' if should_pass else 'reject'}  {why}")
    print(f"\n{len(CASES) - n_bad}/{len(CASES)} as expected")
    return n_bad


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", action="store_true", help="print the JSON Schema the LLM sees")
    args = ap.parse_args()
    if args.schema:
        print(json.dumps(Plan.model_json_schema(), indent=2, ensure_ascii=False))
    else:
        raise SystemExit(selftest())
