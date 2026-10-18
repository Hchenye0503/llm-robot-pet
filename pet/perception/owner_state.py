"""Owner state: the VLM looks at one photo and fills in only what is visible.

    1. Observable facts only (operational definitions): posture, what the arms do, head down or not,
       face toward the duck or not. "Angry" or "tired" are not fields: hands on hips may just be resting.
       Interpretation is left to the planning LLM, together with what the owner says.
    2. Events are not fields: "just came home" cannot be seen in one photo (they might be leaving).
       An event is a change between states, so the program detects it (events.py).
    3. Name calls and scolding come from the microphone, petting from a touch sensor: not camera fields.
    4. Every field allows "unknown": unclear means unclear, never a guess.
    5. The descriptions are the operational definitions and are sent to the VLM as instructions.

    python -m pet.perception.owner_state
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class OwnerState(BaseModel):
    """The owner as seen in one photo from the duck's head camera."""
    model_config = ConfigDict(extra="forbid")

    present: bool = Field(description="The owner is visible in the frame (a human figure, at least the torso or both legs visible)")
    posture: Literal["standing", "walking", "sitting", "lying", "unknown"] = Field(
        description="standing = legs straight, torso upright; walking = upright with the feet apart, one in front of the "
                    "other; sitting = bottom on the floor, torso upright or leaning forward; lying = torso horizontal on "
                    "the floor; not in the frame or unclear = unknown")
    arms: Literal["resting", "waving", "on_hips", "pointing", "open", "hugging_knees", "holding_toy", "unknown"] = Field(
        # The priority rule: seen from below, a toy held in both hands looks like hands above the head and
        # was read as waving in 5/8 frames.
        description="[PRIORITY] If there is a yellow cube in a hand or between the hands -> holding_toy (no matter how "
                    "high the hands are or from what angle). Otherwise: resting = hands visible but not doing any of "
                    "the gestures below (e.g. at the sides, on the lap or knees, or lying on the floor or body); waving = only one hand raised above the head, the other hand down, nothing in the "
                    "hand; on_hips = both hands on the waist, elbows out; pointing = one arm stretched straight forward; "
                    "open = both arms spread out to the sides; hugging_knees = both arms wrapped around the knees; "
                    "unclear = unknown")
    head: Literal["up", "down", "unknown"] = Field(
        description="down = head clearly hanging forward, face toward the floor; up = otherwise; unclear = unknown")
    facing_duck: Literal["yes", "no", "unknown"] = Field(
        description="yes = the owner's eyes and nose are visible, face toward the camera; no = back of the head or a "
                    "side view; unclear = unknown")

    @model_validator(mode="after")
    def _consistent(self):
        # Not present -> nothing else can be known (no "I don't see them, and they are waving").
        if not self.present and (self.posture, self.arms, self.head, self.facing_duck) != ("unknown",) * 4:
            raise ValueError("present=false requires posture / arms / head / facing_duck to be unknown")
        if self.posture == "lying" and self.arms in ("waving", "on_hips", "pointing", "open"):
            raise ValueError(f"arms={self.arms} is impossible while lying (these are standing gestures)")
        if self.arms == "hugging_knees" and self.posture not in ("sitting", "unknown"):
            raise ValueError("hugging_knees is only possible while sitting")
        return self


# Visible cue -> *possible* emotion, a hint for the planning LLM, never a fact.
# tired and sad are merged: the VLM cannot tell a hanging head from hugging knees, and both call for
# the same response.
CUES = {"lying": "may be tired or sad", "head_down": "may be tired or sad", "on_hips": "may be angry", "pointing": "may be angry",
        "waving": "may be happy", "open": "may be welcoming", "hugging_knees": "may be tired or sad"}


def openai_schema() -> dict:
    """Strict mode: every field required (OwnerState has no defaults)."""
    s = OwnerState.model_json_schema()
    s["additionalProperties"] = False
    s["required"] = list(s["properties"])
    return s


def selftest() -> int:
    ok_cases = [
        dict(present=True, posture="sitting", arms="resting", head="up", facing_duck="yes"),
        dict(present=True, posture="standing", arms="on_hips", head="up", facing_duck="no"),
        dict(present=True, posture="sitting", arms="hugging_knees", head="down", facing_duck="unknown"),
        dict(present=False, posture="unknown", arms="unknown", head="unknown", facing_duck="unknown"),
    ]
    bad_cases = [
        ("not present but waving", dict(present=False, posture="unknown", arms="waving", head="unknown", facing_duck="unknown")),
        ("lying with hands on hips", dict(present=True, posture="lying", arms="on_hips", head="up", facing_duck="no")),
        ("standing and hugging knees", dict(present=True, posture="standing", arms="hugging_knees", head="down", facing_duck="no")),
        ("invented posture", dict(present=True, posture="dancing", arms="resting", head="up", facing_duck="yes")),
        ("extra field emotion", dict(present=True, posture="sitting", arms="resting", head="up", facing_duck="yes", emotion="angry")),
        ("missing field", dict(present=True, posture="sitting", arms="resting", head="up")),
    ]
    n = 0
    for c in ok_cases:
        try:
            OwnerState(**c)
            n += 1
            print(f"ok  valid: {c}")
        except Exception as e:
            print(f"BAD should be valid: {c}  {e}")
    for why, c in bad_cases:
        try:
            OwnerState(**c)
            print(f"BAD should be rejected ({why})")
        except Exception:
            n += 1
            print(f"ok  rejected: {why}")
    s = openai_schema()
    strict_ok = s["required"] == list(s["properties"]) and s["additionalProperties"] is False
    n += strict_ok
    print(f"{'ok ' if strict_ok else 'BAD'} OpenAI strict schema: all {len(s['properties'])} fields required")
    total = len(ok_cases) + len(bad_cases) + 1
    print(f"{n}/{total}")
    return total - n


if __name__ == "__main__":
    raise SystemExit(selftest())
