"""Ask the LLM, validate, rewrite once if rejected, otherwise degrade.

    sentence
      -> ask_llm (structured output; ranges written in the descriptions, not enforced by the API)
      -> process: schema.Plan -> reverse -> preconditions -> translate (with apply_limit)
      -> plan_with_retry: rejected -> send the error back and ask for a rewrite -> rejected again -> degrade

    structure error / number rejected   -> attach the error, ask for a rewrite, at most MAX_RETRIES times
    still wrong after the rewrite       -> degrade
    the LLM call itself fails (offline, timeout, bad key) -> degrade at once, no rewrite
    start posture cannot be planned (fallen) -> degrade without asking the LLM

Degrading means holding the current posture and sending nothing new: a fault should move the robot as
little as possible (a sitting duck stays sitting). "Idle" is product behavior; "degraded" is fault behavior.

The selftest uses a scripted fake LLM (a mock), so it needs no network and costs nothing.

    python -m pet.planner.llm --selftest
"""

import copy
import json
from dataclasses import dataclass, field

import httpx
import openai
from pydantic import ValidationError

from .limits import RangeReject
from .preconditions import INSERTED, POSTURES, insert_preconditions
from .prompts import SYSTEM
from .reverse import expand_backward
from .schema import Behavior, Gesture, Kick, Plan, Refuse, SitStand, Walk
from .translate import translate_plan

MODEL = "gpt-5.4-mini"
TIMEOUT_S = 10.0
MAX_RETRIES = 1  # rewrites after the first answer -> at most 2 LLM calls


def openai_schema() -> dict:
    """Plan's JSON Schema in the shape OpenAI's strict mode accepts.

    Strict mode rejects oneOf (what a pydantic discriminated union produces), so it becomes anyOf and
    every field is required. Ranges are moved into the description text instead of minimum/maximum:
    with enforced ranges, 9/9 deliberately out-of-range requests ("turn left 150 degrees") came back
    legal but wrong (wrong direction or 0), and our limits passed them silently; with the ranges only in
    the text, 9/9 kept the right direction. The reply is validated again with our Plan regardless:
    the API's format guarantee is not our validator.
    """
    def fix(node):
        if isinstance(node, dict):
            if "oneOf" in node:
                node["anyOf"] = node.pop("oneOf")
            node.pop("discriminator", None)
            node.pop("default", None)
            if "minimum" in node or "maximum" in node:
                lo, hi = node.pop("minimum", None), node.pop("maximum", None)
                node["description"] = f"{node.get('description', '')} (legal range {lo} ~ {hi})"
            if node.get("type") == "object" and "properties" in node:
                node["required"] = list(node["properties"])
                node["additionalProperties"] = False
            for v in node.values():
                fix(v)
        elif isinstance(node, list):
            for v in node:
                fix(v)
        return node

    return fix(copy.deepcopy(Plan.model_json_schema()))


@dataclass
class Result:
    status: str                                   # "ok" or "degraded"
    commands: list = field(default_factory=list)  # [(policy, 13 numbers), ...]
    log: list = field(default_factory=list)       # clamps, insertions, rejections, degradation
    llm_calls: int = 0
    steps: list = field(default_factory=list)     # the steps behind `commands` (the scheduler reads duration_s)


def degrade(reason: str, log: list, llm_calls: int) -> Result:
    return Result("degraded", [], log + [f"degraded: {reason}"], llm_calls, [])


def process(raw: str, start: str) -> tuple[list, list, list]:
    """The LLM's raw text -> (commands, log, steps). Raises ValidationError (structure),
    RangeReject (numbers) or PreconditionReject (start posture)."""
    plan = Plan.model_validate_json(raw)
    log: list = []
    steps = insert_preconditions(expand_backward(plan.steps), start, log)
    commands, log2 = translate_plan(Plan.model_construct(steps=steps))
    return commands, log + log2, steps


def plan_text(steps) -> str:
    """Plan -> one subtitle line. [auto] marks steps inserted by preconditions, not chosen by the LLM."""
    parts = []
    for s in steps:
        if isinstance(s, Walk):
            bits = []
            if s.forward_mps:
                bits.append(f"{s.forward_mps:g} m/s")
            if s.turn_left_deg_s:
                bits.append(f"turn {s.turn_left_deg_s:g} deg/s")
            parts.append(f"walk({', '.join(bits) or 'stay'}, {s.duration_s:g} s)")
        elif isinstance(s, SitStand):
            auto = "[auto] " if any(s is x for x in INSERTED) else ""
            parts.append(f"{auto}sitstand({s.posture})")
        elif isinstance(s, Behavior):
            parts.append(f"behavior({s.name})")
        elif isinstance(s, Gesture):
            parts.append(f"gesture({s.name})")
        elif isinstance(s, Kick):
            parts.append(f"kick({s.target})")
        elif isinstance(s, Refuse):
            parts.append(f"refuse(\"{s.reason}\")")
    return "LLM → " + " → ".join(parts) if parts else "LLM → (no action)"


def ask_llm(messages: list) -> str:
    """Call OpenAI and return the filled-in JSON text. Reads OPENAI_API_KEY from the environment.

    max_retries=0 turns off the SDK's own retries: plan_with_retry owns the retry count. Otherwise the
    SDK's 2 silent retries x our 1 rewrite could mean up to 6 calls, with latency and cost out of control.
    """
    client = openai.OpenAI(timeout=TIMEOUT_S, max_retries=0)
    r = client.responses.create(
        model=MODEL,
        input=messages,
        text={"format": {"type": "json_schema", "name": "Plan", "schema": openai_schema(), "strict": True}},
    )
    return r.output_text


def plan_with_retry(sentence: str, start: str, ask=ask_llm, system: str = SYSTEM) -> Result:
    """system: prompts.SYSTEM for typed commands, prompts.SYSTEM_PET for perception-triggered sentences."""
    if start not in POSTURES:
        return degrade(f"cannot plan from start posture {start!r}", [], 0)

    messages = [{"role": "system", "content": system}, {"role": "user", "content": sentence}]
    log: list = []
    calls = 0
    for _ in range(1 + MAX_RETRIES):
        try:
            raw = ask(messages)
            calls += 1                     # after the call: a call that never answered is not counted
        except openai.APIError as e:
            return degrade(f"LLM call failed: {e}", log, calls)
        try:
            commands, plog, steps = process(raw, start)
            return Result("ok", commands, log + plog, calls, steps)
        except (ValidationError, RangeReject) as e:
            log.append(f"rejected on call {calls}: {e}")
            # the LLM must see its own answer and what was wrong with it
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": f"The previous plan was rejected: {e}. "
                                                        "Please fix it and output the complete plan again."})
    return degrade(f"still rejected after {MAX_RETRIES} rewrite(s)", log, calls)


# ─────────────────────────────────────────────────────────────
# selftest: a scripted fake LLM, no network, no cost
# ─────────────────────────────────────────────────────────────
def _walk(v):
    return json.dumps({"steps": [{"action": "walk", "forward_mps": v, "turn_left_deg_s": 0,
                                  "head": {"yaw_left_deg": 0, "pitch_deg": 0}, "duration_s": 2}]})


GOOD, BAD5, FLY = _walk(0.2), _walk(5.0), json.dumps({"steps": [{"action": "fly"}]})
NAN = '{"steps": [{"action": "walk", "forward_mps": NaN, "turn_left_deg_s": 0, "head": {"yaw_left_deg": 0, "pitch_deg": 0}, "duration_s": 2}]}'
NET = "offline"


def fake_llm(script: list):
    """A fake `ask`: call 1 returns script[0], call 2 returns script[1], ...; NET raises a connection error."""
    seen = []

    def ask(messages):
        seen.append([dict(m) for m in messages])
        reply = script[len(seen) - 1]
        if reply == NET:
            raise openai.APIConnectionError(request=httpx.Request("POST", "https://api.openai.com"))
        return reply

    ask.seen = seen
    return ask


CASES = [
    # (description, start, script, expected status, expected calls)
    ("right the first time", "standing", [GOOD], "ok", 1),
    ("writes 5 -> rejected -> rewrites 0.2", "standing", [BAD5, GOOD], "ok", 2),
    ("structure error (fly) -> rewrite", "standing", [FLY, GOOD], "ok", 2),
    ("NaN -> rejected -> rewrite", "standing", [NAN, GOOD], "ok", 2),
    ("writes 5 twice -> degrade", "standing", [BAD5, BAD5], "degraded", 2),
    ("offline -> degrade at once, no rewrite", "standing", [NET], "degraded", 0),
    ("rejected, then offline on the rewrite -> degrade", "standing", [BAD5, NET], "degraded", 1),
    ("fallen -> degrade without asking", "fallen", [GOOD], "degraded", 0),
    ("sitting, LLM says walk -> stand up inserted", "sitting", [GOOD], "ok", 1),
]


def selftest() -> int:
    n_bad = 0
    for desc, start, script, exp_status, exp_calls in CASES:
        ask = fake_llm(script)
        res = plan_with_retry("(test)", start, ask=ask)
        ok = res.status == exp_status and res.llm_calls == exp_calls
        if ok and exp_status == "ok" and exp_calls == 2:      # the rewrite request carries the answer and the error
            second = ask.seen[1]
            ok = (len(second) == 4 and second[2]["role"] == "assistant"
                  and second[2]["content"] == script[0] and "rejected" in second[3]["content"])
        if ok and exp_status == "degraded":                   # degraded = nothing new is sent
            ok = res.commands == [] and res.steps == []
        n_bad += not ok
        print(f"{'ok ' if ok else 'BAD'} {desc}\n     status={res.status} calls={res.llm_calls} "
              f"(want {exp_status}, {exp_calls})  log: {res.log}")
    print(f"\n{len(CASES) - n_bad}/{len(CASES)} as expected")
    return n_bad


if __name__ == "__main__":
    import sys
    raise SystemExit(selftest()) if "--selftest" in sys.argv else SystemExit("usage: python -m pet.planner.llm --selftest")
