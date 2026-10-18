"""Ask the LLM in the background, and re-check the premises before acting on the answer.

Stale observations are dropped; a stale *plan* is re-checked before it runs. In the synchronous version
the simulator stopped while the LLM thought, so the world never changed and the problem was hidden.
In the background, the owner may put the toy down and walk away during those 2 seconds, and the duck
would invite empty floor to play.

Which changes invalidate a plan? A plan is based on the facts written into its sentence: its premises.
Same facts -> go ahead; any fact different -> ask again.
    the owner leaves (a new event)                 -> ask again (every behavior is about the owner)
    energy 0.80 -> 0.78 (still "high")             -> go ahead: the sentence did not change
    energy crosses a level (medium -> low)         -> ask again: the sentence changed
    the owner says something new                   -> ask again
    detector motion, exact positions               -> not in the sentence, not a premise

    python -m pet.mind.async_plan
"""

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .inner import level

MAX_REASK = 2       # if premises keep changing, give up after this many re-asks (or the duck waits forever)


def premise(state: dict, inner=None, events_seen: int = 0, heard: int = 0, duck_posture: str = "") -> dict:
    """Snapshot of the facts written into the sentence sent to the LLM.

    `state` must be the debounced state (EventDetector.confirmed), not one raw VLM answer. facing_duck is
    not debounced and flickered while the owner walked away with their back turned; with it as a premise
    both re-asks were discarded and the duck did nothing. It is still written into the sentence."""
    state = state or {}
    p = {
        "present": state.get("present", state.get("posture") is not None),
        "posture": state.get("posture"),
        "arms": state.get("arms"),
        "region": state.get("region"),
        "events_seen": events_seen,          # a new event (e.g. leaving) means the story moved on
        "heard": heard,                      # sentences the owner said (microphone): one more = a new command
        "duck_posture": duck_posture,        # the duck's own posture (preconditions were planned from it)
    }
    if inner is not None:
        p.update({k: level(v) for k, v in inner.d.x.items()})   # drives as levels, exactly as the sentence says them
    return p


def still_valid(then: dict, now: dict) -> bool:
    """The plan was made under `then`; the world is now `now`. Go ahead only if every premise is unchanged."""
    return all(then[k] == now.get(k) for k in then)


@dataclass
class Ask:
    fut: object
    sentence: str
    premise: dict
    t_sim: float
    wall0: float = field(default_factory=time.perf_counter)
    asks: int = 1
    wall_done: float | None = None


class AsyncPlanner:
    """Asks the LLM in the background; the answer counts as arrived at sim time = sent time + real seconds taken."""

    def __init__(self, ask_fn):
        self.ask_fn = ask_fn                 # ask_fn(sentence, posture) -> Result (wraps llm.plan_with_retry)
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.pending: Ask | None = None
        self.log = []

    def submit(self, sentence, posture, prem, t_sim, asks=1):
        a = Ask(self.pool.submit(self.ask_fn, sentence, posture), sentence, prem, t_sim, asks=asks)
        a.fut.add_done_callback(lambda f, a=a: setattr(a, "wall_done", time.perf_counter()))
        self.pending = a
        self.log.append(f"t={t_sim:5.1f} ask LLM (try {asks}): {sentence[:70]}...")

    def poll(self, t_sim, current_premise, current_sentence, posture):
        """Return the plan to execute (Result) or None. Premises changed -> ask again with the current sentence."""
        a = self.pending
        if a is None or not a.fut.done() or a.wall_done is None:
            return None
        if t_sim + 1e-9 < a.t_sim + (a.wall_done - a.wall0):
            return None                      # in real time the answer has not arrived yet
        res = a.fut.result()
        self.pending = None
        if still_valid(a.premise, current_premise):
            self.log.append(f"t={t_sim:5.1f} plan back, premises unchanged -> execute")
            return res
        changed = [k for k in a.premise if a.premise[k] != current_premise.get(k)]
        if a.asks > MAX_REASK:
            self.log.append(f"t={t_sim:5.1f} plan back, premises changed ({changed}), re-asked {MAX_REASK} times -> give up (do nothing)")
            return None
        self.log.append(f"t={t_sim:5.1f} plan back, premises changed ({changed}) -> discard, ask again")
        self.submit(current_sentence, posture, current_premise, t_sim, a.asks + 1)
        return None

    def close(self):
        self.pool.shutdown(wait=True)


def selftest() -> int:
    ok = n = 0

    def check(desc, good, got=""):
        nonlocal ok, n
        ok, n = ok + bool(good), n + 1
        print(f"{'ok ' if good else 'BAD'} {desc}" + (f"\n     {got}" if got else ""))

    from types import SimpleNamespace
    drives = lambda e: SimpleNamespace(d=SimpleNamespace(x={"E": e, "C": 0.5, "A": 0.5, "S": 0.1}))
    obs = dict(present=True, posture="standing", arms="holding_toy", region="living_room", facing_duck="yes")
    inner = drives(0.80)
    then = premise(obs, inner, events_seen=3, heard=1, duck_posture="sitting")
    check("energy 0.80 -> 0.78 (both high) -> go ahead", still_valid(then, premise(obs, drives(0.78), 3, 1, "sitting")))
    check("energy drops to low (crosses a level, sentence changes) -> ask again", not still_valid(then, premise(obs, drives(0.25), 3, 1, "sitting")))
    check("owner puts the toy down (arms changed) -> ask again", not still_valid(then, premise(dict(obs, arms="resting"), inner, 3, 1, "sitting")))
    check("owner leaves (one more event) -> ask again", not still_valid(then, premise(dict(obs, posture="walking"), inner, 4, 1, "sitting")))
    check("owner says something new -> ask again", not still_valid(then, premise(obs, inner, 3, 2, "sitting")))
    check("nothing changed -> go ahead", still_valid(then, premise(obs, inner, 3, 1, "sitting")))

    # fake LLM that answers at once: premises changed -> re-ask -> unchanged -> execute
    calls = []
    p = AsyncPlanner(lambda s, pos: (calls.append(s), SimpleNamespace(plan=s))[1])
    p.submit("owner holds a toy", "sitting", then, 0.0)
    time.sleep(0.05)
    r1 = p.poll(1.0, premise(dict(obs, arms="resting"), inner, 4, 1, "sitting"), "owner walked away", "sitting")
    time.sleep(0.05)
    now = premise(dict(obs, arms="resting"), inner, 4, 1, "sitting")
    r2 = p.poll(2.0, now, "owner walked away", "sitting")
    p.close()
    check("premises changed -> first plan discarded, re-asked with the new sentence -> unchanged -> execute",
          r1 is None and r2 is not None and r2.plan == "owner walked away" and calls == ["owner holds a toy", "owner walked away"],
          f"asked {calls}")
    print(f"\n{ok}/{n}")
    return n - ok


if __name__ == "__main__":
    raise SystemExit(selftest())
