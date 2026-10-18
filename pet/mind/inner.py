"""Cue timers + inner drives: besides events, when should the LLM be called?

With events alone (came home, stood up, leaving, toy out) the owner could sit hugging their knees for a
minute and the duck would ignore them, and the same event always got the same reaction: more butler
than pet. A pet also *wants* things on its own.

    perception step (every 0.5 s): observation + events
      -> Inner.update(t, obs, events, dt) -> call the LLM or not, and what to add to the sentence
           1. an event                          -> call (as before)
           2. an emotion cue held for HOLD_S    -> call (CueTimer)
           3. an inner drive crosses a threshold -> call (Drives: the duck wants something)
         The sentence gets a "[self] ..." part, so the same event in a different state gets a different reaction.
      -> planner -> behavior -> Inner.done(behavior): doing the matching behavior satisfies the need

Design:
    Emotions are not facts: a visible cue -> a *possible* emotion, an operational definition we chose.
    A cue must last HOLD_S = 30 s (answering one message with the head down does not count); after a
      reaction the same cue is quiet for COOLDOWN_S = 120 s.
    Not seen is not over: present=False is skipped and the timer keeps running (while the duck walks
      over to keep the owner company, it often sees only their feet).
    Drives are homeostatic needs (the AIBO approach): unmet, they drift toward a baseline (a first-order
      system); the matching behavior lowers them. Energy works the other way: play costs it, rest and time restore it.
    The BASE / TAU / EFFECTS numbers are chosen starting values with reasons beside them, not measurements.

    python -m pet.mind.inner
"""

import math

HOLD_S = 30.0          # a cue must last this long before reacting
COOLDOWN_S = 120.0     # after reacting, the same cue waits this long
END_S = 12.0           # a cue must be unseen this long before it really ends (grace period)
# One VLM misread (head down read as head up) used to reset the 30 s timer. Ending needs time too, but
# less than starting: hysteresis, two thresholds for the two directions. The VLM re-looks every ~6 s and
# the answer is reused in between, so one misread lasts ~6 s.

# visible -> possible emotion (checked posture, then arms, then head)
# tired and sad are one cue, "low": the VLM confused a hanging head with hugging knees 9/22 times, the
# cue flipped between the two and never lasted 30 s, and both call for the same response.
CUE_POSTURE = {"lying": "low"}
CUE_ARMS = {"on_hips": "angry", "pointing": "angry", "hugging_knees": "low", "waving": "happy", "open": "welcoming"}
CUE_HEAD = {"down": "low"}
CUE_TEXT = {"low": "tired or sad"}     # wording for the LLM (others use the cue name)


def cue_of(obs: dict) -> str | None:
    """One observation -> a cue (possible emotion) or None. Cues are named by emotion, so switching
    between lying and head-down (both "low") does not restart the timer."""
    return CUE_POSTURE.get(obs["posture"]) or CUE_ARMS.get(obs["arms"]) or CUE_HEAD.get(obs["head"])


class CueTimer:
    """A cue must last hold_s before reacting; the same cue then waits cooldown_s; ending takes end_s."""

    def __init__(self, hold_s: float = HOLD_S, cooldown_s: float = COOLDOWN_S, end_s: float | None = None):
        self.hold_s, self.cooldown_s = hold_s, cooldown_s
        self.end_s = END_S if end_s is None else end_s
        self.cue = None         # the believed cue (None = no cue)
        self.since = None       # when it started (s)
        self.fired = {}         # cue -> time of the last reaction
        self.gone_since = None  # when the believed cue stopped being seen; None = seen now

    def update(self, t: float, cue: str | None) -> str | None:
        """t: now (s); cue: what is seen now. Returns the cue when it is time to react, else None."""
        if cue == self.cue:
            self.gone_since = None
        elif self.cue is None:
            self.cue, self.since = cue, t
        elif self.gone_since is None:
            self.gone_since = t
        elif t - self.gone_since >= self.end_s:
            self.cue, self.since, self.gone_since = cue, t, None
        # inside the grace period self.cue is still the old cue, so the timer keeps running
        if self.cue is None:
            return None
        # `cue != self.cue`: time alone is not enough; the cue must actually be seen at this moment.
        # Without it, a 10 s head-down followed by looking up would still fire at 30 s during the grace period.
        if t - self.since < self.hold_s or cue != self.cue:
            return None
        if self.cue in self.fired and t - self.fired[self.cue] < self.cooldown_s:
            return None
        self.fired[self.cue] = t
        return self.cue


# ── drives (all in 0-1) ─────────────────────────────────────────
#   E energy:     play lowers it, time slowly restores it
#   C curiosity:  rises when nothing happens (boredom); exploring and play lower it
#   A attachment: rises without interaction (missing the owner); company and greeting lower it
#   S stress:     scolding or an angry-looking owner raise it; time lowers it
START = {"E": 0.7, "C": 0.5, "A": 0.5, "S": 0.1}
BASE = {"E": 0.8, "C": 0.8, "A": 0.8, "S": 0.1}          # where each drifts when nothing happens
TAU = {"E": 300.0, "C": 100.0, "A": 120.0, "S": 60.0}    # time constants (s): E recovers 63% in 5 min, S calms mostly in 1 min
#   from START to the 0.75 threshold: bored after ~3 min, missing the owner after ~3.6 min,
#   so the pet finds something to do on its own every few minutes

# events, cues, behaviors -> nudges (added, then clipped to 0-1)
EFFECTS = {
    # events (seen)
    "came_home": {"A": +0.2}, "toy_out": {"C": +0.2}, "leaving": {"A": +0.1},
    # cues (reported by CueTimer after HOLD_S)
    "angry": {"S": +0.3}, "low": {"A": +0.1}, "welcoming": {"A": +0.1},
    # behaviors (done by the duck) = needs met
    "greet": {"A": -0.3, "E": -0.05}, "stay_close": {"A": -0.4}, "gaze": {"A": -0.1},
    "play_bow": {"E": -0.2, "C": -0.2, "A": -0.1}, "zoomies": {"E": -0.3, "C": -0.2},
    "investigate": {"C": -0.4}, "guilty": {"S": -0.1}, "rest": {"E": +0.15},
}

# wants (call the LLM): (name, variable, direction, fire threshold, re-arm threshold) -- hysteresis
WANTS = [("miss_owner", "A", "high", 0.75, 0.5),
         ("bored", "C", "high", 0.75, 0.5),
         ("tired", "E", "low", 0.25, 0.5)]


class Drives:
    def __init__(self, start=None, base=None, tau=None):
        self.x = dict(start or START)
        self.base, self.tau = dict(base or BASE), dict(tau or TAU)
        self.active = set()        # wants that fired and have not re-armed yet

    def step(self, dt: float):
        """dt seconds pass: each variable relaxes toward its baseline.
        Exact solution of dx/dt = (base - x) / tau, so the result does not depend on the step size
        (forward Euler with dt = 200 s would overshoot to 1.4)."""
        for k in self.x:
            self.x[k] = self.base[k] + (self.x[k] - self.base[k]) * math.exp(-dt / self.tau[k])

    def bump(self, name: str):
        """An event / cue / behavior nudges the drives (names not in EFFECTS do nothing)."""
        for k, d in EFFECTS.get(name, {}).items():
            if k in self.x:
                self.x[k] = min(1.0, max(0.0, self.x[k] + d))

    def check(self) -> list[str]:
        """Wants that newly fire now (once per threshold crossing)."""
        fired = []
        for name, var, kind, on, off in WANTS:
            if var not in self.x:
                continue
            x = self.x[var]
            if kind == "high":
                if name not in self.active and x >= on:
                    self.active.add(name)
                    fired.append(name)
                elif name in self.active and x < off:
                    self.active.discard(name)
            elif kind == "low":
                if name not in self.active and x <= on:
                    self.active.add(name)
                    fired.append(name)
                elif name in self.active and x > off:
                    self.active.discard(name)
        return fired


def level(v: float) -> str:
    return "low" if v < 0.3 else "high" if v > 0.7 else "medium"


WANT_TEXT = {"miss_owner": "You miss your owner and want to be near them.", "bored": "You are bored and curious.",
             "tired": "You are tired."}


class Inner:
    """CueTimer + Drives: call update() every perception step, done() after a behavior."""

    def __init__(self, hold_s: float = HOLD_S, cooldown_s: float = COOLDOWN_S, drives: Drives | None = None):
        self.cues, self.d = CueTimer(hold_s, cooldown_s), drives or Drives()
        self.log = []

    def update(self, t: float, obs: dict, events: list, dt: float) -> tuple[list, str | None]:
        """Returns (reasons to call the LLM, text to append to the sentence); ([], None) if no call is needed."""
        self.d.step(dt)
        for e in events:
            self.d.bump(e)
        reasons, cue = [f"event:{e}" for e in events], None
        if obs["present"]:                               # not seen is not over: no update, the timer keeps running
            cue = self.cues.update(t, cue_of(obs))
            if cue:
                self.d.bump(cue)
                reasons.append(f"cue:{cue}")
        wants = self.d.check()
        reasons += [f"want:{w}" for w in wants]
        if not reasons:
            return [], None
        self.log.append(f"t={t:6.1f} {reasons}  {self.state_text()}")
        return reasons, self.text(obs, cue, wants)

    def text(self, obs: dict, cue: str | None = None, wants=()) -> str:
        """What is appended after the event sentence: a long-held cue + wants + inner state.
        (The mood evaluation calls this directly, so it tests the exact sentence that is sent.)

        Wants are stated while still active, not only at the moment they fire: when "tired" was said only
        once, a later "owner brings a toy" was answered with play_bow 3/3; with "You are tired." in the same
        sentence, 0/6."""
        parts = []
        if cue:
            bits = [obs["posture"], f"arms {obs['arms'].replace('_', ' ')}"] + (["head down"] if obs["head"] == "down" else [])
            parts.append(f"[camera] The owner has been like this for over {self.cues.hold_s:.0f} s "
                         f"({', '.join(bits)}): may be {CUE_TEXT.get(cue, cue)}.")
        wants = list(wants) + [w for w, *_ in WANTS if w in self.d.active and w not in wants]
        parts += [f"[self] {WANT_TEXT[w]}" for w in wants]
        return " ".join(parts + [self.state_text()])

    def done(self, behavior: str):
        self.d.bump(behavior)

    def state_text(self) -> str:
        x = self.d.x
        return (f"[self] energy {level(x['E'])}, curiosity {level(x['C'])}, "
                f"wants owner {level(x['A'])}, nervous {level(x['S'])}.")


# ─────────────────────────────────────────────────────────────
# selftest (no API, no simulator)
# ─────────────────────────────────────────────────────────────
def _timeline(segments, hold_s=HOLD_S, cooldown_s=COOLDOWN_S):
    """segments = [(seconds, cue), ...], fed once per second -> [(reaction time, cue), ...]"""
    ct, t, out = CueTimer(hold_s, cooldown_s), 0, []
    for secs, cue in segments:
        for _ in range(secs):
            got = ct.update(float(t), cue)
            if got:
                out.append((t, got))
            t += 1
    return out


def cue_cases(E: int):
    """(description, timeline, expected reaction times). A gap shorter than the grace period E is a misread
    (the cue continues); longer is a real end (the timer restarts)."""
    short, long_ = max(1, E // 2), E + 5
    return [
        ("head down 10 s then up (answering a message) -> no reaction", [(10, "tired"), (30, None)], []),
        ("head down 40 s -> one reaction at 30 s", [(40, "tired")], [(30, "tired")]),
        ("head down 5 min -> 30, 150, 270 s (check again every 120 s)", [(300, "tired")], [(30, "tired"), (150, "tired"), (270, "tired")]),
        (f"head down 20 s, misread {short} s (shorter than grace), head down again -> no reset, reacts at {max(30, 20 + short)} s",
         [(20, "tired"), (short, None), (40, "tired")], [(max(30, 20 + short), "tired")]),
        ("head down 10 s then up 40 s -> no reaction during the grace period (whatever its length)", [(10, "tired"), (40, None)], []),
        (f"head down 20 s, really up {long_} s (longer than grace), head down 40 s -> timer restarts, {25 + E + 30} s",
         [(20, "tired"), (long_, None), (40, "tired")], [(25 + E + 30, "tired")]),
        ("as measured: head down 7 s, VLM misread 7 s, head down 40 s -> " + ("no reset, 30 s" if E > 6 else "grace too short -> reset, 44 s"),
         [(7, "tired"), (7, None), (40, "tired")], [(30, "tired")] if E > 6 else [(44, "tired")]),
        (f"head down 40 s then hugging knees -> the new cue starts after the {E} s grace period -> {70 + E} s",
         [(40, "tired"), (E + 40, "sad")], [(30, "tired"), (70 + E, "sad")]),
        (f"after a reaction, misread {short} s, head down again -> cue continues, cooling down, no reaction",
         [(40, "tired"), (short, None), (40, "tired")], [(30, "tired")]),
        (f"after a reaction, really up {long_} s, head down 130 s -> reacts again only after the cooldown (150 s)",
         [(40, "tired"), (long_, None), (130, "tired")], [(30, "tired"), (max(75 + E, 150), "tired")]),
    ]


def selftest() -> int:
    ok, n = 0, 0

    def check(desc, good):
        nonlocal ok, n
        ok, n = ok + bool(good), n + 1
        print(f"{'ok ' if good else 'BAD'} {desc}")

    print("-- CueTimer --")
    for desc, segs, want in cue_cases(int(END_S)):
        got = _timeline(segs)
        check(f"{desc}\n     got {got}" + ("" if got == want else f" (want {want})"), got == want)

    print("-- first-order drift (exact solution) --")
    d = Drives(start={"E": 0.2}, base={"E": 0.8}, tau={"E": 100.0})
    xs = []
    for _ in range(2):
        d.step(1.0)
        xs.append(d.x["E"])
    exact = [0.8 - 0.6 * math.exp(-k / 100) for k in (1, 2)]
    check(f"0.2 -> {xs[0]:.5f} after 1 s, {xs[1]:.5f} after 2 s (Euler by hand: 0.206, 0.21194)",
          all(abs(a - b) < 1e-9 for a, b in zip(xs, exact)) and all(abs(a - b) < 1e-4 for a, b in zip(xs, (0.206, 0.21194))))
    d = Drives(start={"E": 0.2}, base={"E": 0.8}, tau={"E": 100.0})
    d.step(200.0)
    check(f"one step of dt=200 s -> {d.x['E']:.3f} (Euler would give 1.4, past the baseline)",
          abs(d.x["E"] - (0.8 - 0.6 * math.exp(-2))) < 1e-9)
    d = Drives(start={"E": 0.2}, base={"E": 0.8}, tau={"E": 100.0})
    for _ in range(100):
        d.step(1.0)
    frac = (d.x["E"] - 0.2) / 0.6
    check(f"after tau = 100 s -> {frac:.1%} recovered (theory 1 - 1/e = 63.2%)", abs(frac - (1 - math.exp(-1))) < 0.01)
    d = Drives(start={"E": 0.2}, base={"E": 0.8}, tau={"E": 100.0})
    for _ in range(200):
        d.step(0.5)
    frac2 = (d.x["E"] - 0.2) / 0.6
    check(f"dt=0.5 s for 200 steps (also 100 s) -> {frac2:.1%}, same as dt=1", abs(frac2 - frac) < 0.01)

    print("-- wants (hysteresis) --")
    d = Drives(start={"A": 0.4}, base={"A": 0.8}, tau={"A": 120.0})
    fires = []
    for t in range(600):
        d.step(1.0)
        fires += [(t, w) for w in d.check()]
    check(f"ignored for 600 s -> fires once, not every second: {fires}",
          len(fires) == 1 and fires[0][1] == "miss_owner" and 240 <= fires[0][0] <= 260)
    d = Drives(start={"A": 0.4}, base={"A": 0.8}, tau={"A": 120.0})
    fires = []
    for t in range(600):
        d.step(1.0)
        for w in d.check():
            fires.append((t, w))
            d.bump("stay_close")          # misses the owner -> keeps them company -> A below 0.5 -> re-armed
            d.check()
    check(f"keeps the owner company each time -> fires twice in 600 s: {fires}", len(fires) == 2 and 500 <= fires[1][0] <= 530)
    d = Drives(start={"E": 0.7})
    after = []
    for _ in range(3):
        d.bump("play_bow")
        after.append(d.check())
    check(f"energy 0.7, three play bows -> tired only after the third (0.5 -> 0.3 -> 0.1): {after}", after == [[], [], ["tired"]])

    print("-- Inner --")
    inner = Inner()
    seen = {"present": True, "posture": "sitting", "arms": "resting", "head": "down"}
    unseen = {"present": False, "posture": "unknown", "arms": "unknown", "head": "unknown"}
    calls = []
    for t in range(55):
        obs = seen if t < 20 or t >= 40 else unseen      # 20-40 s the duck looks away
        r, _ = inner.update(float(t), obs, [], 1.0)
        calls += [(t, x) for x in r if x.startswith("cue")]
    check(f"head down 20 s, unseen 20 s, seen again -> reacts at 40 s (timer not reset): {calls}", calls == [(40, "cue:low")])
    inner = Inner(drives=Drives(start=dict(START, E=0.1)))
    calm = {"present": True, "posture": "standing", "arms": "resting", "head": "up"}
    r1, _ = inner.update(0.0, calm, [], 0.5)
    r2, s2 = inner.update(0.5, calm, ["toy_out"], 0.5)
    check(f"still tired -> every later call says \"You are tired.\": {r2} {s2[:40]}...",
          r1 == ["want:tired"] and r2 == ["event:toy_out"] and "You are tired." in s2)
    print(f"\n{ok}/{n}")
    return n - ok


if __name__ == "__main__":
    raise SystemExit(selftest())
