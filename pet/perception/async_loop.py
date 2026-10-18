"""The VLM runs in the background: while it thinks, the duck (the simulator) keeps going.

The synchronous loop (loop.Perception.watch_step) pauses the simulator for ~1.5 s per VLM call, twice for confirmation.
A simulator can pause; a real robot cannot (physics does not wait). A single VLM answer is not faster (~1.5 s is the
API's fixed cost); what is saved: the two confirmation calls run at the same time, and the detector, small reactions
and the control loop keep running while waiting.

Only the "ask the VLM" box changes:
    every WATCH_DT: first collect answers that have "arrived" -> _on_result -> event detector
                    then one detector look (loop._dino_look, shared) -> if needed, take this photo, send it to the
                    background (_submit), return at once

Design decisions
    Two timestamps per answer: *capture time* (the moment of the photo: ordering, how old it is) and *arrival time*
    (when the answer comes back: when it can be acted on, latency). The same as a ROS message's header.stamp, which is
    the capture time. The region is computed from the camera pose at capture time: the duck may have turned since.
    Speculative execution: the detector saw a change (appeared, shape changed, yellow box at the hands) -> something
    probably happened and latency matters most -> one WATCH_DT later take a *different* photo and wait for both.
    Only a periodic look -> ask about one photo; take a second only if the answer differs from what is believed
    (usually nothing changed -> saves money). Confirmation must use different photos: asking about the same photo
    twice repeats the same mistake.
    Stale answers: an answer captured before the newest one already used does not update the state (old news cannot
    overwrite new news), unless it agrees with the state waiting for confirmation -> it still counts as one vote.
    This applies to observations only; a stale *plan* is re-checked before it runs (mind/async_plan.py).

Timing in the simulator (kept honest): an answer counts as arrived at sim time = capture time + the real seconds it
took. If the simulator runs faster than real time, wait for real time to catch up (or answers would arrive late in the
simulation and latency would be measured too pessimistically).

    python -m pet.perception.async_loop      # selftest with a fake VLM, no cost
"""

import time
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass, field

from .loop import SEND_MODE, Perception, UnclearAnswer, ask_state, owner_xy
from ..nav.waypoints import region

URGENT = ("appeared", "shape changed", "yellow box at the hands")   # the detector saw a change -> speculative execution
MAX_INFLIGHT = 2                                # at most this many questions in the background at once
TOY_WIN = 2.0                                   # when the VLM says holding_toy, the detector must have seen a yellow box at
                                                # the owner's hands within this many seconds of the photo


def toy_confirmed(arms: str, t_cap: float, toy_times: list, win: float = TOY_WIN) -> str:
    """Accept the VLM's arms? A second opinion must come from an *independent* source: asking the same VLM about the
    same scene again repeats the same mistake.

    Why: the owner hugged their knees, the VLM said holding_toy on two photos in a row (hands in front of the knees),
    both agreed -> toy_out -> the duck ran over to invite play. The VLM reads the hand pose; the detector looks for the
    yellow box itself -> they fail differently and rarely fail together. Inviting a sad owner to play is serious:
    here a false alarm is worse than a miss -> better late, or missed.
    toy_times: the sim times at which the detector saw a yellow box overlapping the owner box (toy_near).
      arms is not holding_toy -> unchanged (only this one value is checked)
      any detector sighting within win of t_cap (before or after: what the detector sees while the VLM is still
      thinking also counts) -> "holding_toy"
      none -> "unknown" (not "resting": we do not know the hands are empty, we only lack evidence)"""
    if arms != "holding_toy":
        return arms
    if any(abs(t - t_cap) <= win for t in toy_times):
        return "holding_toy"
    return "unknown"


@dataclass
class Pending:
    """A claim ticket: one question sent to the background."""
    fut: object               # concurrent.futures.Future (.done(), .result())
    t_cap: float              # capture time (sim seconds)
    kind: str                 # "urgent" (the detector saw a change) / "patrol" (periodic) / "second" (second photo)
    region: str               # region computed at capture time
    motion: str               # detector motion at capture time
    wall0: float = field(default_factory=time.perf_counter)
    wall_done: float | None = None
    secs: float = 0.0


class AsyncPerception(Perception):
    def __init__(self, vw, detector=None):
        super().__init__(vw, detector)
        self.pool = ThreadPoolExecutor(max_workers=MAX_INFLIGHT)
        self.inflight: list[Pending] = []
        self.latest_cap = -1e9      # capture time of the newest answer used so far
        self.second = False         # take another photo next step? (speculative, or a periodic look saw a difference)
        self.vlm_async = []         # [(capture time, arrival time, seconds)] (not added to vlm_calls: the simulator did not pause)
        self._toy_times = []        # when the detector saw a yellow box at the owner's hands
        self.dropped = []           # stale answers dropped [(capture time, arrival time)]

    def close(self):
        self.pool.shutdown(wait=True)

    def _on_result(self, p: Pending, st) -> list:
        """An answer has arrived. p: the ticket (p.t_cap, p.kind, p.region, p.motion); st: OwnerState. Returns events.
          stale (captured at or before the newest answer used): agrees with the pending state -> one vote only
                (latest_cap and last unchanged); otherwise dropped
          fresh: becomes the newest; update the event detector and the last state
          a pending state and nothing else in flight -> take another photo next step (a tie-breaker)"""
        sim = self.vw.sim
        obs = self._obs(p, st)
        events = []

        if p.t_cap <= self.latest_cap:
            is_match = False
            if self.ev.pending is not None:
                if (obs["posture"] == self.ev.pending["posture"] and
                        obs["arms"] == self.ev.pending["arms"] and
                        obs["region"] == self.ev.pending["region"]):
                    is_match = True
            if not is_match:
                self.dropped.append((p.t_cap, sim.t))
            else:
                events = self.ev.update(obs)
        else:
            self.latest_cap = p.t_cap
            events = self.ev.update(obs)
            self.last = obs

        if self.ev.pending is not None and not self.inflight:
            self.second = True

        return events

    def _obs(self, p: Pending, st) -> dict:
        obs = st.model_dump()
        obs["region"] = p.region if st.present else "other"
        # motion is the detector's value *now*, not at capture time: motion carried by each photo jumped between
        # receding and still (the box height does not change while the owner turns), and feeding answers one by one
        # switched it back and forth -> the detector saw "starts walking away" again and "leaving" was reported twice.
        # Motion is re-measured by the detector every 0.5 s, not by the VLM, so VLM answers must not revert it.
        # (The region still uses capture time: it needs the camera pose of the photo. p.motion is kept for the log.)
        obs["motion"] = self.motion
        obs["arms"] = toy_confirmed(obs["arms"], p.t_cap, self._toy_times)   # holding a toy needs the detector's backing
        # Each sensor confirms what it measures: two VLM answers both said standing, but 3 s apart the detector's motion
        # was approaching (the box grew while standing up) on one and still on the other -> the event detector compares
        # the whole dict -> never "the same seen twice" -> never confirmed. When the VLM fields match the pending state,
        # motion follows the pending one (motion is confirmed by the detector on its own, every step).
        pend = self.ev.pending
        if pend is not None and all(pend[k] == obs[k] for k in ("posture", "arms", "region")):
            obs["motion"] = pend["motion"]
        return obs

    def _submit(self, box, kind: str):
        """Send the current photo (vw.last_img, just taken by _dino_look) to the background and return at once."""
        from PIL import Image
        sim, vw = self.vw.sim, self.vw
        reg = region(*owner_xy(sim, vw.cid, box))       # with the camera pose at capture time
        img = Image.fromarray(vw.last_img)
        p = Pending(None, sim.t, kind, reg, self.motion)
        p.fut = self.pool.submit(ask_state, img, SEND_MODE, box)
        p.fut.add_done_callback(lambda f, p=p: setattr(p, "wall_done", time.perf_counter()))
        self.inflight.append(p)
        self.log.append(f"t={sim.t:5.1f} sent to the VLM ({kind})")

    def _arrived(self) -> list:
        """Answers that have arrived in *sim time* (capture time + real seconds <= now), in arrival order: [(ticket, state)]."""
        sim = self.vw.sim
        ready = []
        for p in list(self.inflight):
            ahead = (sim.t - p.t_cap) - (time.perf_counter() - p.wall0)
            if not p.fut.done() and ahead > 0:
                wait([p.fut], timeout=ahead)           # simulator ahead of real time -> let real time catch up
            if not p.fut.done() or p.wall_done is None:
                continue
            p.secs = p.wall_done - p.wall0
            if sim.t + 1e-9 < p.t_cap + p.secs:
                continue                               # in the real world this answer has not arrived yet
            self.inflight.remove(p)
            try:
                st, _ = p.fut.result()
            except UnclearAnswer as e:
                self.log.append(f"t={sim.t:5.1f} VLM (captured {p.t_cap:.1f}) invalid answer -> not seen clearly ({e})")
                continue
            self.vlm_async.append((p.t_cap, sim.t, p.secs))
            self.log.append(f"t={sim.t:5.1f} VLM arrived (captured {p.t_cap:.1f}, {p.secs:.1f}s, {p.kind}): {self._obs(p, st)}")
            ready.append(p)
        ready.sort(key=lambda p: p.t_cap + p.secs)
        return [(p, p.fut.result()[0]) for p in ready]

    def watch_step(self, at=None, hold_fn=None):
        """Same interface as loop.Perception.watch_step (hold_fn is unused: nothing pauses). Returns (state, events)."""
        sim, vw = self.vw.sim, self.vw
        events = []
        for p, st in self._arrived():
            ev = self._on_result(p, st)
            if ev:
                self.log.append(f"t={sim.t:5.1f} events {ev} (captured {p.t_cap:.1f})")
                events += ev
        arr, box, why, toy_near, aspect = self._dino_look()
        if toy_near:
            self._toy_times = [t for t in self._toy_times if sim.t - t < 10.0] + [sim.t]   # for toy_confirmed
        if box is None:
            self.second = False
            return self.last, events + self.ev.update({"present": False})
        room = len(self.inflight) < MAX_INFLIGHT
        if self.second and room:
            self.second = False
            self._submit(box, "second")
        elif why is not None and room:
            self.log.append(f"t={sim.t:5.1f} DINO: {why}")
            self._submit(box, "urgent" if why in URGENT else "patrol")
            self.second = why in URGENT              # a change -> take another photo next step and ask both
            self._vlm_box, self._vlm_toy, self._vlm_t, self._vlm_aspect = box, toy_near, sim.t, aspect
        elif self.last["present"] and not events and not self._waiting_second_photo():
            # no VLM call this step: as in the synchronous loop, keep posture, update only motion ("leaving" relies on this)
            obs = dict(self.last, motion=self.motion)
            self.last = obs
            ev = self.ev.update(obs)
            if ev:
                self.log.append(f"t={sim.t:5.1f} events {ev} (detector only: motion={self.motion})")
                events += ev
        vw.last.update(box=box, status="ok", verdict=f"{self.last['posture']}, {self.last['arms']} ({len(self.inflight)} in flight)")
        return self.last, events

    def _waiting_second_photo(self) -> bool:
        """Is the event detector waiting to confirm posture / arms / region? Then this step may not feed a *copy* of the
        last VLM answer. A copy is not a new photo: the first photo said standing -> pending -> 0.5 s later the copy said
        standing again -> counted as seen twice -> stood_up reported before the second photo came back, breaking
        "confirmation needs different photos". Motion is unaffected: the detector really re-measures it every step
        ("leaving" is still confirmed by copies + motion)."""
        pend, conf = self.ev.pending, self.ev.confirmed or {}
        return pend is not None and any(pend[k] != conf.get(k) for k in ("posture", "arms", "region"))


# ─────────────────────────────────────────────────────────────
# selftest: fake tickets fed straight into _on_result (no simulator, no cost)
# ─────────────────────────────────────────────────────────────
def selftest() -> int:
    from types import SimpleNamespace

    from .owner_state import OwnerState

    def st(posture="sitting", arms="resting"):
        return OwnerState(present=True, posture=posture, arms=arms, head="up", facing_duck="yes")

    def fresh():
        per = AsyncPerception.__new__(AsyncPerception)          # no VisionWorld, no threads
        Perception.__init__(per, SimpleNamespace(sim=SimpleNamespace(t=0.0)))
        per.inflight, per.latest_cap, per.second, per.vlm_async, per.dropped = [], -1e9, False, [], []
        per._toy_times = []
        per.ev.known_home = True
        return per

    def P(t_cap, kind="urgent"):
        return Pending(None, t_cap, kind, "living_room", "still")

    def feed(per, seq):
        out = []
        for t_cap, kind, s in seq:
            per.vw.sim.t = t_cap + 1.5
            out += per._on_result(P(t_cap, kind), s)
        return out

    def base(per):
        """Confirm "sitting" first (each case's start). Reset second: in a real run the next watch_step uses it up."""
        feed(per, [(0.0, "urgent", st()), (0.5, "second", st())])
        per.second = False

    ok = n = 0

    def check(desc, good, got):
        nonlocal ok, n
        ok, n = ok + bool(good), n + 1
        print(f"{'ok ' if good else 'BAD'} {desc}\n     {got}")

    for desc, args, want in [
        ("not holding_toy -> unchanged", ("resting", 10.0, []), "resting"),
        ("holding a toy, detector saw it 1 s before the photo -> accept", ("holding_toy", 10.0, [9.0]), "holding_toy"),
        ("holding a toy, detector saw it 1.5 s after the photo (VLM still thinking) -> accept", ("holding_toy", 10.0, [11.5]), "holding_toy"),
        ("holding a toy, detector saw it only 5 s before -> no evidence -> unknown", ("holding_toy", 10.0, [5.0]), "unknown"),
        ("holding a toy, detector never saw one (the hugging-knees case) -> unknown, not resting", ("holding_toy", 10.0, []), "unknown"),
        ("exactly 2.0 s apart -> counts (<=)", ("holding_toy", 10.0, [8.0]), "holding_toy"),
    ]:
        got = toy_confirmed(*args)
        check(desc, got == want, f"got {got!r} (want {want!r})")
    per = fresh()
    base(per)
    ev = feed(per, [(10.0, "urgent", st(arms="holding_toy")), (10.5, "second", st(arms="holding_toy"))])
    per2 = fresh()
    base(per2)
    per2._toy_times = [9.8, 10.3]
    ev2 = feed(per2, [(10.0, "urgent", st(arms="holding_toy")), (10.5, "second", st(arms="holding_toy"))])
    check("end to end: the VLM says holding a toy twice -- detector saw nothing -> no toy_out; detector saw it -> toy_out",
          "toy_out" not in ev and ev2 == ["toy_out"], f"not seen -> {ev}; seen -> {ev2}")

    per = fresh()
    ev = feed(per, [(0.0, "urgent", st()), (0.5, "second", st())] + [(10.0, "urgent", st("standing")), (10.5, "second", st("standing"))])
    check("in order: sitting -> standing (two different photos) -> stood_up", ev == ["stood_up"], ev)

    per = fresh()
    base(per)
    ev = feed(per, [(10.5, "second", st("standing")), (10.0, "urgent", st())])
    check("the later photo (10.5) arrives first -> used; 10.0 arrives late and disagrees -> stale, dropped",
          per.dropped and per.dropped[0][0] == 10.0 and per.latest_cap == 10.5,
          f"dropped={per.dropped} latest_cap={per.latest_cap} events={ev}")

    per = fresh()
    base(per)
    feed(per, [(6.0, "patrol", st())])
    a = per.second
    feed(per, [(12.0, "patrol", st("standing"))])
    b = per.second
    check("periodic look: same as before -> no second photo; different -> a second photo to confirm", a is False and b is True,
          f"same -> {a}  different -> {b}")

    per = fresh()
    base(per)
    per.inflight = [P(10.5, "second")]
    feed(per, [(10.0, "urgent", st("standing", "holding_toy"))])
    c = per.second
    check("speculative: the first photo returns and differs, the second is still in flight -> no extra photo", c is False, f"second={c}")

    per = fresh()
    base(per)
    ev = feed(per, [(10.5, "second", st("standing")), (10.0, "urgent", st("standing"))])
    check("the second photo returns first, the first arrives late but agrees -> one vote -> stood_up; latest_cap stays 10.5",
          ev == ["stood_up"] and per.latest_cap == 10.5 and not per.dropped, f"events={ev} latest_cap={per.latest_cap} dropped={per.dropped}")

    per = fresh()
    base(per)
    per.inflight = [P(10.0, "urgent")]
    feed(per, [(10.5, "second", st("standing"))])
    a = per.second
    per.inflight = []
    feed(per, [(10.0, "urgent", st())])
    b = per.second
    check("speculative photos disagree (later says standing, earlier says sitting) -> both back, nothing in flight -> one more photo",
          a is False and b is True and per.dropped, f"after the first: second={a}; after both: second={b}; dropped={per.dropped}")
    print(f"\n{ok}/{n}")
    return n - ok


if __name__ == "__main__":
    raise SystemExit(selftest())
