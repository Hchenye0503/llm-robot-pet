"""Event detection from the camera alone: are events right (no misses, no false alarms) and fast enough?

Each scenario: the owner does one thing at a known time (or nothing) while the duck stands and watches
(perception.async_loop: the detector every 0.5 s, the VLM in the background when the image changes).
    1. the expected event appears (no misses)
    2. no other event appears (no false alarms; an event reported *before* the action also counts as false)
    3. latency: small reaction (detector notices the change; the duck looks up) <= 2 s from the start of the action;
       full event (the event detector reports it) <= 5 s. The VLM costs ~1.5 s per answer (API fixed cost) and a
       confirmation needs two -> the reaction the owner first notices is the detector's, hence two limits.
Every scenario runs REPEAT times (VLM answers vary); a scenario passes only if all runs pass.

Sets (positions differ between sets; the duck stands 1.6-2.2 m from the owner, where the whole body is visible)
    DEV        S1-S6, used while building the system
    HOLDOUT_2  J1-J8, written before its only run after the "sitting down reported as leaving" fix: 21/24 then.
               The "toy in the owner's hands" rule changed afterwards (at least 80% of the yellow box inside the owner's
               box instead of any overlap), so later runs of this set are development measurements, not held-out ones.

    python -m eval.events --check            # start positions free, owner visible to the detector (no VLM cost)
    python -m eval.events --set dev          # needs OPENAI_API_KEY
    python -m eval.events --set holdout2 [--only J6]
"""

import argparse
import math

from pet.body import sim as S
from pet.body.scheduler import hold
from pet.perception import vision_world as V
from pet.perception.async_loop import AsyncPerception
from pet.perception.loop import WATCH_DT, Perception
from pet.world import owner as O

REPEAT = 3
LIMIT_S = 5.0          # full event
NOTICE_S = 2.0         # small reaction
T_ACT = 6.0            # the owner acts this long after the start (the first 6 s also measure false alarms)
O_SPOT = (-1.4, -1.9)  # the owner's place in the living room (facing west)
ENTRY_IN, ENTRY_STAND = (0.05, 2.75), (0.05, 2.2)


def face(src, dst):
    """Heading (deg) of src looking at dst."""
    return round(math.degrees(math.atan2(dst[1] - src[1], dst[0] - src[0])), 1)


DEV = {
    "S1 came home": dict(duck=(0.05, 0.0, 90), owner=dict(pose="stand", at=O.AWAY, yaw=-90), home=False, dur=16,
                         act=lambda o: o.do("appear", at=ENTRY_IN, yaw=-90, pose="stand").do("walk", path=[ENTRY_STAND], then="stand"),
                         want=["came_home"], pose_change=False),
    "S2 stands up": dict(duck=(-3.565, -1.9, 0), owner=dict(pose="sit", at=O_SPOT, yaw=180), home=True, dur=14,
                         act=lambda o: o.do("pose", name="stand", blend=0.8), want=["stood_up"], pose_change=True),
    "S3 leaving": dict(duck=(-3.55, -1.9, 0), owner=dict(pose="stand", at=O_SPOT, yaw=180), home=True, dur=18,
                       act=lambda o: o.do("walk", path=[(-1.3, -1.4), (-0.6, -1.4)], then="stand"), want=["leaving"], pose_change=False),
    "S4 holds a toy": dict(duck=(-3.75, -1.9, 0), owner=dict(pose="stand", at=(-1.6, -1.9), yaw=180), home=True, dur=14,
                           act=lambda o: o.do("pose", name="hold_toy", blend=0.8), want=["toy_out"], pose_change=True),
    "S5 still": dict(duck=(-3.565, -1.9, 0), owner=dict(pose="sit", at=O_SPOT, yaw=180), home=True, dur=20,
                     act=lambda o: o, want=[], pose_change=False),
    "S6 waves": dict(duck=(-3.75, -1.9, 0), owner=dict(pose="stand", at=(-1.6, -1.9), yaw=180), home=True, dur=14,
                     act=lambda o: o.do("pose", name="wave", blend=0.8), want=[], pose_change=True),   # waving is not an event
}

J1_DUCK = (-0.67, -0.35)
J2_DUCK, J2_OWNER = (-1.2, -0.6), (-2.2, -2.3)
J3_DUCK, J3_OWNER = (-3.3, -1.9), (-1.6, -2.4)
J4_DUCK, J4_OWNER = (-1.2, -0.4), (-2.8, -2.0)
J5_DUCK, J5_OWNER = (-1.5, -0.5), (-1.5, -2.5)
J6_DUCK, J6_OWNER = (-1.1, -1.9), (-3.0, -2.2)
J7_DUCK, J7_OWNER = (-1.0, -1.4), (-2.6, -2.2)
J8_DUCK, J8_OWNER = (-1.3, -0.5), (-2.4, -2.3)

HOLDOUT_2 = {
    "J1 came home": dict(duck=(*J1_DUCK, face(J1_DUCK, ENTRY_STAND)), owner=dict(pose="stand", at=O.AWAY, yaw=-90), home=False, dur=16,
                         act=lambda o: o.do("appear", at=ENTRY_IN, yaw=-100, pose="stand").do("walk", path=[(0.05, 1.8)], then="stand"),
                         want=["came_home"], pose_change=False),
    "J2 sits down (front)": dict(duck=(*J2_DUCK, face(J2_DUCK, J2_OWNER)), owner=dict(pose="stand", at=J2_OWNER, yaw=face(J2_OWNER, J2_DUCK)),
                                 home=True, dur=14, act=lambda o: o.do("pose", name="sit", blend=0.8), want=[], pose_change=True),
    "J3 sits down (side)": dict(duck=(*J3_DUCK, face(J3_DUCK, J3_OWNER)), owner=dict(pose="stand", at=J3_OWNER, yaw=face(J3_OWNER, J3_DUCK) + 90),
                                home=True, dur=14, act=lambda o: o.do("pose", name="sit", blend=0.8), want=[], pose_change=True),
    "J4 leaving": dict(duck=(*J4_DUCK, face(J4_DUCK, J4_OWNER)), owner=dict(pose="stand", at=J4_OWNER, yaw=face(J4_OWNER, J4_DUCK)),
                       home=True, dur=18, act=lambda o: o.do("walk", path=[(-3.4, -2.6)], then="stand"), want=["leaving"], pose_change=False),
    "J5 stands up": dict(duck=(*J5_DUCK, face(J5_DUCK, J5_OWNER)), owner=dict(pose="sit", at=J5_OWNER, yaw=face(J5_OWNER, J5_DUCK)),
                         home=True, dur=14, act=lambda o: o.do("pose", name="stand", blend=0.8), want=["stood_up"], pose_change=True),
    "J6 holds a toy": dict(duck=(*J6_DUCK, face(J6_DUCK, J6_OWNER)), owner=dict(pose="stand", at=J6_OWNER, yaw=face(J6_OWNER, J6_DUCK)),
                           home=True, dur=14, act=lambda o: o.do("pose", name="hold_toy", blend=0.8), want=["toy_out"], pose_change=True),
    "J7 sits still": dict(duck=(*J7_DUCK, face(J7_DUCK, J7_OWNER)), owner=dict(pose="sit", at=J7_OWNER, yaw=face(J7_OWNER, J7_DUCK)),
                          home=True, dur=20, act=lambda o: o, want=[], pose_change=False),
    "J8 lies down": dict(duck=(*J8_DUCK, face(J8_DUCK, J8_OWNER)), owner=dict(pose="stand", at=J8_OWNER, yaw=face(J8_OWNER, J8_DUCK)),
                         home=True, dur=14, act=lambda o: o.do("pose", name="lie", blend=0.8), want=[], pose_change=True),
}

SETS = {"dev": DEV, "holdout2": HOLDOUT_2}


def run(name, sc, det, use_async=True):
    """One run of one scenario -> (passed, events, latencies [(sim, real)], VLM calls, small-reaction latency)."""
    sim = S.DuckSim(O.T.SCENE, None, spec_edit=O.stage_live, start=sc["duck"])
    owner = O.Owner(sim, **sc["owner"])
    for _ in range(50):
        sim.step("walk", S.zeros())
    x, y, _ = sim.xy_yaw()
    hit = sim.touching_what()
    if hit or math.dist((x, y), sc["duck"][:2]) > 0.08:
        raise RuntimeError(f"{name}: start position is not free (touching {hit}) -> invalid test")
    owner.look = (x, y)
    vw = V.VisionWorld(sim, det=det)
    per = AsyncPerception(vw) if use_async else Perception(vw)    # background VLM: the simulator does not pause
    per.ev.known_home = sc["home"]
    idle = ("walk", S.zeros())
    hold_fn = lambda s: hold(sim, idle, s)
    t0 = sim.t
    truth_t = [None]
    owner.do("wait", secs=T_ACT)
    sc["act"](owner)
    got = []
    while sim.t - t0 < sc["dur"]:
        if truth_t[0] is None and sim.t - t0 >= T_ACT:
            truth_t[0] = sim.t + (0.4 if sc["pose_change"] else 0.0)   # a pose change is visible about halfway through the blend
        obs, ev = per.watch_step(hold_fn=hold_fn)
        for e in ev:
            got.append((e, sim.t))
        hold(sim, idle, WATCH_DT)
    if use_async:
        per.close()
    vw.r.close()
    sim.close()
    # an event reported before the action is a false alarm (once "came home" fired before the owner appeared and still passed)
    early = [e for e, t in got if truth_t[0] is not None and t < truth_t[0] - 0.01]
    events = [e for e, t in got if e not in early]
    ok_set = sorted(events) == sorted(sc["want"]) and not early
    if early:
        events = events + [f"{e} (too early = false alarm)" for e in early]
    lat = []
    for e, t in got:
        if e in sc["want"] and truth_t[0] is not None:
            api = sum(s for (tc, s) in per.vlm_calls if truth_t[0] - 0.01 <= tc <= t)   # synchronous mode: add the paused API time
            lat.append((t - truth_t[0], t - truth_t[0] + api))
    ok_lat = all(real <= LIMIT_S for _, real in lat)
    # small reaction: the first "noticed" after the owner *starts* moving
    notice, t_start = None, t0 + T_ACT
    if sc["want"]:
        after = [tn for tn, _ in per.notices if tn >= t_start - 0.01]
        if after:
            api = sum(s for (tc, s) in per.vlm_calls if t_start - 0.01 <= tc < after[0])
            notice = after[0] - t_start + api
    ok_notice = not sc["want"] or (notice is not None and notice <= NOTICE_S)
    return ok_set and ok_lat and ok_notice, events, lat, len(getattr(per, "vlm_async", None) or per.vlm_calls), notice


def check(det, scenarios):
    """No VLM: is each start position free, and does the detector see the owner (after the action for came-home)?"""
    import mujoco
    from PIL import Image
    from pet.perception import geometry as G
    rows = []
    for name, sc in scenarios.items():
        sim = S.DuckSim(O.T.SCENE, None, spec_edit=O.stage_live, start=sc["duck"])
        owner = O.Owner(sim, **sc["owner"])
        for _ in range(50):
            sim.step("walk", S.zeros())
        x, y, _ = sim.xy_yaw()
        free = not sim.touching_what() and math.dist((x, y), sc["duck"][:2]) <= 0.08
        owner.look = (x, y)
        V.mount_camera(sim)
        r = mujoco.Renderer(sim.model, G.H_PX, G.W)
        seen = []
        for when in ("start", "after"):
            if when == "after":
                owner.do("wait", secs=0.5)
                sc["act"](owner)
                for _ in range(40):
                    hold(sim, ("walk", S.zeros()), 0.5)
                    if not owner.busy:
                        break
                hold(sim, ("walk", S.zeros()), 1.0)
            f = det(Image.fromarray(V.shoot(sim, r)[0]))["owner"]
            seen.append(f"{f[1]:.2f}" if f and f[1] >= V.MIN_SCORE["owner"] else "x")
        r.close()
        sim.close()
        need = seen[1:] if not sc["home"] else seen          # came home: the owner is not there at the start
        ok = free and "x" not in need
        rows.append(ok)
        print(f"{'ok ' if ok else 'BAD'} {name}: start {'free' if free else 'NOT free'}; detector sees the owner: start {seen[0]}, after {seen[1]}")
    print(f"\nvalid setups {sum(rows)}/{len(rows)}")
    return len(rows) - sum(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="dev", choices=list(SETS))
    ap.add_argument("--check", action="store_true", help="setup only, no VLM")
    ap.add_argument("--sync", action="store_true", help="synchronous perception (the simulator pauses for the VLM)")
    ap.add_argument("--only", help="only scenarios whose name starts with this, e.g. J6")
    a = ap.parse_args()
    det = V.Detector()
    scenarios = {n: sc for n, sc in SETS[a.set].items() if not a.only or n.startswith(a.only)}
    if a.check:
        raise SystemExit(check(det, scenarios))
    summary = {}
    for name, sc in scenarios.items():
        res = []
        for k in range(REPEAT):
            ok, events, lat, calls, notice = run(name, sc, det, use_async=not a.sync)
            res.append(ok)
            lat_s = ("" if notice is None else f"reaction ~{notice:.1f} s  ") + "  ".join(f"event {b:.1f} s" for _, b in lat)
            print(f"{name} #{k}: {'ok ' if ok else 'BAD'} events {events} (want {sc['want']})  {lat_s}  VLM {calls} calls", flush=True)
        summary[name] = sum(res)
    print(f"\n-- events, {a.set} --")
    for name, n in summary.items():
        print(f"  {name}: {n}/{REPEAT}")
    print(f"total {sum(summary.values())}/{REPEAT * len(summary)}")


if __name__ == "__main__":
    main()
