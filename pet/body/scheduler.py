"""Scheduler: turn a validated plan into a policy command every 20 ms.

    planner.llm.plan_with_retry -> Result (steps)
      -> run_steps (this file): one step at a time; every 20 ms picks the policy and its 13 numbers
      -> DuckSim.step(policy, cmd)

When is a step done? (three kinds of timing)
    walk      time is up (duration_s)                                  displacement: has a lifetime
    sitstand  the posture is *measured* (height + upright), or timeout   posture: latched once reached
    behavior / gesture / kick / spin_hop: run their own fixed-length routine

Two things always run in the background
    1. Fall detection, measured every 20 ms (not bookkept): switch to velstand with zero command to get up,
       then drop the rest of the plan and stand still. The world has changed; the old plan may no longer make sense.
    2. Watchdog when the queue is empty: after sitstand, latch its posture (sitting stays sitting);
       after walk, zero the command and stand (displacement must never continue on its own).

    python -m pet.body.scheduler --selftest           # fixed plans, no LLM
    python -m pet.body.scheduler --video chat.mp4      # type sentences, the duck acts them out (needs OPENAI_API_KEY)
"""

import argparse
import math

from . import sim as S
from .keyframes import cmd_at, duration
from ..planner.preconditions import insert_preconditions
from ..planner.schema import Behavior, Gesture, Kick, Refuse, SitStand, Walk
from ..planner.translate import TRANSLATORS

# sitstand posture targets (upstream microduck_sitstand_env_cfg; measured 0.060 / 0.116)
STAND_Z, SIT_Z = 0.115, 0.060
POSTURE_TOL = 0.01          # within 1 cm of the target counts as reached (SitStand steps use the sitstand policy)
STAND_LO, STAND_HI, SIT_HI = 0.090, 0.130, 0.070   # for sense_posture
SITSTAND_TIMEOUT_S = 4.0    # ramp is 2 s; measured: always reached within 4 s -> stop waiting after 4 s (logged)
UPRIGHT_OK = 0.9            # upright cos > 0.9 (within about 26 deg)


def is_fallen(sim) -> bool:
    return sim.upright() < 0.5 or sim.trunk_height() < 0.045


def sense_posture(sim) -> str:
    """Measured posture: standing / sitting / fallen / between (in transition).
    Lying also has height ~0.06, so height alone would call lying "sitting": check upright first."""
    if sim.upright() < 0.5:
        return "fallen"
    h = sim.trunk_height()
    # With the pet gait the trunk bobs 9.6-12.3 cm while walking (the earlier gait 11.3-12.3), so a single
    # 11.5 +- 1 cm band read the low point of each stride as "between", the LLM was told "fallen", and the plan
    # degraded. Measured extremes: standing / walking 9.6-12.3, sitting 5.9-6.0 -> two lines with margin:
    # standing 9.0-13.0 cm, sitting <= 7.0 cm, 7-9 cm = between (sitting down always passes through it).
    if sim.upright() > UPRIGHT_OK and STAND_LO <= h <= STAND_HI:
        return "standing"
    if sim.upright() > UPRIGHT_OK and h <= SIT_HI:
        return "sitting"
    return "between"


def step_done(step, elapsed: float, sim, log: list) -> bool:
    """Is this step finished? Walk: time is up. SitStand: the target height is reached and the trunk is upright,
    or the timeout passed (logged). A fixed 2 s ramp is not used: the posture is measured, not assumed."""
    if isinstance(step, Walk):
        return elapsed >= step.duration_s
    if isinstance(step, SitStand):
        target = SIT_Z if step.posture == "sit" else STAND_Z
        if abs(sim.trunk_height() - target) < POSTURE_TOL and sim.upright() > UPRIGHT_OK:
            return True
        if elapsed >= SITSTAND_TIMEOUT_S:
            log.append(f"t={sim.t:.1f}s SitStand({step.posture}) timed out after {SITSTAND_TIMEOUT_S:.1f}s -> stop waiting for the height")
            return True
        return False
    return True


def idle_command(last_policy: str, last_cmd: list) -> tuple[str, list]:
    """What to do when the queue is empty (watchdog): postures latch, displacement goes to zero.

    Standing is held with walk + zero command, not sitstand flag 0: switching from walking straight to
    sitstand(0) left the trunk tilted at cos 0.85 (about 32 deg) and sliding 6 cm, not recovered in 3 s;
    walk + zero gives cos 1.00 and 1.9 cm. sitstand only handles sitting and the sit <-> stand transition."""
    if last_policy == "sitstand" and last_cmd[0] >= 0.5:   # flag 1 = sit -> keep sitting
        return "sitstand", last_cmd
    return "walk", S.zeros()


WORLD: dict = {}   # where the owner / toy are, for behaviors; ground truth unless a VisionWorld is passed


def run_steps(sim, steps: list, log: list, max_s: float = 60.0, world: dict | None = None) -> dict:
    """Run a whole plan (preconditions already inserted) and return a summary."""
    queue = []
    for st in steps:
        if isinstance(st, Refuse):
            continue                       # refusal: no new action (the reason is already logged) -> watchdog holds the posture
        policy, cmd = TRANSLATORS[type(st)](st, log)
        queue.append((st, policy, cmd))

    t0, aborted, rescued = sim.t, False, None
    x0, y0, yaw0 = sim.xy_yaw()
    # with an empty queue (e.g. a degraded plan) the watchdog keeps the *measured* posture
    if sense_posture(sim) == "sitting":
        last_policy, last_cmd = "sitstand", [1.0] + [0.0] * 12
    else:
        last_policy, last_cmd = "walk", S.zeros()
    for qi, (st, policy, cmd) in enumerate(queue):
        start = sim.t
        if isinstance(st, Behavior):
            from .behaviors import _idle, run_behavior
            # world: a dict (ground truth) or f(sim, behavior name) -> dict, so the camera looks only when the behavior starts
            w = world(sim, st.name) if callable(world) else (world if world is not None else WORLD)
            run_behavior(sim, st.name, w, log, nxt=queue[qi + 1][0] if qi + 1 < len(queue) else None)
            last_policy, last_cmd = _idle(sim)
            if is_fallen(sim):
                rescued, aborted = rescue(sim, log), True
                log.append("fell -> drop the rest of the plan")
                last_policy, last_cmd = "walk", S.zeros()
                break
            continue
        if isinstance(st, Kick):
            # right foot; the ball must already be in the window in front of the right toe (-4 to +2 cm forward,
            # +-2 cm sideways). No turning to aim yet: the owner is assumed straight ahead.
            if st.target == "owner":
                log.append("kick(target=owner): not aimed (owner assumed straight ahead)")
            for _ in range(100):
                sim.step("kick_right", S.zeros())
            last_policy, last_cmd = "walk", S.zeros()
            if is_fallen(sim):
                rescued, aborted = rescue(sim, log), True
                log.append("fell -> drop the rest of the plan")
                break
            continue
        if isinstance(st, Gesture) and st.name == "spin_hop":
            # spin 1.5 s -> velstand brake 0.5 s -> back to walk. Switching straight to walk fell 3 of 21; braking first 0 of 21.
            for _ in range(75):
                sim.step("spin", S.zeros())
            for _ in range(25):
                sim.step("recover", S.zeros())
            last_policy, last_cmd = "walk", S.zeros()
            if is_fallen(sim):
                rescued, aborted = rescue(sim, log), True
                log.append("fell -> drop the rest of the plan")
                break
            continue
        if isinstance(st, Gesture):
            # the measured posture is the base; the head follows the keyframes every 20 ms; fixed length, no sensing needed
            policy, base = (("sitstand", [1.0] + [0.0] * 12) if sense_posture(sim) == "sitting" else ("walk", S.zeros()))
            while sim.t - start < duration(st.name) and not is_fallen(sim):
                sim.step(policy, cmd_at(st.name, sim.t - start, base))
            last_policy, last_cmd = policy, base
            if is_fallen(sim):
                rescued, aborted = rescue(sim, log), True
                log.append("fell -> drop the rest of the plan")
                last_policy, last_cmd = "walk", S.zeros()
                break
            continue
        while True:
            if is_fallen(sim):
                rescued = rescue(sim, log)
                aborted = True
                break
            if step_done(st, sim.t - start, sim, log) or sim.t - t0 > max_s:
                break
            sim.step(policy, cmd)
        last_policy, last_cmd = policy, cmd
        if aborted:
            log.append("fell -> drop the rest of the plan")
            last_policy, last_cmd = "walk", S.zeros()
            break
    x1, y1, yaw1 = sim.xy_yaw()
    return {"aborted": aborted, "rescued": rescued, "idle": idle_command(last_policy, last_cmd),
            "dist": math.hypot(x1 - x0, y1 - y0), "dyaw": (yaw1 - yaw0 + 180) % 360 - 180,
            "posture": sense_posture(sim), "secs": sim.t - t0}


def rescue(sim, log: list, timeout_s: float = 8.0) -> bool:
    """Fallen -> velstand with zero command until upright and tall enough for 0.5 s in a row."""
    log.append(f"t={sim.t:.1f}s fall detected -> velstand recovery")
    ok = 0
    for k in range(int(timeout_s / S.CONTROL_DT)):
        sim.step("recover", S.zeros())
        ok = ok + 1 if (sim.upright() > UPRIGHT_OK and sim.trunk_height() > 0.10) else 0
        if ok >= 25:
            for _ in range(25):                 # settle in place before handing back to walk
                sim.step("walk", S.zeros())
            log.append(f"recovered ({(k + 1) * S.CONTROL_DT:.1f} s)")
            return True
    log.append("recovery failed -> keep velstand with zero command (degraded)")
    return False


def hold(sim, idle, secs: float):
    for _ in range(int(secs / S.CONTROL_DT)):
        sim.step(*idle)


# ─────────────────────────────────────────────────────────────
# selftest: fixed plans (no LLM, no cost), run in the simulator, checked by measurement
# ─────────────────────────────────────────────────────────────
def W(v=0.0, turn=0.0, secs=2.0, yaw=0.0):
    return Walk(action="walk", forward_mps=v, turn_left_deg_s=turn, duration_s=secs, head={"yaw_left_deg": yaw})


def SS(p, yaw=0.0):
    return SitStand(action="sitstand", posture=p, head={"yaw_left_deg": yaw})


CASES = [
    # (description, start, plan, check(summary) -> bool, what is checked)
    ("walk 0.3 for 2 s", "standing", [W(0.3)], lambda r: r["dist"] > 0.15 and r["posture"] == "standing", "moved > 15 cm, ends standing"),
    ("sit", "standing", [SS("sit")], lambda r: r["posture"] == "sitting" and r["secs"] < SITSTAND_TIMEOUT_S, "ends sitting, no timeout"),
    ("walk while sitting -> stands up first", "sitting", [W(0.3)], lambda r: r["posture"] == "standing" and r["dist"] > 0.1, "ends standing, moved"),
    ("stand, look left, turn left 50 deg/s for 2 s", "standing", [SS("stand", yaw=30), W(turn=50)],
     lambda r: r["dyaw"] > 40 and r["posture"] == "standing", "body turned > 40 deg"),
    ("walk, sit, stand, walk (four steps)", "standing", [W(0.3), SS("sit"), SS("stand"), W(0.3)],
     lambda r: r["dist"] > 0.25 and r["posture"] == "standing" and not r["aborted"], "all four done, moved > 25 cm"),
    ("queue empty after sitting -> keeps sitting for 2 s", "standing", [SS("sit")],
     lambda r: r["idle"][0] == "sitstand" and r["after"] == "sitting", "still sitting after 2 s idle"),
    ("kick (no ball in this scene: only checks it does not fall)", "standing", [Kick(action="kick", target="forward")],
     lambda r: r["posture"] == "standing" and not r["aborted"], "no fall, ends standing"),
    ("queue empty after walking -> stands, does not wander", "standing", [W(0.3)],
     lambda r: r["idle"][0] == "walk" and r["drift"] < 0.03, "drift < 3 cm over 2 s after settling"),
]


def selftest() -> int:
    sim = S.DuckSim(f"{S.config.SCENES}/scene_allcollisions.xml")
    n_bad = 0
    for desc, start, plan, check, what in CASES:
        sim.reset()
        if start == "sitting":
            for _ in range(200):
                sim.step("sitstand", [1.0] + [0.0] * 12)
        for _ in range(25):
            sim.step(*(("sitstand", [1.0] + [0.0] * 12) if start == "sitting" else ("walk", S.zeros())))
        log: list = []
        steps = insert_preconditions(plan, sense_posture(sim), log)
        r = run_steps(sim, steps, log)
        # settle 0.5 s before measuring drift: the pet gait slides ~5 cm when braking from 0.3 m/s, then 0 cm.
        # The test is about not wandering *after* stopping; the braking distance is printed separately.
        xb, yb, _ = sim.xy_yaw()
        hold(sim, r["idle"], 0.5)
        x0, y0, _ = sim.xy_yaw()
        r["brake"] = math.hypot(x0 - xb, y0 - yb)
        hold(sim, r["idle"], 2.0)
        x1, y1, _ = sim.xy_yaw()
        r["after"], r["drift"] = sense_posture(sim), math.hypot(x1 - x0, y1 - y0)
        ok = check(r)
        n_bad += not ok
        print(f"{'ok ' if ok else 'BAD'} {desc} (want: {what})\n     moved {r['dist']:.2f} m, turned {r['dyaw']:+.0f} deg, "
              f"{r['secs']:.1f} s, ends {r['posture']}, after idle {r['after']} (brake {r['brake'] * 100:.1f} cm, "
              f"drift after settling {r['drift'] * 100:.1f} cm)  log: {log}")
    print(f"\n{len(CASES) - n_bad}/{len(CASES)}")
    return n_bad


# ─────────────────────────────────────────────────────────────
# interactive: type -> LLM -> the duck moves -> video
# ─────────────────────────────────────────────────────────────
def chat(video: str):
    from ..planner.llm import plan_with_retry
    sim = S.DuckSim(f"{S.config.SCENES}/scene_allcollisions.xml", video)
    hold(sim, ("walk", S.zeros()), 1.0)
    print("Type a sentence (empty line to finish and save the video).\n")
    try:
        while True:
            sentence = input("you> ").strip()
            if not sentence:
                break
            posture = sense_posture(sim)
            res = plan_with_retry(sentence, posture if posture in ("standing", "sitting") else "fallen")
            log = list(res.log)
            print(f"   duck: {posture}   LLM: {res.status} ({res.llm_calls} call(s))")
            for st in res.steps:
                print(f"   plan: {st.model_dump_json()}")
            r = run_steps(sim, res.steps, log)
            hold(sim, r["idle"], 1.0)
            for line in log:
                print(f"   log: {line}")
            print(f"   result: moved {r['dist']:.2f} m, turned {r['dyaw']:+.0f} deg, {r['secs']:.1f} s, now {sense_posture(sim)}\n")
    finally:
        sim.close()
        print(f"video: {video}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--video", default="chat.mp4")
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(selftest())
    chat(args.video)
