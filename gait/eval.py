"""Pass criteria for the pet gait, C1-C7, at 7.4 V and 6.5 V battery (CPU only, no training).

The criteria were written before training (C4's fall budget, C6 and C7 were added later and are marked).
Battery voltage is changed in the simulated actuator model; the same ONNX is used.

Operational definitions
    speed  = body speed along its own heading, averaged over 0.48 s (one gait cycle)
    C1     command 0.05 for 6 s, mean of the last 4 s; pass = error <= 0.025
    C2     commands 0.05 / 0.1 / 0.2 / 0.3 / 0.4, 6 s each, mean of the last 4 s; pass = error <= 0.02 + 10% x command
    C3     0 -> 0.4 (4 s) -> 0 (3 s). Settling time = the first moment the *centered* 0.48 s average (0.24 s each side,
           no lag) enters the tolerance band and stays there. Measured 24 times, the command shifted by 20 ms each time
           to cover a whole gait cycle (braking time depends on where the feet are when the command arrives: 0.40-0.56 s
           for the same policy). pass = all 24 starts and stops <= 0.48 s, no fall
    C4     N x 60 s random scripts: forward 0-0.4 (30% chance of a 0 <-> 0.4 step, 10% of 0.05), sideways +-0.3,
           turning +-1.0 rad/s, changed every 2-6 s. pass = <= 2 falls per 600 (one fall per 5+ hours of walking).
           The budget was set after seeing an earlier version's 3/1800, so it is lenient toward that version.
    C5     command 0 for 10 s, last 8 s. pass = mean speed < 0.01 and feet not lifting
           (ankle height variation < 1 mm stands in for contact counts, which this CPU pipeline does not have)
    C6     (added in v5) forward commands 0.1 / 0.2 / 0.4 from standing, 10 s; heading change / 10 s.
           pass = <= 1 deg/s at every point (the planner corrects the rest)
    C7     (added in v5, after a version passed C1-C5 but could no longer turn in place) turning in place +-0.5 and
           +-1.0 rad/s, and walking 0.2 while turning +-0.5, 6 s each, mean yaw rate of the last 4 s.
           pass = error <= 0.05 + 10% x |turn command| (the same formula as the reward)

    python -m gait.eval                         # the pet gait (config.PET_GAIT_ONNX)
    python -m gait.eval path/to/policy.onnx
    python -m gait.eval official                # the official velstand policy, for comparison
    options: --n 600 (C4 episodes, slow)  --volts 7.4 6.5  --skip-c4
"""

import argparse

import mujoco
import numpy as np
import onnxruntime as ort

from pet import config
from pet.body import sim as S
from pet.body.scheduler import is_fallen

WIN = 24                       # 0.48 s
STOP_BAND = 0.05               # C3: "stopped" for braking
YAW_DRIFT_MAX = 1.0            # C6: deg/s allowed walking straight
FALLS_MAX = 2                  # C4: falls allowed per 600 episodes (scaled for other N)
DT = S.CONTROL_DT


def tol(cmd):
    return 0.02 + 0.10 * abs(cmd)


def forward_speed(sim):
    """Body speed along its own heading (m/s): world velocity projected on the trunk's horizontal x axis."""
    b = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    x = sim.data.xmat[b].reshape(3, 3)[:, 0].copy()
    x[2] = 0.0
    x /= np.linalg.norm(x)
    return float(sim.data.qvel[0:3] @ x)


def make_sim(pol, onnx, volts):
    """DuckSim with the battery at `volts` (DuckSim uses 7.4)."""
    orig = S.ip.load_bam_model
    S.ip.load_bam_model = lambda kp, vin, max_current: orig(kp, volts, max_current)
    try:
        sim = S.DuckSim(S.DEFAULT_SCENE, None)
    finally:
        S.ip.load_bam_model = orig
    if onnx is not None:
        sim.sessions[pol] = ort.InferenceSession(str(onnx))
    return sim


def cmd_vec(vx=0.0, vy=0.0, wz=0.0):
    c = S.zeros()
    c[0], c[1], c[2] = vx, vy, wz
    return c


def steady(pol, onnx, volts, vx, secs=6.0, last=4.0):
    sim = make_sim(pol, onnx, volts)
    for _ in range(50):
        sim.step(pol, cmd_vec())
    v = []
    for _ in range(int(secs / DT)):
        sim.step(pol, cmd_vec(vx))
        v.append(forward_speed(sim))
    fallen = is_fallen(sim)
    sim.close()
    return float(np.mean(v[-int(last / DT):])), fallen


def settle_time(sm, cmd, band):
    """Averaged speed sm (from the command onward): first time it enters cmd +- band and stays (nan if never)."""
    ok = np.abs(sm - cmd) <= band
    for i in range(len(ok)):
        if ok[i:].all():
            return (i + 1) * DT
    return float("nan")


def c3(pol, onnx, volts, phases=24):
    """Start / stop measured `phases` times, each shifted by one control step (20 ms) to cover a gait cycle."""
    go, stop, fallen = [], [], False
    for k in range(phases):
        sim = make_sim(pol, onnx, volts)
        for _ in range(50 + k):
            sim.step(pol, cmd_vec())
        n_go, v = int(4.0 / DT) + k, []
        for n, vx in ((n_go, 0.4), (int(3.0 / DT), 0.0)):
            for _ in range(n):
                sim.step(pol, cmd_vec(vx))
                v.append(forward_speed(sim))
        fallen |= is_fallen(sim)
        sim.close()
        # the centered average needs speeds from before each command -> average the whole run, then cut it
        sm = np.convolve(np.array(v), np.ones(WIN) / WIN, mode="same")
        go.append(settle_time(sm[:n_go - WIN // 2], 0.4, tol(0.4)))     # the last half window already sees the stop
        stop.append(settle_time(sm[n_go:-WIN // 2], 0.0, STOP_BAND))
    return np.array(go), np.array(stop), fallen


def c4(pol, onnx, volts, n, seed=0):
    rng = np.random.default_rng(seed)
    sim = make_sim(pol, onnx, volts)
    falls, when = 0, []
    for ep in range(n):
        sim.reset()
        for _ in range(50):
            sim.step(pol, cmd_vec())
        t, vx, down = 0.0, 0.0, False
        while t < 60.0 and not down:
            r = rng.random()
            if r < 0.3:
                vx = 0.0 if vx > 0.2 else 0.4                       # 0 <-> 0.4 step
                vy = wz = 0.0
            elif r < 0.4:
                vx, vy, wz = 0.05, 0.0, 0.0                         # the slowest point
            else:
                vx, vy, wz = rng.uniform(0, 0.4), rng.uniform(-0.3, 0.3), rng.uniform(-1.0, 1.0)
            for _ in range(int(rng.uniform(2.0, 6.0) / DT)):
                sim.step(pol, cmd_vec(vx, vy, wz))
                t += DT
                if is_fallen(sim):
                    down = True
                    break
                if t >= 60.0:
                    break
        if down:
            falls += 1
            when.append((ep, round(t, 1), round(vx, 2), round(vy, 2), round(wz, 2)))
        if (ep + 1) % 10 == 0:
            print(f"      C4 {volts} V: {ep + 1}/{n} episodes, {falls} falls", flush=True)
    sim.close()
    return falls, when


def c5(pol, onnx, volts):
    sim = make_sim(pol, onnx, volts)
    feet = [mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, b) for b in ("ankle_left", "ankle_right")]
    v, z = [], []
    for i in range(int(10.0 / DT)):
        sim.step(pol, cmd_vec())
        if i >= int(2.0 / DT):
            v.append(forward_speed(sim))
            z.append([sim.data.xpos[b][2] for b in feet])
    fallen = is_fallen(sim)
    sim.close()
    z = np.array(z)
    return abs(float(np.mean(v))), float((z.max(0) - z.min(0)).max() * 1000), fallen


def c6(pol, onnx, volts, vx, secs=10.0):
    """Walk straight for secs: (mean deg/s, deg turned in the first second, fallen). Heading is unwrapped."""
    sim = make_sim(pol, onnx, volts)
    for _ in range(50):
        sim.step(pol, cmd_vec())
    yaw0 = yaw = sim.xy_yaw()[2]
    first = 0.0
    for i in range(int(secs / DT)):
        sim.step(pol, cmd_vec(vx))
        yaw += (sim.xy_yaw()[2] - yaw + 180) % 360 - 180
        if i + 1 == int(1.0 / DT):
            first = yaw - yaw0
    fallen = is_fallen(sim)
    sim.close()
    return (yaw - yaw0) / secs, first, fallen


def c7(pol, onnx, volts, vx, wz, secs=6.0, last=4.0):
    """Turn command: (mean yaw rate of the last `last` seconds in rad/s, fallen)."""
    sim = make_sim(pol, onnx, volts)
    for _ in range(50):
        sim.step(pol, cmd_vec())
    yaw = sim.xy_yaw()[2]
    n, n0, y0 = int(secs / DT), int((secs - last) / DT), 0.0
    for i in range(n):
        sim.step(pol, cmd_vec(vx, 0.0, wz))
        yaw += (sim.xy_yaw()[2] - yaw + 180) % 360 - 180
        if i + 1 == n0:
            y0 = yaw
    fallen = is_fallen(sim)
    sim.close()
    return float(np.radians(yaw - y0) / last), fallen


def mark(ok):
    return "PASS" if ok else "FAIL"


def main():
    global STOP_BAND
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="?", default=None, help="an ONNX path, or 'official' (velstand); default: the pet gait")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--volts", type=float, nargs="+", default=[7.4, 6.5])
    ap.add_argument("--skip-c4", action="store_true")
    ap.add_argument("--stop-band", type=float, default=None, help="C3 braking band (default STOP_BAND)")
    a = ap.parse_args()
    if a.stop_band is not None:
        STOP_BAND = a.stop_band
    if a.which == "official":
        pol, onnx = "recover", None                    # official velstand.onnx ("recover" in DuckSim)
    else:
        pol, onnx = "petgait", (a.which or config.PET_GAIT_ONNX)

    summary = []
    for volts in a.volts:
        print(f"\n======== {a.which or 'pet gait'} @ {volts} V ========", flush=True)
        res = {}
        print("  C1 / C2 tracking (mean of the last 4 s, m/s)")
        c2_ok = True
        for vx in (0.05, 0.1, 0.2, 0.3, 0.4):
            v, fallen = steady(pol, onnx, volts, vx)
            ok = abs(v - vx) <= tol(vx) and not fallen
            c2_ok &= ok
            if vx == 0.05:
                res["C1"] = ok
            print(f"      command {vx:4.2f} -> {v:6.3f}  error {abs(v - vx):.3f} (allowed {tol(vx):.3f}) {mark(ok)}{' FELL' if fallen else ''}", flush=True)
        res["C2"] = c2_ok

        go, stop, fallen = c3(pol, onnx, volts)
        n_go, n_stop = int(np.sum(go <= 0.48)), int(np.sum(stop <= 0.48))      # nan (never) does not pass
        res["C3"] = n_go == len(go) and n_stop == len(stop) and not fallen
        print(f"  C3 steps ({len(go)} phases; all must be <= 0.48 s){' FELL' if fallen else ''} {mark(res['C3'])}")
        for name, x, n in (("start", go, n_go), ("stop", stop, n_stop)):
            if np.isnan(x).all():
                print(f"      {name}: never within tolerance  pass 0/{len(x)}", flush=True)
            else:
                print(f"      {name}: median {np.nanmedian(x):.2f} s, fastest {np.nanmin(x):.2f}, slowest {np.nanmax(x):.2f}  pass {n}/{len(x)}", flush=True)

        spd, lift, fallen = c5(pol, onnx, volts)
        res["C5"] = spd < 0.01 and lift < 1.0 and not fallen
        print(f"  C5 command 0: mean speed {spd:.4f} (< 0.01), ankle variation {lift:.1f} mm (< 1) {mark(res['C5'])}", flush=True)

        print("  C6 walking straight from standing, 10 s (<= 1 deg/s)")
        c6_ok = True
        for vx in (0.1, 0.2, 0.4):
            rate, first, fallen = c6(pol, onnx, volts, vx)
            ok = abs(rate) <= YAW_DRIFT_MAX and not fallen
            c6_ok &= ok
            print(f"      command {vx:4.2f} -> {rate:+5.2f} deg/s (first second {first:+5.1f} deg) {mark(ok)}{' FELL' if fallen else ''}", flush=True)
        res["C6"] = c6_ok

        print("  C7 turning (mean yaw rate of the last 4 s, rad/s)")
        c7_ok = True
        for vx, wz in ((0.0, 0.5), (0.0, -0.5), (0.0, 1.0), (0.0, -1.0), (0.2, 0.5), (0.2, -0.5)):
            w, fallen = c7(pol, onnx, volts, vx, wz)
            ok = abs(w - wz) <= 0.05 + 0.10 * abs(wz) and not fallen
            c7_ok &= ok
            name = "in place" if vx == 0.0 else f"walking {vx}"
            print(f"      {name} turn {wz:+4.1f} -> {w:+5.2f}  error {abs(w - wz):.2f} (allowed {0.05 + 0.10 * abs(wz):.2f}) {mark(ok)}{' FELL' if fallen else ''}", flush=True)
        res["C7"] = c7_ok

        if not a.skip_c4:
            falls, when = c4(pol, onnx, volts, a.n)
            res["C4"] = falls <= FALLS_MAX * a.n / 600
            print(f"  C4 falls: {falls}/{a.n} (<= {FALLS_MAX}/600) {mark(res['C4'])}")
            for w in when[:10]:
                print(f"      episode {w[0]}, fell at {w[1]} s; command vx={w[2]} vy={w[3]} wz={w[4]}")
        summary.append((volts, res))

    print("\n======== summary ========")
    for volts, res in summary:
        print(f"  {volts} V: " + "  ".join(f"{k} {mark(v)}" for k, v in sorted(res.items())))


if __name__ == "__main__":
    main()
