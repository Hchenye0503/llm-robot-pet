"""Start with an empty map and walk to a goal: draw the map while walking, replan only when the path is blocked.

    every 0.2 s: one ToF scan -> three-layer map (mapping.py: ToF + cliff + bump)
                 -> inflate 15 cm -> is the rest of the path still clear? (path_blocked) -> only if not, re-run A*
                    (unknown = optimistically free)
    every 20 ms: walk toward the next waypoint (body/goto.py controller)
    bumped into something that is not the goal -> mark it in the bump layer -> turn away -> replan
    (like a robot vacuum's bumper)

Why replan only when blocked instead of every time
    every time: the path keeps shifting and the duck sways left and right; every bit of map noise changes the route
    only when blocked: a stable path; but when the map improves (a shorter way appears) it is not taken -> not always shortest

    python -m pet.nav.explore            # selftest
    python -m pet.nav.explore --demo     # two scenarios + map figure
"""

import math

import numpy as np

from . import grid as M
from .route import clear, plan

SCAN_EVERY = 10          # scan every 10 control steps (0.2 s)
WAYPOINT_R = 0.15        # switch to the next waypoint within this distance
GRACE = M.INFLATE_R      # grace distance: the first leg is not checked within this distance of the duck
#   Walking along a wall puts the duck itself inside the wall's 15 cm inflation, so the first leg was "blocked" from
#   the start and the path was replanned every 0.2 s (118 times in 79 s). "No replanning unless bumped" would waste
#   what the ToF sees 1.5 m ahead -> ignore only the first INFLATE_R (the duck's own safety margin).


def path_blocked(grid, pos, pts) -> bool:
    """From pos through the remaining waypoints pts in order: is that path still clear on the new grid? Blocked -> True.
    No waypoints -> nothing to check -> False."""
    L = math.hypot(pos[0] - pts[0][0], pos[1] - pts[0][1]) if pts else 0.0
    if L > GRACE:  # the first GRACE meters from the duck are not checked
        check_pts = [(pos[0] + GRACE * (pts[0][0] - pos[0]) / L, pos[1] + GRACE * (pts[0][1] - pos[1]) / L)] + pts
    else:
        check_pts = [pos] + pts
    for a, b in zip(check_pts, check_pts[1:]):
        if not clear(grid, a, b):
            return True
    return False


def explore_to(sim, goal_xy, stop_dist=0.3, goal_body=None, timeout_s=90.0, log=None, trace=None):
    """From an empty map to goal_xy. Returns a summary; trace (a list) receives the (x, y) the duck walked, for plotting."""
    import mujoco
    from mjlab_microduck.sim.tof import Tof
    from ..body import sim as S
    from ..body.goto import _detour_side, _turn_by, goto_command_hyst, to_robot_frame
    from .mapping import LogOddsMap, scan
    log = [] if log is None else log
    tof = Tof(sim.model, mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_SITE, "tof"))
    lmap = LogOddsMap()
    t0, k, turning = sim.t, 0, False
    replans = bumps = 0
    pts = None
    falls = 0
    while sim.t - t0 < timeout_s:
        x, y, yaw = sim.xy_yaw()
        if trace is not None and k % 5 == 0:
            trace.append((x, y))
        if math.dist((x, y), goal_xy) <= stop_dist:
            break
        if k % SCAN_EVERY == 0 or pts is None:
            scan(tof, sim.data, lmap)
            grid = M.inflate(lmap.blocked(), M.INFLATE_R)
            if pts is None or path_blocked(grid, (x, y), pts):
                new = plan(grid, (x, y), goal_xy)
                if new is None:
                    log.append(f"t={sim.t - t0:5.1f} goal unreachable on the map -> stop")
                    break
                if pts is not None:
                    replans += 1
                    log.append(f"t={sim.t - t0:5.1f} path blocked -> replan ({replans}): " + " -> ".join(f"({p[0]:+.1f},{p[1]:+.1f})" for p in new))
                pts = new
        while len(pts) > 1 and math.dist((x, y), pts[0]) < WAYPOINT_R:
            pts.pop(0)
        last = len(pts) == 1
        rx, ry = to_robot_frame(sim, *pts[0])
        vx, wz, done, turning = goto_command_hyst(rx, ry, stop_dist if last else 0.05, turning)
        if done and last:
            break
        c = S.zeros()
        c[0], c[2] = vx, math.radians(wz)
        sim.step("walk", c)
        k += 1
        hit = sim.touching_what()
        if hit and hit != goal_body:
            bumps += 1
            lmap.bumped(x, y, yaw)
            log.append(f"t={sim.t - t0:5.1f} bumped into {hit} -> marked in the bump layer, turn away, replan")
            # the pet gait does not walk backward: turn away instead of stepping back
            _turn_by(sim, 90 * _detour_side(sim, hit))
            pts = None
        elif hit and hit == goal_body:
            break
        if sim.upright() < 0.5:
            falls += 1
            log.append(f"t={sim.t - t0:5.1f} fell -> stop")
            break
    for _ in range(25):
        sim.step("walk", S.zeros())
    x, y, _ = sim.xy_yaw()
    return dict(reached=math.dist((x, y), goal_xy) <= stop_dist + 0.15 or (goal_body and sim.touching(goal_body)),
                dist=math.dist((x, y), goal_xy), secs=sim.t - t0, replans=replans, bumps=bumps, falls=falls, lmap=lmap)


def selftest() -> int:
    ok = n = 0

    def check(desc, good, got):
        nonlocal ok, n
        ok, n = ok + bool(good), n + 1
        print(f"{'ok ' if good else 'BAD'} {desc}\n     {got}")

    save = (M.X0, M.Y0)
    M.X0, M.Y0 = 0.0, 0.0
    try:
        g = np.zeros((40, 40), bool)                             # 2 m x 2 m open floor
        check("open floor: path clear -> False", path_blocked(g, (0.2, 0.2), [(1.0, 1.0), (1.8, 0.2)]) is False, "")
        g[30, 0:20] = True                                       # a new wall on the second leg (x = 1.5 m, y 0-1 m)
        check("second leg blocked by a new wall -> True (checking only the first leg would miss it)",
              path_blocked(g, (0.2, 0.2), [(1.0, 1.0), (1.8, 0.2)]) is True, "")
        check("the wall only blocks what is already behind -> False", path_blocked(g, (1.6, 0.5), [(1.8, 0.2)]) is False, "already past the wall")
        check("no waypoints -> False", path_blocked(g, (0.2, 0.2), []) is False, "")
        g = np.zeros((40, 40), bool)
        g[0:40, 0:3] = True                                      # a wall along the south (y 0-0.15 m)
        wall = M.inflate(g, M.INFLATE_R)                         # after inflation y < 0.30 is blocked
        check("duck inside the wall's inflation (y = 0.25), walking north away from it -> not blocked (grace distance)",
              path_blocked(wall, (1.0, 0.25), [(1.0, 1.5)]) is False, "")
        wall2 = wall.copy()
        wall2[10:30, 24:27] = True                               # a new wall 1 m ahead (y 1.2-1.35 m)
        check("same, but a new wall appears 1 m ahead -> still blocked (what the ToF sees early still counts)",
              path_blocked(wall2, (1.0, 0.25), [(1.0, 1.5)]) is True, "")
    finally:
        M.X0, M.Y0 = save
    print(f"\n{ok}/{n}")
    return n - ok


DEMOS = [  # (name, start (x, y, yaw), goal, goal body, stage)
    ("a: around the coffee table to the owner", (-2.6, -0.45, 280), (-2.45, -2.0), "owner", "static"),
    ("b: living room to the entrance", (-2.0, -1.5, 0), (0.05, 2.2), None, "live"),
]


def demo(out="explore.png"):
    """Two scenarios from an empty map -> the map the duck drew, the path it walked, and the true furniture outline."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ..body import sim as S
    from ..world import owner as O
    from ..world import stage as T
    fig, axes = plt.subplots(1, len(DEMOS), figsize=(7 * len(DEMOS), 7.5))
    for ax, (name, start, goal, body, stage) in zip(np.atleast_1d(axes), DEMOS):
        if stage == "static":
            sim = S.DuckSim(T.SCENE, None, spec_edit=T.stage, start=start)
        else:
            sim = S.DuckSim(T.SCENE, None, spec_edit=O.stage_live, start=start)
            O.Owner(sim, pose="stand", at=O.AWAY, yaw=0)
        for _ in range(50):
            sim.step("walk", S.zeros())
        truth = M.build(sim.model)
        log, trace = [], []
        r = explore_to(sim, goal, goal_body=body, log=log, trace=trace)
        sim.close()
        lm = r["lmap"]
        img = np.full(truth.T.shape + (3,), 0.80)                 # gray = unseen
        img[lm.known_free().T] = (1.0, 1.0, 1.0)                  # white = seen, free
        img[lm.occupied().T] = (0.20, 0.22, 0.25)                 # black = obstacle seen by the ToF
        img[(lm.H > 0.5).T] = (0.75, 0.25, 0.20)                  # red = cliff (hole)
        img[lm.B.T] = (0.95, 0.60, 0.10)                          # orange = bumped
        ax.imshow(img, origin="lower", extent=[M.X0, M.X1, M.Y0, M.Y1], interpolation="nearest")
        ax.contour(np.linspace(M.X0, M.X1, truth.shape[0]), np.linspace(M.Y0, M.Y1, truth.shape[1]), truth.T.astype(float),
                   levels=[0.5], colors="#2E86C1", linewidths=0.6)
        tr = np.array(trace)
        ax.plot(tr[:, 0], tr[:, 1], "-", color="#27AE60", lw=2.2, label="path walked")
        ax.plot(start[0], start[1], "s", color="#27AE60", ms=10, label="start")
        ax.plot(*goal, "*", color="#C0392B", ms=16, label="goal")
        ax.set_title(f"{name}: {'reached' if r['reached'] else 'not reached'}, {r['secs']:.0f} s, "
                     f"{r['replans']} replans, {r['bumps']} bumps, {r['falls']} falls")
        ax.set_xlim(-4.2, 0.9)
        ax.set_ylim(-3.2, 3.2)
        ax.legend(loc="lower right", fontsize=8)
        print(f"{name}: {r['reached']}, {r['secs']:.0f} s, replans {r['replans']}, bumps {r['bumps']}, falls {r['falls']}")
    fig.suptitle("From an empty map (white = seen free, black = ToF obstacle, red = cliff, orange = bumped, gray = unseen; "
                 "blue = true furniture)")
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    print("figure:", out)


if __name__ == "__main__":
    import sys
    demo() if "--demo" in sys.argv else sys.exit(selftest())
