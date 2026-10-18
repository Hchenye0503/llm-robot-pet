"""From an A* path to a route the duck can actually walk: simplify into waypoints, then goto one leg at a time.

    1. occupancy grid -> 2. inflate (grid.py) -> 3. A* (astar.py) -> 4. simplify + 5. follow (this file)

Simplify (line-of-sight shortcut, a.k.a. string pulling): an A* path is a cell-by-cell zigzag. From the start, find
the farthest cell still in straight-line sight (every point on the line outside the inflated area); that is a
waypoint; repeat. The zigzag becomes a few straight legs for the walk-to-target controller (body/goto.py).

When it is used (goto_planned)
    the straight line does not touch the inflated area -> plain goto, as before
    the straight line is blocked -> A* -> simplify -> intermediate waypoints with stop 0.15 m, the last leg with stop_dist
    the map only knows the apartment: other scenes (no apartment walls) -> plain goto
    targets often sit next to furniture (the owner by the sofa) -> the goal cell is inside the inflated area -> the
    nearest free cell is the A* goal; the last goto still walks toward the real target
    bump-and-detour stays inside goto, so things the map does not have (toys, balls) are still handled

    python -m pet.nav.route
"""

import math

import numpy as np

from . import grid as M
from .astar import astar

_CACHE = {}


def grid_for(model):
    """Inflated grid of this scene (computed once per model); not the apartment -> None."""
    key = id(model)
    if key not in _CACHE:
        import mujoco
        is_apartment = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "liv_table") >= 0
        _CACHE[key] = M.inflate(M.build(model), M.INFLATE_R) if is_apartment else None
    return _CACHE[key]


_RAW = {}


def occ_for(model):
    """*Uninflated* grid of this scene (for perception's line-of-sight check, grid.view_blocked); not the apartment -> None."""
    key = id(model)
    if key not in _RAW:
        import mujoco
        _RAW[key] = M.build(model) if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "liv_table") >= 0 else None
    return _RAW[key]


def feet_hidden(model, cam_xy, land_xy) -> bool:
    """When the floor-plane and known-size estimates disagree: is that line of sight really blocked on the map?
    No map (not the apartment) -> assume blocked, as the rule did before maps."""
    occ = occ_for(model)
    return True if occ is None or land_xy is None else M.view_blocked(occ, cam_xy, land_xy)


def clear(grid, a, b) -> bool:
    """Straight line a -> b (world coordinates), sampled every RES/2: no point in the inflated area or outside -> True."""
    n = max(2, int(math.dist(a, b) / (M.RES / 2)) + 1)
    nx, ny = grid.shape
    for k in range(n + 1):
        x = a[0] + (b[0] - a[0]) * k / n
        y = a[1] + (b[1] - a[1]) * k / n
        i, j = M.to_cell(x, y)
        if not (0 <= i < nx and 0 <= j < ny) or grid[i, j]:
            return False
    return True


def simplify(grid, cells, start_xy):
    """A* cells -> a few waypoints (world coordinates, start excluded)."""
    pts = [M.to_xy(*c) for c in cells]
    pts[0] = start_xy   # the start is the current position, inside the inflated area or not
    out, k = [], 0
    while k < len(pts) - 1:
        far = k + 1
        for m in range(len(pts) - 1, k, -1):
            if clear(grid, pts[k], pts[m]):
                far = m
                break
        out.append(pts[far])
        k = far
    return out


def nearest_free(grid, cell):
    if not grid[cell]:
        return cell
    free = np.argwhere(~grid)
    return tuple(free[np.argmin(((free - np.array(cell)) ** 2).sum(1))])


def plan(grid, start_xy, goal_xy):
    """World coordinates -> list of waypoints (start excluded); unreachable -> None."""
    s = nearest_free(grid, M.to_cell(*start_xy))
    g = nearest_free(grid, M.to_cell(*goal_xy))
    cells, _ = astar(grid, s, g)
    return None if cells is None else simplify(grid, cells, start_xy)


def goto_planned(sim, target, stop_dist=0.4, log=None, **kw):
    """Same interface as body.goto.goto; follows A* waypoints first when furniture blocks the straight line."""
    from ..body.goto import goto
    grid = grid_for(sim.model)
    if grid is None:
        return goto(sim, target, stop_dist=stop_dist, **kw)
    x, y, _ = sim.xy_yaw()
    d = math.dist((x, y), target)
    # check only up to where the duck will stop: the target itself is often next to furniture (inside the inflated area)
    stop_at = (target[0] + (x - target[0]) * stop_dist / max(d, 1e-6), target[1] + (y - target[1]) * stop_dist / max(d, 1e-6))
    if d <= stop_dist or clear(grid, (x, y), stop_at):
        return goto(sim, target, stop_dist=stop_dist, **kw)
    pts = plan(grid, (x, y), stop_at)
    if pts is None:
        if log is not None:
            log.append(f"A*: ({x:+.2f},{y:+.2f}) -> ({target[0]:+.2f},{target[1]:+.2f}) unreachable -> walk straight")
        return goto(sim, target, stop_dist=stop_dist, **kw)
    if log is not None:
        log.append("A*: straight line blocked -> waypoints " + " -> ".join(f"({p[0]:+.2f},{p[1]:+.2f})" for p in pts[:-1]))
    leg_kw = {k: v for k, v in kw.items() if k in ("bump", "goal_body", "max_detours", "walk_v", "head")}
    for p in pts[:-1]:
        r = goto(sim, p, stop_dist=0.15, **leg_kw)
        if r.get("fell") or (r.get("bumped") and r["bumped"] == kw.get("goal_body")):
            return r
    return goto(sim, target, stop_dist=stop_dist, **kw)


def selftest() -> int:
    ok = n = 0

    def check(desc, good, got):
        nonlocal ok, n
        ok, n = ok + bool(good), n + 1
        print(f"{'ok ' if good else 'BAD'} {desc}\n     {got}")

    save = (M.X0, M.Y0)
    M.X0, M.Y0 = 0.0, 0.0
    try:
        g = np.zeros((40, 40), bool)                         # 2 m x 2 m; a wall at x = 1 m, y 0-1.5 m
        g[20, 0:30] = True
        check("straight line not blocked -> clear", clear(g, (0.2, 1.8), (1.8, 1.8)), "")
        check("straight line through the wall -> not clear", not clear(g, (0.5, 0.5), (1.5, 0.5)), "")
        pts = plan(g, (0.5, 0.5), (1.5, 0.5))
        check("around the wall: few waypoints (simplified), every leg clear",
              pts is not None and len(pts) <= 4 and all(clear(g, a, b) for a, b in zip([(0.5, 0.5)] + pts, pts)),
              f"{[(round(p[0], 2), round(p[1], 2)) for p in pts] if pts else pts}")
    finally:
        M.X0, M.Y0 = save
    print(f"\n{ok}/{n}")
    return n - ok


if __name__ == "__main__":
    raise SystemExit(selftest())
