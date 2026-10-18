"""A*: the shortest path on the occupancy grid from start to goal.

    1. occupancy grid -> 2. inflate 15 cm (grid.py) -> 3. A* (this file) -> 4. simplify into waypoints (route.py)

Each cell's score f = g + h: g = the cost already paid from the start, h = the estimated cost left to the goal.
Each round expands the cell with the smallest f; reaching the goal stops the search, and following
"which cell did I come from" back to the start gives the path.

Neighbors: 8 directions, cost 1 straight and sqrt(2) diagonal. Diagonal moves may not cut corners: moving up-right
needs both the right and the upper cell free, or the duck would squeeze through the diagonal gap between two obstacles.
Heuristic: straight-line distance x w; w = 1 is standard A*, w = 0 is Dijkstra.
Ties in the priority queue compare the second element, so a running counter sits between f and the cell.

    python -m pet.nav.astar
"""

import heapq
import itertools
import math

import numpy as np

SQRT2 = math.sqrt(2)


def neighbors(grid, cell, diag=True):
    """Neighbors of cell -> [(neighbor, cost)]. Out of bounds and blocked cells are skipped; no corner cutting."""
    i, j = cell
    nx, ny = grid.shape
    out = []
    steps = [(1, 0), (-1, 0), (0, 1), (0, -1)] + ([(1, 1), (1, -1), (-1, 1), (-1, -1)] if diag else [])
    for di, dj in steps:
        ni, nj = i + di, j + dj
        if not (0 <= ni < nx and 0 <= nj < ny) or grid[ni, nj]:
            continue
        if di and dj and (grid[i + di, j] or grid[i, j + dj]):
            continue                                        # corner cutting
        out.append(((ni, nj), SQRT2 if di and dj else 1.0))
    return out


def heuristic(a, b, w=1.0, diag=True):
    """Estimated cost a -> b: straight-line distance with diagonals, Manhattan distance without. Times w."""
    di, dj = abs(a[0] - b[0]), abs(a[1] - b[1])
    return w * (math.hypot(di, dj) if diag else di + dj)


def reconstruct(came_from, cell):
    """Follow "which cell did I come from" back from the goal -> [start, ..., goal]."""
    path = [cell]
    while cell in came_from:
        cell = came_from[cell]
        path.append(cell)
    return path[::-1]


def astar(grid, start, goal, w=1.0, diag=True):
    """Returns (path list or None, number of expanded cells). grid[i, j] True = blocked; start, goal = (i, j)."""
    if grid[start] or grid[goal]:
        return None, 0
    count = itertools.count()
    open_q = [(heuristic(start, goal, w, diag), next(count), start)]
    g = {start: 0.0}
    came_from = {}
    closed = set()
    while open_q:
        f, _, cur = heapq.heappop(open_q)
        if cur in closed:
            continue                                        # the same cell may be queued several times; handle it once
        if cur == goal:
            return reconstruct(came_from, cur), len(closed)
        closed.add(cur)
        for nxt, cost in neighbors(grid, cur, diag):
            new_g = g[cur] + cost
            if new_g < g.get(nxt, math.inf):
                g[nxt] = new_g
                came_from[nxt] = cur
                heapq.heappush(open_q, (new_g + heuristic(nxt, goal, w, diag), next(count), nxt))
    return None, len(closed)


def path_cost(path):
    return sum(SQRT2 if a[0] != b[0] and a[1] != b[1] else 1.0 for a, b in zip(path, path[1:]))


def selftest() -> int:
    ok = n = 0

    def check(desc, good, got):
        nonlocal ok, n
        ok, n = ok + bool(good), n + 1
        print(f"{'ok ' if good else 'BAD'} {desc}\n     {got}")

    # the example worked by hand: 5x3, the top two cells at x=2 are wall, straight moves only
    g = np.zeros((5, 3), bool)
    g[2, 0] = g[2, 1] = True
    p, k = astar(g, (0, 0), (4, 0), diag=False)
    check("hand-worked example (straight moves only): around through the bottom row -> 8 steps",
          p is not None and len(p) - 1 == 8 and p[0] == (0, 0) and p[-1] == (4, 0), f"path {p}")
    g = np.zeros((10, 10), bool)
    p, _ = astar(g, (0, 0), (9, 9))
    check("empty grid, opposite corner -> 9 diagonal steps, cost 9*sqrt(2)", p is not None and len(p) == 10 and abs(path_cost(p) - 9 * SQRT2) < 1e-9,
          f"{len(p) - 1} steps, cost {path_cost(p):.3f}")
    g = np.zeros((7, 7), bool)
    g[3, :] = True
    p, _ = astar(g, (0, 3), (6, 3))
    check("a wall splits the grid in two -> unreachable (None)", p is None, f"{p}")
    g = np.zeros((3, 3), bool)
    g[1, 0] = g[0, 1] = True
    p, _ = astar(g, (0, 0), (1, 1))
    check("diagonal gap (walls right and above) -> no corner cutting -> unreachable", p is None, f"{p}")
    g = np.zeros((40, 40), bool)
    g[20, 5:35] = True
    pa, ka = astar(g, (5, 20), (35, 20), w=1.0)
    pd, kd = astar(g, (5, 20), (35, 20), w=0.0)
    check("same problem: A* (w=1) and Dijkstra (w=0) find equally short paths, A* expands far fewer cells",
          pa and pd and abs(path_cost(pa) - path_cost(pd)) < 1e-9 and ka < kd,
          f"A* expanded {ka}, Dijkstra {kd}; length {path_cost(pa):.2f} vs {path_cost(pd):.2f}")
    print(f"\n{ok}/{n}")
    return n - ok


if __name__ == "__main__":
    raise SystemExit(selftest())
