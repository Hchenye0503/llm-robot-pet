"""Mapping from an empty map: the ToF sensor draws the map while the duck walks (occupancy grid mapping, log-odds).

    every 0.2 s: 1. one ToF scan (upstream sim/tof.py: 8x8 rays, 45 deg, 4 m, with noise)
                 2. rays -> map (this file): cells a ray *passes through* = evidence of free space,
                    the cell it *hits* = evidence of an obstacle
                 3. is the current path still walkable? no -> re-run A* on the new map, unknown = optimistically free (explore.py)
                 4. walk a little along the path -> back to 1
This is mapping with known position plus replanning: the "M" half of SLAM, not SLAM (the position comes from the simulator).

A cell hit once and then passed through: which to believe? Trust a state only after it is seen twice (the same
debouncing as the event detector), generalized to log-odds: each cell has an evidence score L; a hit adds L_OCC,
a pass adds L_FREE (negative), clamped to +-L_MAX. L > T_OCC = obstacle, L < T_FREE = free, in between = unknown
(unseen cells start at 0). Debouncing and hysteresis (two thresholds) are both built in.
Why +0.85 for a hit but only -0.4 for a pass: a hit is more reliable; a ray grazing a cell's corner counts as passing
through, yet that cell may contain something.

Three layers: the ToF layer (L), a cliff layer (H) and a bump layer (B), like a robot vacuum.
    only the part of a ray at the duck's height counts: above BAND_HI (30 cm) a ray cannot say the floor is clear
    a ray hitting the floor (below BAND_LO) -> that stretch of floor is clear, not an obstacle
    status "no target" (nothing within 4 m) -> the whole ray (at duck height) is free evidence;
    "no measurement" -> no information (upstream: not seen is not absent)

    python -m pet.nav.mapping
"""

import math

import numpy as np

from . import grid as M

L_OCC, L_FREE = 0.85, -0.40      # evidence for a hit / a pass
L_MAX = 3.5                      # clamp (otherwise a moved chair would need hundreds of passes to clear)
T_OCC, T_FREE = 0.5, -0.5        # decision thresholds (two different ones = hysteresis)


class LogOddsMap:
    def __init__(self):
        self.L = np.zeros(M.shape())             # evidence per cell; 0 = unseen / unknown

    def integrate(self, free_cells, hit_cell):
        """One ray: free_cells = cells it passed through; hit_cell = the cell where it hit an obstacle (None = no obstacle)."""
        for cell in free_cells:
            self.L[cell] = max(-L_MAX, self.L[cell] + L_FREE)
        if hit_cell is not None:
            self.L[hit_cell] = min(L_MAX, self.L[hit_cell] + L_OCC)

    # Two more layers. Why not one: low rays pass under the coffee table top (14-22 cm) and keep giving "free" evidence
    # that erases the hits on the top; rays over the stairwell erase the hole. Each kind of obstacle is cleared only by
    # the observation that can prove it is not there.
    def _extra(self):
        if not hasattr(self, "H"):
            self.H = np.zeros(M.shape())        # cliff layer: + = evidence of a hole, - = the floor was really hit
            self.B = np.zeros(M.shape(), bool)  # bump layer: places the duck bumped into (ToF cannot clear them)

    def hole(self, cell):
        self._extra()
        self.H[cell] = min(L_MAX, self.H[cell] + L_OCC)

    def floor(self, cell):
        self._extra()
        self.H[cell] = max(-L_MAX, self.H[cell] + L_FREE)

    def bumped(self, x, y, yaw_deg, ahead=0.14, radius=0.06):
        """The duck bumped into something: cells within `radius` of the point `ahead` meters in front are marked."""
        self._extra()
        cx, cy = x + ahead * math.cos(math.radians(yaw_deg)), y + ahead * math.sin(math.radians(yaw_deg))
        r = math.ceil(radius / M.RES)
        i0, j0 = M.to_cell(cx, cy)
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                if di * di + dj * dj <= r * r and 0 <= i0 + di < self.B.shape[0] and 0 <= j0 + dj < self.B.shape[1]:
                    self.B[i0 + di, j0 + dj] = True

    def occupied(self):
        return self.L > T_OCC

    def known_free(self):
        return self.L < T_FREE

    def unknown(self):
        return ~self.occupied() & ~self.known_free()

    def blocked(self):
        """For planning: blocked if any layer says so (unknown = optimistically free)."""
        self._extra()
        return self.occupied() | (self.H > T_OCC) | self.B


FREE_H = 0.15       # a ray counts as "no obstacle" evidence only below this height (a ray skimming over the 22 cm
                    # coffee table drew it as free)
CLIFF_M = 0.05      # a downward ray reaching this much farther than where the floor should be -> no floor there = hole
                    # (the stairwell was drawn as free)
FLOOR_Z = 0.02      # a hit within +-2 cm of floor height = the floor (cliff-layer evidence that the floor is there)


def ray_cells(origin, direction, dist, hit):
    """One ray -> (cells passed, obstacle cell or None, hole cell or None, floor-hit cell or None).
    origin and direction are 3D world coordinates, dist in meters; hit = whether the ray hit anything.

    Two blind spots found by a smoke test (turn in place once, compare with the true map):
      1. passing high cannot prove there is no obstacle (table 22 cm, ToF 23 cm: the ray skims over the top)
         -> only the stretch with z <= FREE_H counts
      2. cliffs: a downward ray should hit the floor at t_floor; if it goes farther (or hits nothing) the floor is
         missing -> the landing cell is a hole (how robot vacuums avoid stairs)"""
    ox, oy, oz = origin
    dx, dy, dz = direction
    nx, ny = M.shape()
    inside = lambda c: 0 <= c[0] < nx and 0 <= c[1] < ny
    reach = dist                                                  # how far the ray really went (no hit = MAX_RANGE)
    hole_cell = floor_cell = None
    if dz < -1e-9:
        t_floor = -oz / dz
        if reach > t_floor + CLIFF_M:                             # 2. should have hit the floor but did not -> hole
            hole_cell, reach = M.to_cell(ox + dx * t_floor, oy + dy * t_floor), t_floor
    if hit and hole_cell is None and abs(oz + dz * dist) <= FLOOR_Z:
        floor_cell = M.to_cell(ox + dx * dist, oy + dy * dist)
    # 1. free evidence only for z in [0, FREE_H]
    t_lo, t_hi = 0.0, reach
    if dz > 1e-9:
        t_hi = min(t_hi, (FREE_H - oz) / dz)
    elif dz < -1e-9:
        t_lo = max(0.0, (FREE_H - oz) / dz)
    elif oz > FREE_H:
        t_hi = -1.0                                               # horizontal and above FREE_H -> no evidence at all
    hit_cell = None
    if hit and hole_cell is None:
        hz = oz + dz * dist
        if M.BAND_LO < hz < M.BAND_HI:                           # hit within the duck's height band = obstacle
            hit_cell = M.to_cell(ox + dx * dist, oy + dy * dist)
    cells, seen = [], set()
    step = M.RES / 2
    for k in range(max(0, int((t_hi - t_lo) / step) + 1)):
        t = t_lo + k * step
        c = M.to_cell(ox + dx * t, oy + dy * t)
        if c == hit_cell or c in seen or not inside(c):
            continue
        seen.add(c)
        cells.append(c)
    keep = lambda c: c if c is not None and inside(c) else None
    return cells, keep(hit_cell), keep(hole_cell), keep(floor_cell)


def scan(tof, data, lmap):
    """One ToF scan -> 64 rays into two of the three layers (ToF, cliff). Returns (rays hitting obstacles, rays seeing holes)."""
    from mjlab_microduck.sim.tof import MAX_RANGE, STATUS_NO_TARGET, STATUS_VALID
    origin = data.site_xpos[tof.site].copy()
    world = data.site_xmat[tof.site].reshape(3, 3) @ tof.directions.T
    dist_mm, status = tof.frame(data)
    hits = holes = 0
    for z in range(len(status)):
        if status[z] == STATUS_VALID:
            cells, hc, hole, flo = ray_cells(origin, world[:, z], dist_mm[z] / 1000.0, True)
        elif status[z] == STATUS_NO_TARGET:
            cells, hc, hole, flo = ray_cells(origin, world[:, z], MAX_RANGE, False)
        else:
            continue                                              # no measurement = no information
        lmap.integrate(cells, hc)
        if hole is not None:
            lmap.hole(hole)
        if flo is not None:
            lmap.floor(flo)
        hits += hc is not None
        holes += hole is not None
    return hits, holes


def selftest() -> int:
    ok = n = 0

    def check(desc, good, got):
        nonlocal ok, n
        ok, n = ok + bool(good), n + 1
        print(f"{'ok ' if good else 'BAD'} {desc}\n     {got}")

    a, b, c = (10, 10), (11, 10), (12, 10)
    m = LogOddsMap()
    m.integrate([a, b], c)
    check("one ray: the 2 passed cells move toward free, the hit cell toward obstacle", m.L[a] == L_FREE and m.L[b] == L_FREE and m.L[c] == L_OCC,
          f"{m.L[a]:+.2f} {m.L[b]:+.2f} {m.L[c]:+.2f}")
    check("decided after one: a hit +0.85 > 0.5 -> obstacle; one pass -0.4 -> still unknown (not past -0.5)",
          m.occupied()[c] and m.unknown()[a], f"obstacle {m.occupied()[c]}, a unknown {m.unknown()[a]}")
    m = LogOddsMap()
    m.integrate([], c)                                            # one noisy hit
    m.integrate([c], None)
    m.integrate([c], None)                                        # then passed through twice
    check("noise: hit once, then passed twice -> 0.85 - 0.8 = 0.05 -> no longer an obstacle (unknown)",
          not m.occupied()[c] and abs(m.L[c] - 0.05) < 1e-9, f"L = {m.L[c]:+.2f}")
    m = LogOddsMap()
    for _ in range(20):
        m.integrate([], c)
    check(f"hit again and again -> clamped at +{L_MAX} (no unbounded growth)", m.L[c] == L_MAX, f"L = {m.L[c]}")
    for _ in range(20):
        m.integrate([c], None)
    check(f"passed again and again -> clamped at -{L_MAX}; free", m.L[c] == -L_MAX and m.known_free()[c], f"L = {m.L[c]}")
    print(f"\n{ok}/{n}")
    return n - ok


if __name__ == "__main__":
    raise SystemExit(selftest())
