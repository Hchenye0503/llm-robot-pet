"""Occupancy grid and inflation: which cells of the apartment the duck can walk on.

    1. occupancy grid (build): the room cut into RES-meter cells; cells with fixed furniture, walls, or the stairwell = True
    2. inflation (inflate): obstacles grow by "duck radius + safety margin", so the duck can be treated as a point
       (configuration space)
    3. A* (astar.py) -> 4. simplify into waypoints (route.py)

What goes into the map
    yes  things that do not move: walls, furniture (boxes, cylinders), the charger -- only the part within the duck's
         height band (BAND_LO-BAND_HI); a 1 cm rug can be walked over
    yes  holes going down: stairs (top surface below the floor by HOLE_Z) -> the duck would fall
    no   things that move: the owner (mocap), toys, balls, small cubes -> handled by bump-and-detour in goto
    yes  outside the map (on no floor) = not walkable

Index convention: grid[i, j], i along x (east), j along y (north); cell (i, j) center = (X0 + (i+0.5)·RES, Y0 + (j+0.5)·RES)

    python -m pet.nav.grid           # selftest (inflation, view_blocked)
    python -m pet.nav.grid --draw    # draw the map to grid.png
"""

import math

import numpy as np

RES = 0.05                       # 5 cm cells
X0, X1, Y0, Y1 = -4.5, 1.5, -3.5, 3.5
BAND_LO, BAND_HI = 0.03, 0.30    # the duck's body is 3-30 cm above the floor; anything lower (a rug) can be walked over
HOLE_Z = -0.03                   # top surface more than 3 cm below the floor = a hole (stairwell)
DUCK_R = 0.094                   # duck radius: farthest collision point from the center while standing (the head, forward)
INFLATE_R = 0.15                 # obstacles grow 15 cm: "must still fit, but not too fat" (9.4 cm + walking sway +
#                                  the controller never walks exactly through cell centers)
#   left after inflation: living room opening 60 cm, hallway 135 cm, coffee table <-> sofa 30 cm, table <-> chair 55 cm
#   (at 18 cm the table <-> sofa gap would be only 20 cm)
MOVING = ("owner", "toy", "ball_", "obj_")   # things that move: not in the map


def to_cell(x, y):
    return int((x - X0) // RES), int((y - Y0) // RES)


def to_xy(i, j):
    return X0 + (i + 0.5) * RES, Y0 + (j + 0.5) * RES


def shape():
    return int(round((X1 - X0) / RES)), int(round((Y1 - Y0) / RES))


def build(model=None):
    """MuJoCo scene -> occupancy grid (numpy bool, True = not walkable). No model -> load the apartment with the
    moving owner (world/owner.py)."""
    import mujoco
    if model is None:
        from ..world import owner as O
        spec = mujoco.MjSpec.from_file(str(_scene()))
        O.stage_live(spec)
        model = spec.compile()
    m, d = model, mujoco.MjData(model)
    mujoco.mj_forward(m, d)
    nx, ny = shape()
    xs = X0 + (np.arange(nx) + 0.5) * RES
    ys = Y0 + (np.arange(ny) + 0.5) * RES
    CX, CY = np.meshgrid(xs, ys, indexing="ij")          # center of each cell (CX[i, j], CY[i, j])
    occ = np.zeros((nx, ny), bool)
    floor = np.zeros((nx, ny), bool)
    for g in range(m.ngeom):
        b = m.geom_bodyid[g]
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, _root(m, b)) or ""
        if m.body_jntnum[b] > 0 or m.body_mocapid[b] >= 0 or name.startswith(MOVING) or _root(m, b) != b and _moving_root(m, b):
            continue
        t = m.geom_type[g]
        p, R, s = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3), m.geom_size[g]
        if t == mujoco.mjtGeom.mjGEOM_BOX:
            half_z = abs(R[2, 0]) * s[0] + abs(R[2, 1]) * s[1] + abs(R[2, 2]) * s[2]
            dx, dy = CX - p[0], CY - p[1]                   # cell centers in the box's own frame -> inside its footprint?
            inside = (np.abs(dx * R[0, 0] + dy * R[1, 0]) <= s[0]) & (np.abs(dx * R[0, 1] + dy * R[1, 1]) <= s[1])
        elif t in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_CAPSULE):
            half_z = s[1] if t == mujoco.mjtGeom.mjGEOM_CYLINDER else s[0] + (s[1] if t == mujoco.mjtGeom.mjGEOM_CAPSULE else 0)
            inside = (CX - p[0]) ** 2 + (CY - p[1]) ** 2 <= s[0] ** 2
        else:
            continue
        top, bot = p[2] + half_z, p[2] - half_z
        if abs(top) <= -HOLE_Z and bot <= 0.0:                # top surface near floor level = floor (including a 1 cm rug)
            floor |= inside
        elif top < HOLE_Z:                                    # hole (stairs going down)
            occ |= inside
            floor |= inside
        elif bot < BAND_HI and top > BAND_LO:                 # in the duck's height band
            occ |= inside
    return occ | ~floor                                       # no floor (outside the walls) = not walkable


def _scene():
    from ..config import APARTMENT, MICRODUCK_RL_DIR
    return MICRODUCK_RL_DIR / APARTMENT


def _root(m, b):
    while m.body_parentid[b] > 0:
        b = m.body_parentid[b]
    return b


def _moving_root(m, b):
    while b > 0:
        if m.body_jntnum[b] > 0 or m.body_mocapid[b] >= 0:
            return True
        b = m.body_parentid[b]
    return False


def inflate(occ, radius_m: float, res: float = RES):
    """Grow obstacles by radius_m: every cell within radius_m of an obstacle cell center becomes not walkable.
    Returns a new array (occ is not modified). The radius in cells is rounded up: rounding down grew
    9.4 cm by only one 5 cm cell, and 0.15 / 0.05 = 2.9999... gave 2 cells instead of 3."""
    r_cells = math.ceil(radius_m / res)
    di_dj = [(di, dj) for di in range(-r_cells, r_cells + 1) for dj in range(-r_cells, r_cells + 1) if di ** 2 + dj ** 2 <= r_cells ** 2]
    out = occ.copy()
    for i, j in zip(*np.nonzero(occ)):
        for di, dj in di_dj:
            ni, nj = i + di, j + dj
            if 0 <= ni < out.shape[0] and 0 <= nj < out.shape[1]:
                out[ni, nj] = True
    return out


# ─────────────────────────────────────────────────────────────
# Using the map to help perception. VisionWorld cross-checks two distance estimates: when the floor-plane estimate is
# much farther than the known-size estimate, it *guessed* that furniture hid the feet and trusted the size. But an
# owner seen at an angle, sitting with legs stretched out, has a wide box, so the size estimate was too close (0.9 m)
# while the floor plane (1.8 m, feet visible) was right. With a map there is no need to guess: *check* whether
# furniture lies on the line from the camera to the floor point.
# ─────────────────────────────────────────────────────────────
END_MARGIN = 0.15    # the last 15 cm are not checked: owners often sit next to furniture, the toe cell may touch it


def view_blocked(occ, cam_xy, land_xy, end_margin=END_MARGIN, res=RES):
    """Is there furniture / wall on the line of sight from the camera (cam_xy) to the floor point (land_xy)?
    occ = the *uninflated* grid. Blocked -> True; a point outside the apartment also counts as blocked
    (the contact point must be hidden, like the 5.6 m estimate that landed outside the walls)."""
    L = math.dist(cam_xy, land_xy)
    check_length = L - end_margin
    if check_length <= 0:
        return False
    n = max(2, int(check_length / (res / 2)) + 1)
    nx, ny = occ.shape
    for k in range(n + 1):
        x = cam_xy[0] + (land_xy[0] - cam_xy[0]) * k / n
        y = cam_xy[1] + (land_xy[1] - cam_xy[1]) * k / n
        i, j = to_cell(x, y)
        if not (0 <= i < nx and 0 <= j < ny) or occ[i, j]:
            return True
    return False


def selftest() -> int:
    ok = n = 0

    def check(desc, good, got):
        nonlocal ok, n
        ok, n = ok + bool(good), n + 1
        print(f"{'ok ' if good else 'BAD'} {desc}\n     {got}")

    g = np.zeros((9, 9), bool)
    g[4, 4] = True
    out = inflate(g, 0.10, 0.05)                            # radius 2 cells
    check("one obstacle cell, radius 2 cells -> a 13-cell disk (di^2+dj^2 <= 4)", out.sum() == 13 and g.sum() == 1, f"{out.sum()} cells; original still {g.sum()} (not modified)")
    out = inflate(g, 0.0, 0.05)
    check("radius 0 -> unchanged", out.sum() == 1, f"{out.sum()} cells")
    g = np.zeros((9, 9), bool)
    g[4, 4] = True
    a, b = inflate(g, DUCK_R, 0.05).sum(), inflate(g, 0.15, 0.05).sum()
    check("duck radius 9.4 cm -> 2 cells (13-cell disk); 15 cm -> 3 cells (29-cell disk)", a == 13 and b == 29, f"9.4 cm: {a} cells, 15 cm: {b} cells")
    # view_blocked: real map coordinates (X0, Y0), an empty map with a hand-placed "coffee table"
    occ = np.zeros(shape(), bool)
    ti, tj = to_cell(-2.8, -1.0)
    occ[ti - 6:ti + 6, tj - 4:tj + 4] = True                  # 60 cm x 40 cm table centered at (-2.8, -1.0)
    check("nothing in between -> feet visible (False)", view_blocked(occ, (-1.5, -2.0), (-2.7, -2.0)) is False, "")
    check("line through the table -> blocked (True)", view_blocked(occ, (-2.8, 0.0), (-2.8, -2.0)) is True, "")
    check("toes touching the table (only the last 15 cm reach it) -> not blocked (False)",
          view_blocked(occ, (-2.8, 0.5), (-2.8, -0.70)) is False, "floor point 10 cm outside the table's north edge")
    check("floor point outside the map (like a 5.6 m estimate) -> blocked (True)", view_blocked(occ, (-1.5, -2.0), (-7.0, -2.0)) is True, "")
    check("points closer than end_margin -> False", view_blocked(occ, (-2.8, -0.6), (-2.8, -0.7)) is False, "")
    g = np.zeros((9, 9), bool)
    g[0, 0] = True
    out = inflate(g, 0.10, 0.05)
    check("obstacle in the corner -> grows only inside the map (no wrap-around)", out.sum() == 6 and not out[8, 8], f"{out.sum()} cells")
    g = np.zeros((11, 5), bool)
    g[0:4, 2] = True
    g[7:11, 2] = True                                       # a 3-cell (15 cm) gap between two walls: i = 4, 5, 6
    a = inflate(g, 0.05, 0.05)[4:7, 2].tolist()
    b = inflate(g, 0.10, 0.05)[4:7, 2].tolist()
    check("15 cm gap: radius 5 cm (1 cell) -> the middle cell stays open; 10 cm (2 cells) -> closed",
          a == [True, False, True] and b == [True, True, True], f"1 cell: {a}  2 cells: {b}")
    print(f"\n{ok}/{n}")
    return n - ok


def draw(path="grid.png", radius=None):
    """Draw the occupancy grid (plus the inflated one if radius is given) and the owner's position."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    occ = build()
    fig, ax = plt.subplots(figsize=(9, 10))
    ext = [X0, X1, Y0, Y1]
    img = np.full(occ.T.shape + (3,), 1.0)
    if radius is not None:
        img[inflate(occ, radius).T] = (0.80, 0.86, 0.95)
    img[occ.T] = (0.25, 0.27, 0.30)
    ax.imshow(img, origin="lower", extent=ext, interpolation="nearest")
    ax.plot(-3.25, -2.0, "o", color="#C0392B", ms=12, label="owner (sitting)")
    ax.set_xlim(-4.3, 1.3)
    ax.set_ylim(-3.3, 3.3)
    ax.set_title(f"Occupancy grid ({RES*100:.0f} cm cells; dark = not walkable" + ("; light blue = inflation" if radius is not None else "") + ")")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print("saved", path, f"; not walkable {occ.mean():.0%}")


if __name__ == "__main__":
    import sys
    if "--draw" in sys.argv:
        draw(radius=INFLATE_R)
    else:
        raise SystemExit(selftest())
