"""Waypoint navigation: living room <-> hallway <-> entrance, one goto per leg.

    the owner calls "I'm home!" (microphone) -> LLM: greet -> owner at the entrance, duck in the living room
    -> a straight goto would hit the wall or fall into the stairwell
    -> this file: look up the waypoint graph -> [living room -> wall opening -> hallway west side -> around the
       stairwell -> entrance] -> body/goto.py for each leg
The free-map alternative is grid + A* (route.py, explore.py); the waypoints are kept for routing toward a sound.

Where the waypoints come from (obstacles read from the model, not by eye)
    living room <-> hallway: west wall wB at x -1.0, opening at y -1.8 to -1.0
    the middle of the hallway, x -0.4 to 0, y < -0.7, is the stairwell -> walk the west lane, x -0.67
    a console table on the west side at y 0.73-0.87 -> cross to the east side x 0.05 at y 0.45
    a small box on the east side at (-0.3, 0.1) -> 0.35 m clearance when crossing
    "entrance" = the north end of the hallway (the model has no front door; north wall y 2.94, coat rack at (-0.8, 2.8))

    python -m pet.nav.waypoints --selftest
    python -m pet.nav.waypoints            # simulate living room -> entrance -> living room, draw the trace
"""

import math

NODES = {
    "living": (-2.0, -1.5),
    "door_w": (-1.35, -1.4),        # west of the opening (in the living room)
    "door_e": (-0.67, -1.4),        # east of the opening (hallway west lane)
    "hall_s": (-0.67, -0.35),       # next to the north end of the stairwell
    "hall_w": (-0.67, 0.45),        # in front of the console table
    "hall_e": (0.05, 0.50),         # crossed to the east side
    "entry": (0.05, 2.30),          # entrance
}
EDGES = [("living", "door_w"), ("door_w", "door_e"), ("door_e", "hall_s"), ("hall_s", "hall_w"),
         ("hall_w", "hall_e"), ("hall_e", "entry")]
REGIONS = {  # position -> region name (the owner is in the entrance / hallway / living room)
    "living_room": lambda x, y: x < -1.0 and y < 0.5,
    "entrance": lambda x, y: -1.0 <= x <= 0.5 and y >= 1.6,
    "hallway": lambda x, y: -1.0 <= x <= 0.5 and y < 1.6,
}


def region(x, y) -> str:
    return next((k for k, f in REGIONS.items() if f(x, y)), "other")


def _nbrs(n):
    return [b for a, b in EDGES if a == n] + [a for a, b in EDGES if b == n]


def route(a: str, b: str) -> list:
    """Shortest route a -> b on the waypoint graph (breadth-first: few nodes, equal edges is good enough)."""
    prev, frontier = {a: None}, [a]
    while frontier:
        nxt = []
        for n in frontier:
            for m in _nbrs(n):
                if m not in prev:
                    prev[m] = n
                    nxt.append(m)
        frontier = nxt
    if b not in prev:
        raise ValueError(f"{b} is unreachable from {a}")
    path, n = [], b
    while n is not None:
        path.append(n)
        n = prev[n]
    return path[::-1]


def nearest(x, y) -> str:
    """The nearest waypoint in the same region as (x, y) (so it never picks a point on the other side of a wall)."""
    reg = region(x, y)
    cands = [n for n, p in NODES.items() if region(*p) == reg] or list(NODES)
    return min(cands, key=lambda n: math.hypot(NODES[n][0] - x, NODES[n][1] - y))


def plan_path(start_xy, goal_xy) -> list:
    """Start -> nearest waypoint -> ... -> waypoint near the goal -> goal. Returns a list of coordinates."""
    a, b = nearest(*start_xy), nearest(*goal_xy)
    pts = [NODES[n] for n in route(a, b)]
    # the start is already closer to the second waypoint than the first one is -> skip the first (no walking back)
    if len(pts) > 1 and math.dist(start_xy, pts[1]) < math.dist(NODES[a], pts[1]):
        pts = pts[1:]
    return pts + [tuple(goal_xy)]


def navigate(sim, goal_xy, log: list, stop_dist=0.3, **goto_kw) -> dict:
    """Walk to goal_xy along the waypoints: intermediate stops 0.15 m (no precision needed), the last leg stop_dist."""
    from ..body.goto import goto
    x, y, _ = sim.xy_yaw()
    pts = plan_path((x, y), goal_xy)
    log.append("waypoints: " + " -> ".join(f"({p[0]:+.1f},{p[1]:+.1f})" for p in pts))
    r = {}
    for i, p in enumerate(pts):
        last = i == len(pts) - 1
        r = goto(sim, p, stop_dist=stop_dist if last else 0.15, **goto_kw)
        if r.get("fell") or r.get("bumped"):
            log.append(f"leg {i}: {'fell' if r.get('fell') else 'touched ' + str(r['bumped'])} -> stop")
            break
    return r


def selftest() -> int:
    ok = 0
    cases = [(((-2.2, -1.8), (0.0, 2.4)), "living", "entry"),
             (((0.0, 2.4), (-3.0, -2.0)), "entry", "living"),
             (((-0.6, -0.3), (0.0, 2.2)), "hall_s", "entry")]
    for (s, g), a, b in cases:
        pts = plan_path(s, g)
        good = nearest(*s) == a and nearest(*g) == b and pts[-1] == g
        ok += good
        print(f"{'ok ' if good else 'BAD'} {s} -> {g}: nearest waypoints {nearest(*s)} -> {nearest(*g)}, {len(pts)} legs")
    print(f"{ok}/{len(cases)}")
    return len(cases) - ok


def sim_test(out="nav_trace.png"):
    """Duck: living room -> entrance -> living room; counts falls and draws the top-down trace."""
    import mujoco
    from PIL import Image, ImageDraw
    from ..body import sim as S
    from ..body.scheduler import is_fallen
    from ..world import stage as T
    sim = S.DuckSim(T.SCENE, None, spec_edit=lambda sp: T.stage(sp, owner=False), start=(-2.2, -1.8, 0.0))
    for _ in range(50):
        sim.step("walk", S.zeros())
    trace, falls, orig = [], [0], sim.step

    def step(*a):
        orig(*a)
        trace.append(sim.xy_yaw()[:2])
        falls[0] += is_fallen(sim)
    sim.step = step
    for goal in (NODES["entry"], (-2.2, -1.8)):
        log = []
        t0 = sim.t
        navigate(sim, goal, log)
        x, y, _ = sim.xy_yaw()
        print(f"-> {goal}: {sim.t - t0:.0f} s, {math.dist((x, y), goal):.2f} m from the goal, {log[0]}")
    print(f"fallen steps {falls[0]} (0 = no fall)")
    m = sim.model
    m.vis.global_.offwidth = m.vis.global_.offheight = 1000
    r = mujoco.Renderer(m, 1000, 1000)
    cam = mujoco.MjvCamera()
    cam.type, cam.lookat[:], cam.distance, cam.elevation, cam.azimuth = mujoco.mjtCamera.mjCAMERA_FREE, [0, 0, 0], 13, -90, 90
    r.update_scene(sim.data, cam)
    im = Image.fromarray(r.render())
    px = lambda x, y: (495 + 94 * x, 502 - 94 * y)        # pixel mapping of this top-down view (checked against known positions)
    ImageDraw.Draw(im).line([px(*p) for p in trace[::10]], fill=(255, 0, 0), width=4)
    for n, p in NODES.items():
        ImageDraw.Draw(im).ellipse([px(*p)[0] - 6, px(*p)[1] - 6, px(*p)[0] + 6, px(*p)[1] + 6], fill=(0, 90, 255))
    im.save(out)
    print(f"trace: {out}")


if __name__ == "__main__":
    import sys
    sys.exit(selftest()) if "--selftest" in sys.argv else sim_test()
