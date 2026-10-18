"""Navigation: finding targets and getting there.

    search      turn and look until the target is found (or honestly not found)
    grid        occupancy grid of the apartment + inflation; line-of-sight check for perception
    astar       A* on the grid
    route       A* path -> waypoints -> goto; used when furniture blocks the straight line
    mapping     log-odds mapping from the ToF sensor (ToF, cliff and bump layers)
    explore     from an empty map to a goal, replanning only when the path is blocked
    waypoints   hand-placed waypoint graph between rooms (used to walk toward a sound)
    servo       visual servoing on the detector box (kept for comparison; not the default)
"""
