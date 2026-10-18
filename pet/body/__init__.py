"""Body: run plans on the simulated Microduck.

    sim         DuckSim: the upstream simulator driven by code (policy + 13 command numbers every 20 ms)
    scheduler   plan steps -> policy commands; fall detection and recovery; idle watchdog
    keyframes   head gestures as keyframes, interpolated every 20 ms
    behaviors   ethogram behaviors expanded into motion (walk to the owner, face, look up, gestures)
    goto        closed-loop walk-to-target controller with hysteresis
    idle        sit <-> stand with hysteresis when nothing happens
"""
