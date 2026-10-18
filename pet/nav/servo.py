"""Visual servoing: once the VLM has confirmed the target, keep tracking the same detector box while walking.

Coarse to fine has a pause: the far estimate is imprecise, so the duck walks to 0.9 m, stops, takes a photo and asks
the VLM again (~2 s), even when the owner has not moved. The target is already confirmed, so it only needs to
follow the same box: the detector takes 0.3 s, costs nothing, and needs no stop.

    look_for finds the target and the VLM confirms -> servo(...): every SERVO_DT s
        photo while walking -> detector -> the box most like the predicted one (never jump to another object)
        -> box center off by how many degrees, box width as a fraction of the image -> servo_command -> walk command
        -> box big enough (arrived) / lost / bumped -> stop
The error comes straight from the image, with no world coordinates and no flat-floor assumption
(goto_command needs the target in the robot frame; servo_command needs only where the box is).

Not the default: with the pet gait the camera bobs 13.5-31.6 deg while walking, and tracking lost the target
(3 photos in a row). VisionWorld(servo=True) turns it on for comparison.

    python -m pet.nav.servo      # selftest, no simulator
"""

import math

V_FWD = 0.3            # forward speed while tracking (m/s)
K_TURN = 1.5           # proportional gain: 10 deg off -> turn 15 deg/s
W_MAX = 57.0           # turn rate limit (deg/s; schema limit)
W_MIN_INPLACE = 43.0   # dead zone of turning in place: below 43 deg/s it does not move
TURN_ONLY = 20.0       # more than this off -> turn in place first, no walking
STOP_W = {"owner": 0.40, "toy": 0.18}     # box width fraction that counts as arrived (owner ~0.4 m, toy ~0.35 m)
SERVO_DT = 0.5         # look every SERVO_DT s
LOST_MAX = 3           # this many misses in a row = lost


def servo_command(bearing_deg: float, width_frac: float, stop_w: float):
    """Box center off by bearing_deg (positive = left), box width width_frac of the image -> (vx m/s, wz deg/s, done).
      1. width_frac >= stop_w       -> (0, 0, True) arrived
      2. |bearing| > TURN_ONLY      -> turn in place: wz has the sign of bearing, magnitude
                                       max(W_MIN_INPLACE, min(W_MAX, K_TURN x |bearing|)) (the dead zone needs a minimum)
      3. otherwise (slightly off)   -> walk and correct: (V_FWD, K_TURN x bearing clamped to +-W_MAX)"""
    if width_frac >= stop_w:
        return 0.0, 0.0, True
    elif abs(bearing_deg) > TURN_ONLY:
        wz = max(W_MIN_INPLACE, min(W_MAX, K_TURN * abs(bearing_deg)))
        return 0.0, math.copysign(wz, bearing_deg), False
    else:
        wz = max(-W_MAX, min(W_MAX, K_TURN * bearing_deg))
        return V_FWD, wz, False


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter, 1)


def servo(sim, vw, target: str, box0, log: list, timeout_s: float = 20.0) -> dict:
    """Start from the confirmed box box0 and track target while walking. Returns {"reached", "lost", "bumped", "secs"}."""
    from PIL import Image
    from ..body import sim as S
    from ..perception import geometry as G
    from ..perception.vision_world import MIN_SCORE, shoot
    prev, lost, t0, stop_w = list(box0), 0, sim.t, STOP_W[target]
    pred = list(prev)                                 # where the box should be in the next photo
    c = S.zeros()
    while sim.t - t0 < timeout_s:
        arr, _ = shoot(sim, vw.r)                     # photo while walking (no stop)
        vw.last_img = arr
        found = vw.det(Image.fromarray(arr))[target]
        # Track the *same* object: the box must overlap the predicted position (IoU > 0.1), or it may jump to something
        # else (a green book). Comparing with the previous box failed: the duck turned 15 deg in 0.5 s, the box moved
        # 86 px, overlapped by only 4 px, and the target was declared lost although it was seen every time.
        if found is None or found[1] < MIN_SCORE[target] or _iou(found[0], pred) < 0.1:
            lost += 1
            vw.last.update(box=None, status="none", verdict=f"tracking: lost {lost}")
            if lost >= LOST_MAX:
                log.append(f"  servo: missed {lost} times in a row -> lost")
                break
            for _ in range(int(SERVO_DT / S.CONTROL_DT)):
                sim.step("walk", S.zeros())
            continue
        lost, prev = 0, list(found[0])
        box = prev
        pred = list(prev)
        bearing = G.pixel_to_ground((box[0] + box[2]) / 2, G.CY)[0]
        vx, wz, done = servo_command(bearing, (box[2] - box[0]) / G.W, stop_w)
        vw.last.update(box=box, status="ok", verdict=f"tracking: {bearing:+.0f} deg, width {(box[2] - box[0]) / G.W:.0%}")
        if done:
            log.append(f"  servo: box width {(box[2] - box[0]) / G.W:.0%} -> arrived ({sim.t - t0:.1f} s, no stop to ask the VLM)")
            for _ in range(15):
                sim.step("walk", S.zeros())
            return {"reached": True, "lost": False, "bumped": None, "secs": sim.t - t0, "box": prev}
        c[0], c[2] = vx, math.radians(wz)
        # ego-motion compensation (prediction): turning left wz x SERVO_DT deg moves things in the image right by F x tan(angle) px
        du = G.F * math.tan(math.radians(wz * SERVO_DT))
        pred = [pred[0] + du, pred[1], pred[2] + du, pred[3]]
        for _ in range(int(SERVO_DT / S.CONTROL_DT)):
            sim.step("walk", c)
            hit = sim.touching_what()
            if hit:
                log.append(f"  servo: touched {hit} -> stop")
                for _ in range(15):
                    sim.step("walk", S.zeros())
                return {"reached": hit == target, "lost": False, "bumped": hit, "secs": sim.t - t0, "box": prev}
    for _ in range(15):
        sim.step("walk", S.zeros())
    return {"reached": False, "lost": lost >= LOST_MAX, "bumped": None, "secs": sim.t - t0, "box": prev}


def selftest() -> int:
    cases = [  # (bearing, width, stop) -> (vx, wz, done)
        ((0.0, 0.45, 0.40), (0.0, 0.0, True)),        # box big enough -> arrived (however far off)
        ((30.0, 0.45, 0.40), (0.0, 0.0, True)),
        ((30.0, 0.10, 0.40), (0.0, 45.0, False)),     # 30 deg left -> turn left in place 1.5 x 30 = 45
        ((-25.0, 0.10, 0.40), (0.0, -43.0, False)),   # 25 deg right -> 37.5 < dead zone 43 -> -43
        ((-60.0, 0.10, 0.40), (0.0, -57.0, False)),   # 60 deg right -> 90 over the limit -> -57
        ((10.0, 0.10, 0.40), (0.3, 15.0, False)),     # 10 deg left -> walk, correct left 15
        ((-4.0, 0.20, 0.40), (0.3, -6.0, False)),     # 4 deg right -> walk, correct right 6 (no dead zone while walking)
        ((0.0, 0.10, 0.40), (0.3, 0.0, False)),       # straight ahead -> walk
        ((20.0, 0.10, 0.40), (0.3, 30.0, False)),     # exactly 20 deg still counts as slightly off (> turns in place)
    ]
    ok = 0
    for args, want in cases:
        got = servo_command(*args)
        good = all(abs(g - w) < 1e-6 for g, w in zip(got[:2], want[:2])) and got[2] == want[2]
        ok += good
        print(f"{'ok ' if good else 'BAD'} off {args[0]:+5.1f} deg, width {args[1]:.0%} -> {tuple(round(x, 2) if isinstance(x, float) else x for x in got)}"
              + ("" if good else f" (want {want})"))
    print(f"{ok}/{len(cases)}")
    return len(cases) - ok


if __name__ == "__main__":
    raise SystemExit(selftest())
