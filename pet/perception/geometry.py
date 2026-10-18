"""Geometry: a pixel in the photo -> which direction and how far, relative to the robot.

The VLM / detector only gives a bounding box; the program computes the angle (language models are bad
at arithmetic). The bottom center of the box is where the owner's feet touch the floor.

Assumptions (when they fail the answer is legal but wrong):
    the floor is flat and the target stands on it (stairs or things on a table break this)
    camera height and tilt are known constants (walking shakes the camera, so photos are taken standing still)
The same trick as a table-plane homography, with the table replaced by the floor.

Conventions:
    pixel (u, v): u right, v down, origin top-left; 640x480 image, center (CX, CY)
    robot frame: forward, left, up; bearing positive = left (same as Head.yaw and goto)

    python -m pet.perception.geometry
"""

import math

W, H_PX = 640, 480
CX, CY = W / 2, H_PX / 2
FOVY_DEG = 90.0                                        # MuJoCo fovy is the vertical field of view (assumed for the real lens)
F = (H_PX / 2) / math.tan(math.radians(FOVY_DEG / 2))  # focal length in pixels = 240
CAM_H = 0.24                                           # camera height above the floor (m), measured 0.237
TILT_DEG = 9.6                                         # downward camera tilt: CAD mount 8.5°, measured 8.9-10.1° standing
                                                       # (median 9.6; the walking policy leans slightly forward). A calibrated
                                                       # value, the same job as extrinsic calibration on hardware.


def pixel_to_camera(u: float, v: float) -> tuple[float, float, float]:
    """Pinhole camera: pixel -> a ray (forward, left, up) in the camera frame, with forward = 1.
    A pixel F away from the center is 1 unit off-center on a screen 1 unit in front of the camera.
    u grows to the right but left is positive; v grows downward but up is positive."""
    left_c = (CX - u) / F
    up_c = (CY - v) / F
    return 1.0, left_c, up_c


def tilt(ray_c: tuple[float, float, float]) -> tuple[float, float, float]:
    """Camera tilted down by TILT_DEG: camera frame -> level robot frame.
    Camera forward is (cos, 0, -sin) in the robot frame, camera up is (sin, 0, cos), left is unchanged."""
    f, l, u = ray_c
    t = math.radians(TILT_DEG)
    return f * math.cos(t) + u * math.sin(t), l, -f * math.sin(t) + u * math.cos(t)


def pixel_to_ground(u: float, v: float) -> tuple[float, float | None]:
    """Pixel -> (bearing in degrees, positive = left; horizontal distance on the floor in m).
    A ray that does not point down (at or above the horizon) never hits the floor: distance None."""
    fwd, left, up = tilt(pixel_to_camera(u, v))
    bearing = math.degrees(math.atan2(left, fwd))
    if up >= 0:
        return bearing, None
    t = -CAM_H / up            # how far along the ray until it has dropped CAM_H
    return bearing, t * math.hypot(fwd, left)


def selftest() -> int:
    cases = [  # (u, v) -> (bearing, distance); None = above the horizon
        ((320, 240), (0.0, 1.419)),      # center: along the camera axis, 9.6° down
        ((420, 240), (-22.9, 1.54)),     # 100 px right of center: neither 14° nor 22.6°
        ((320, 400), (0.0, 0.255)),      # low in the image -> close
        ((100, 300), (44.1, 0.764)),     # lower left
        ((320, 0), (0.0, None)),         # top of the image -> above the horizon
    ]
    ok = 0
    for (u, v), (b_exp, d_exp) in cases:
        b, d = pixel_to_ground(u, v)
        good = abs(b - b_exp) < 0.2 and (d is None if d_exp is None else (d is not None and abs(d - d_exp) < 0.005))
        ok += good
        print(f"{'ok ' if good else 'BAD'} ({u:3d},{v:3d}) -> bearing {b:+6.1f}° (want {b_exp:+.1f})"
              f"  distance {d if d is None else round(d, 3)} (want {d_exp})")
    print(f"{ok}/{len(cases)}")
    return len(cases) - ok


if __name__ == "__main__":
    raise SystemExit(selftest())
