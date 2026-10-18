"""A Microduck simulator driven by code instead of a keyboard (the layer the scheduler calls).

    planner -> [(policy, 13 numbers), ...]
      -> scheduler: every 20 ms decides which policy runs with which 13 numbers
      -> DuckSim.step(policy, cmd): observation -> ONNX inference -> actuator model -> 4 physics steps

The upstream repo is not modified. Its infer_policy.py (physics, BAM actuator model, ONNX, observations,
the PolicyInference class) is loaded at runtime; only its keyboard loop is skipped. Each step sets
    policy.ort_session = which ONNX
    policy.command[:]  = which 13 numbers
and infer() reads only those two.

Policies (paths in config.py):
    walk      = the pet gait trained in this project (walks at 0.05 m/s, turns in place, tracks 0.37 at 0.4)
    recover   = official velstand.onnx (gets up after a fall)
    spin      = spin hop trained in this project (simulation only: never validated on hardware)
    sitstand  = official alpha_sitstand.onnx
    stand     = official alpha_stand.onnx (body pitch command: leans back 15 deg + head up 60 deg -> camera ~68 deg up)
    kick_left / kick_right = official ball_kick_*.onnx

Delay 1 step (aligned with training).

    python -m pet.body.sim                    # stand 1 s -> walk 3 s -> sit 4 s -> stand 4 s -> walk 2 s
    python -m pet.body.sim --video smoke.mp4
"""

import argparse
import importlib.util
import os
import sys
import types

import mujoco
import numpy as np
import onnxruntime as ort

from .. import config

ONNX = {
    "walk": config.PET_GAIT_ONNX,
    "recover": config.POLICY_DIR / "velstand.onnx",
    "spin": config.SPIN_HOP_ONNX,
    "sitstand": config.POLICY_DIR / "alpha_sitstand.onnx",
    "stand": config.POLICY_DIR / "alpha_stand.onnx",
    "kick_left": config.POLICY_DIR / "ball_kick_left.onnx",
    "kick_right": config.POLICY_DIR / "ball_kick_right.onnx",
}
DEFAULT_SCENE = f"{config.SCENES}/scene_ball.xml"
DECIMATION = 4          # physics 5 ms x 4 = control 20 ms (50 Hz)
CONTROL_DT = 0.02

# upstream infer_policy.py imports termios, which does not exist on Windows -> insert empty modules first
for _name in ("termios", "tty"):
    sys.modules.setdefault(_name, types.ModuleType(_name))
_spec = importlib.util.spec_from_file_location("infer_policy", config.MICRODUCK_RL_DIR / "scripts" / "infer_policy.py")
ip = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ip)


class DuckSim:
    """One duck driven by code. step() = 20 ms."""

    def __init__(self, scene: str = DEFAULT_SCENE, video: str | None = None, size=(480, 360),
                 cam=(0.9, -20, 135), overlay=None, spec_edit=None, start=(0.0, 0.0, 0.0)):
        """spec_edit: f(MjSpec) that edits the scene before compiling (e.g. adds the owner);
        start: (x, y, heading in degrees)."""
        os.chdir(config.MICRODUCK_RL_DIR)  # upstream XML paths are relative to the repo
        self.start = start
        bam_model = ip.load_bam_model(ip.BAM_KP_FW, 7.4, 0.0)
        if spec_edit is None:
            self.model, self.data, self.bam, _ = ip.load_mujoco_with_bam(scene, bam_model, 0.005, 0.1, ip.BAM_VIN_MIN)
        else:
            # upstream load_mujoco_with_bam reads and compiles the file with no hook in between ->
            # temporarily replace the MjSpec.from_file it uses with "read + spec_edit" (only during this load)
            orig = ip.mujoco.MjSpec

            class _Spec:
                @staticmethod
                def from_file(path):
                    spec = orig.from_file(path)
                    spec_edit(spec)
                    return spec
            ip.mujoco.MjSpec = _Spec
            try:
                self.model, self.data, self.bam, _ = ip.load_mujoco_with_bam(scene, bam_model, 0.005, 0.1, ip.BAM_VIN_MIN)
            finally:
                ip.mujoco.MjSpec = orig
        self.p = ip.PolicyInference(
            self.model, self.data, bam_ctrl=self.bam,
            walking_onnx_path=str(ONNX["walk"]),
            sitstand_onnx_path=str(ONNX["sitstand"]),
            kick_left_onnx_path=str(ONNX["kick_left"]),
            kick_right_onnx_path=str(ONNX["kick_right"]),
            new_cmd_obs=True, delay_min_lag=1, delay_max_lag=1,
            # Must be True: the class default is False (raw accelerometer); only upstream main() switches it on.
            # The official ONNX metadata says observation_names = projected_gravity. Forgetting it raised no error
            # and the duck still walked, but leaned to cos 0.33 and slid 10 cm while standing (a silent failure).
            use_projected_gravity=True,
        )
        self.sessions = {
            "walk": self.p.walking_session,
            "sitstand": self.p.sit_session,
            "kick_left": self.p.behavior_sessions["kick_left"],
            "kick_right": self.p.behavior_sessions["kick_right"],
            "recover": ort.InferenceSession(str(ONNX["recover"])),
            "spin": ort.InferenceSession(str(ONNX["spin"])),
            "stand": ort.InferenceSession(str(ONNX["stand"])),
        }
        self._fj = self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")]
        self.t = 0.0
        self.policy_name = None
        self.n_switch = 0
        self._writer = None
        if video:
            import imageio
            w, h = size
            self.model.vis.global_.offwidth = max(self.model.vis.global_.offwidth, w)   # offscreen buffer must fit
            self.model.vis.global_.offheight = max(self.model.vis.global_.offheight, h)
            self._renderer = mujoco.Renderer(self.model, h, w)
            self.overlay = overlay   # f(image, sim) -> image (e.g. captions)
            # cam: one camera or a list of cameras -> each frame puts the views side by side (split screen)
            self._cams = [self._make_cam(c) for c in (cam if isinstance(cam, list) else [cam])]
            self._writer = imageio.get_writer(video, fps=int(1 / CONTROL_DT))
        self.reset()

    def reset(self):
        """Same start pose as upstream main(): HOME, 0.125 m above the floor.
        Velocities, the last action and the delay buffer are cleared too, so episodes are independent."""
        d, q = self.data, self._fj
        mujoco.mj_resetData(self.model, d)          # velocities, accelerations, time -> 0
        self.p.last_action[:] = 0.0
        if self.p.use_delay:
            self.p.action_buffer = [np.zeros_like(a) for a in self.p.action_buffer]
            self.p.buffer_index = 0
        x0, y0, yaw0 = self.start
        half = np.radians(yaw0) / 2
        d.qpos[q:q + 3] = [x0, y0, 0.125]
        d.qpos[q + 3:q + 7] = [np.cos(half), 0, 0, np.sin(half)]   # rotate yaw0 about z
        for i, idx in enumerate(self.p.joint_qpos_indices):
            d.qpos[idx] = self.p.default_pose[i]
        self.bam.reset(d.qpos)
        self.p.set_position_targets(self.p.default_pose)
        mujoco.mj_forward(self.model, d)
        self.t = 0.0

    def step(self, policy_name: str, cmd) -> None:
        """Run 20 ms with ONNX `policy_name` and command `cmd` (13 numbers)."""
        if policy_name != self.policy_name:
            self.n_switch += 1
            self.policy_name = policy_name
        self.p.ort_session = self.sessions[policy_name]
        self.p.command[:] = np.asarray(cmd, dtype=np.float32)
        self.p.apply_action(self.p.infer())
        for _ in range(DECIMATION):
            self.bam.update()
            mujoco.mj_step(self.model, self.data)
        self.t += CONTROL_DT
        if self._writer is not None:
            panels = []
            for cam, spec in self._cams:
                if isinstance(spec, dict) and isinstance(spec.get("lookat"), str):
                    self._aim(cam, spec["pos"], self._trunk_pos() + np.array([0, 0, 0.05]))   # fixed camera following the duck
                self._renderer.update_scene(self.data, cam)
                panels.append(self._renderer.render())
            img = np.concatenate(panels, axis=1)
            self._writer.append_data(self.overlay(img, self) if self.overlay else img)

    def _trunk_pos(self):
        return np.array(self.data.qpos[self._fj:self._fj + 3], float)

    @staticmethod
    def _aim(cam, pos, look):
        pos, look = np.asarray(pos, float), np.asarray(look, float)
        v = look - pos
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = look
        cam.distance = float(np.linalg.norm(v))
        cam.azimuth = float(np.degrees(np.arctan2(v[1], v[0])))
        cam.elevation = float(np.degrees(np.arcsin(v[2] / np.linalg.norm(v))))

    def _make_cam(self, spec):
        """spec: (distance, elevation, azimuth) = tracking; a camera name in the model (e.g. "head_camera");
        {"pos", "lookat": [x, y, z]} = fixed; {"pos", "lookat": "duck"} = fixed position, aimed at the duck."""
        cam = mujoco.MjvCamera()
        if isinstance(spec, str):
            cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            cam.fixedcamid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, spec)
        elif isinstance(spec, dict):
            look = self._trunk_pos() if isinstance(spec["lookat"], str) else spec["lookat"]
            self._aim(cam, spec["pos"], look)
        else:
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            cam.trackbodyid = self.model.jnt_bodyid[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")]
            cam.distance, cam.elevation, cam.azimuth = spec
        return cam, spec

    # ── sensing: measured every step, not tracked by bookkeeping ──
    def trunk_height(self) -> float:
        """Trunk height above the floor (m). Ground truth in simulation; joint angles + forward kinematics on hardware."""
        return float(self.data.qpos[self._fj + 2])

    def upright(self) -> float:
        """cos of the angle between the trunk's up and the world's up: 1 = upright, 0 = on its side, < 0 = flipped."""
        w, x, y, z = self.data.qpos[self._fj + 3:self._fj + 7]
        return float(1 - 2 * (x * x + y * y))

    def xy_yaw(self) -> tuple[float, float, float]:
        w, x, y, z = self.data.qpos[self._fj + 3:self._fj + 7]
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return float(self.data.qpos[self._fj]), float(self.data.qpos[self._fj + 1]), float(np.degrees(yaw))

    def touching_what(self):
        """What the robot touches right now besides the floor: a body name (geom name for world furniture), or None."""
        m, d = self.model, self.data
        root = m.body_rootid[m.jnt_bodyid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")]]
        for i in range(d.ncon):
            g1, g2 = d.contact[i].geom1, d.contact[i].geom2
            r1, r2 = m.body_rootid[m.geom_bodyid[g1]], m.body_rootid[m.geom_bodyid[g2]]
            if (r1 == root) == (r2 == root):
                continue
            g = g2 if r1 == root else g1
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE or (m.geom(g).name or "").startswith(("floor", "stair")):
                continue
            b = m.geom_bodyid[g]
            return m.body(b).name if b != 0 else (m.geom(g).name or f"geom{g}")
        return None

    def touching(self, body: str) -> bool:
        """Does any part of the robot touch `body` (e.g. "owner") right now?
        Contact list in simulation; a bumper switch or a motor-current spike on hardware."""
        m, d = self.model, self.data
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
        if b < 0:
            return False
        root = m.body_rootid[m.jnt_bodyid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")]]
        for i in range(d.ncon):
            b1, b2 = m.geom_bodyid[d.contact[i].geom1], m.geom_bodyid[d.contact[i].geom2]
            if (b1 == b and m.body_rootid[b2] == root) or (b2 == b and m.body_rootid[b1] == root):
                return True
        return False

    def place_ball(self, fx: float, fy: float, body: str = "ball"):
        """Put the ball at (fx, fy) in the robot frame (forward, left). body: "ball" in scene_ball, "ball_0" in the apartment.
        Height comes from the body's z in the XML (scene_ball: z = radius; apartment: z = 0, the geom is offset up);
        hard-coding z = radius left the apartment ball floating 4 cm up."""
        b = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body)
        j = self.model.body_jntadr[b]
        qa, va = self.model.jnt_qposadr[j], self.model.jnt_dofadr[j]
        x, y, yaw = self.xy_yaw()
        c, s = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
        self.data.qpos[qa:qa + 7] = [x + c * fx - s * fy, y + s * fx + c * fy, self.model.body_pos[b][2], 1, 0, 0, 0]
        self.data.qvel[va:va + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def close(self):
        if self._writer is not None:
            self._writer.close()


# ─────────────────────────────────────────────────────────────
# smoke test: a fixed schedule to check that switching policies works on this machine
# ─────────────────────────────────────────────────────────────
def zeros():
    return [0.0] * 13


def walk(vx):
    c = zeros(); c[0] = vx; return c


def flag(sit):
    c = zeros(); c[0] = 1.0 if sit else 0.0; return c


SMOKE = [
    # (description, policy, cmd, seconds)
    ("stand (walk, zero cmd)", "walk", zeros(), 1.0),
    ("walk forward 0.2", "walk", walk(0.2), 3.0),
    ("sit (sitstand flag 1)", "sitstand", flag(True), 4.0),
    ("stand up (sitstand flag 0)", "sitstand", flag(False), 4.0),
    ("walk 0.2 again", "walk", walk(0.2), 2.0),
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=DEFAULT_SCENE)
    ap.add_argument("--video", default=None)
    args = ap.parse_args()

    sim = DuckSim(args.scene, args.video)
    print(f"\nscene = {args.scene}\n")
    print(f"{'segment':<28}{'end height m':>13}{'min height m':>13}{'min upright':>12}{'dx m':>8}")
    for desc, pol, cmd, secs in SMOKE:
        x0, _, _ = sim.xy_yaw()
        h_min, up_min = 9.9, 9.9
        for _ in range(int(round(secs / CONTROL_DT))):
            sim.step(pol, cmd)
            h_min, up_min = min(h_min, sim.trunk_height()), min(up_min, sim.upright())
        x1, _, _ = sim.xy_yaw()
        print(f"{desc:<28}{sim.trunk_height():>13.3f}{h_min:>13.3f}{up_min:>12.2f}{x1 - x0:>8.3f}")
    sim.close()
    print("\nreference: sitstand training targets STAND_Z = 0.115, SIT_Z = 0.060")
