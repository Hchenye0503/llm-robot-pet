"""Watch "the owner comes home -> greeting" live in a MuJoCo window (the showcase's segment 4, without recording).

Mouse: left drag = rotate, right drag = pan, wheel = zoom; [ and ] switch model cameras (head_camera = what the duck
sees); Esc = free camera; Tab = side panel. The window pauses for the 1-3 s the LLM plans (a synchronous call);
the VLM runs in the background and does not pause it.

    python -m demos.live        # a = owner comes home; b = owner holds up a toy (needs OPENAI_API_KEY)
"""

import sys
import time

import mujoco
import mujoco.viewer

from pet.body import sim as S

from . import showcase as SC


class LiveSim(S.DuckSim):
    """No recording; syncs the window every 20 ms control step and slows down to wall-clock time."""

    def __init__(self, scene, video=None, **kw):
        kw.pop("overlay", None)
        super().__init__(scene, None, **kw)                              # the showcase passes a video path: dropped
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING      # the camera follows the duck
        self.viewer.cam.trackbodyid = self.model.body(self.model.jnt_bodyid[0]).id
        self.viewer.cam.distance, self.viewer.cam.elevation, self.viewer.cam.azimuth = 2.5, -30, 135
        self._wall = None

    def step(self, policy_name, cmd):
        super().step(policy_name, cmd)
        if not self.viewer.is_running():
            raise SystemExit("window closed")
        self.viewer.sync()
        now = time.perf_counter()
        if self._wall is None or now - self._wall > 0.5:                # after an LLM pause, do not fast-forward to catch up
            self._wall = now
        self._wall += S.CONTROL_DT
        if self._wall > now:
            time.sleep(self._wall - now)

    def close(self):
        print("Done; the window shows the last frame. Close it when finished.", flush=True)
        while self.viewer.is_running():
            time.sleep(0.1)


if __name__ == "__main__":
    shot = sys.argv[1] if len(sys.argv) > 1 else "a"
    S.DuckSim = LiveSim
    SC.finish = lambda sim, vw, falls: (print(f"{len(falls)} falls; {sim.t:.0f} s"), sim.close(), vw.r.close())
    SC.seg4(video=True, shot=shot)
