"""MDP terms for the pet gait: command generator, velocity-tracking rewards, curriculum logging.

Built on the Microduck velocity task (microduck_rl, Apache-2.0); the upstream repo is imported, not modified.

Design
  1. Track the *average* speed over the last 0.48 s (one gait cycle) with sigma = k x (0.02 + 10% x command speed);
     k starts large and shrinks one step each time the tracking score reaches 80%, down to 1.
  2. 15% of the moving robots practice walking slowly straight; the rest draw forward speeds skewed toward slow.
  3. Commands are resampled every 1.5-3 s (env_cfg.py).
Tests: test_petgait_cfg.py.
"""

from __future__ import annotations

from dataclasses import dataclass as _dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.managers.manager_base import ManagerTermBase
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_microduck.tasks.mdp import VelocityCommandCommandOnly, VelocityCommandCommandOnlyCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def sigma_of(cmd_speed: torch.Tensor, k: float) -> torch.Tensor:
    """sigma = k x (0.02 + 10% x command speed), per robot. cmd_speed: (N,), >= 0. With k = 1 this is exactly the
    pass criterion's tolerance; the 0.02 floor keeps it from dividing by zero at a zero command."""
    return k * (0.02 + 0.1 * cmd_speed)  # (N,)


def next_k(k: float, window_mean: float, threshold: float = 0.8,
           shrink: float = 0.8, k_min: float = 1.0) -> float:
    """The next k from the tracking score averaged over the last window (~50 iterations, 0-1):
    score >= threshold -> shrink by `shrink`, never below k_min; otherwise unchanged."""
    if window_mean >= threshold:
        k = max(k * shrink, k_min)
    return k


def still_gate(cmd_size: torch.Tensor, avg_speed: torch.Tensor,
               cmd_zero: float = 0.01, stopped: float = 0.05) -> torch.Tensor:
    """Which robots should be standing still right now? (N,) booleans.
    The command is zero *and* the 0.48 s average speed is already below `stopped` (the pass criterion's definition of
    stopped). Robots still braking are not penalized: those one or two steps cannot be avoided."""
    return (cmd_size < cmd_zero) & (avg_speed < stopped)


def yaw_sigma_of(cmd_wz: torch.Tensor, k: float, base: float = 0.05) -> torch.Tensor:
    """Turning tolerance sigma = k x (base + 10% x |turn command|), the same idea as sigma_of for yaw.
    cmd_wz is signed (left positive); left and right turns of the same size get the same tolerance.
    base: allowed average yaw rate when walking straight (0.05 rad/s ~ 3 deg/s; a 0.48 s average already wobbles
    +-3-6 deg/s, so 1 deg/s would penalize robots that do walk straight)."""
    return k * (base + 0.1 * torch.abs(cmd_wz))  # (N,)


def turn_sigma_floor(sigma: torch.Tensor, cmd_wz: torch.Tensor, k: float,
                     floor: float = 0.04, wz_on: float = 0.01) -> torch.Tensor:
    """With a turn command, sigma is at least k x floor; without one it is unchanged (v10).

    Why: while stepping, the 0.48 s average linear speed is still off by ~0.017 m/s regardless of turn rate. Walking,
    the 10% x speed margin covers it; turning in place (or very slowly) sigma is only k x 0.02, so every step cost
    ~1.2 and in-place turning collapsed on 2 of 3 seeds (v9). floor 0.04 = the width at 0.2 m/s. A max (the same trick
    as k_min in next_k): at >= 0.2 m/s sigma is already >= 0.04 and nothing changes."""
    turning = torch.abs(cmd_wz) > wz_on
    return torch.where(turning, torch.clamp(sigma, min=k * floor), sigma)


class PetGaitLinVelTracking(ManagerTermBase):
    """exp(-(command - 0.48 s average velocity)^2 / sigma^2), sigma = sigma_of(command speed, k).

    Why the average: the body speed swings +-0.05 m/s within every step (4.1 Hz foot strikes + a 2 Hz left / right
    asymmetry). With a tight sigma on instantaneous speed every step is penalized for an unavoidable wobble, and the
    policy learns to stand still. Averaged over one gait cycle the wobble cancels; only "too slow overall" is penalized.

    The k curriculum lives in this class: every check_every_steps steps, next_k() is called with the mean score."""

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        p = cfg.params
        self.window = max(1, round(p.get("window_s", 0.48) / env.step_dt))   # 0.48 s / 0.02 s = 24 steps
        self.buf = torch.zeros(env.num_envs, self.window, 3, device=env.device)
        self.pos = 0
        self.k = float(p.get("k_init", 8.0))
        self.last_window_mean = 0.0
        self._sum = torch.zeros((), device=env.device)
        self._n = 0
        self.v_avg = torch.zeros(env.num_envs, 3, device=env.device)   # latest average velocity (read by standing_foot_lift)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        # new episode: clear these robots' velocity history (or they would see the previous episode's speed)
        if env_ids is None:
            env_ids = slice(None)
        self.buf[env_ids] = 0.0

    def __call__(self, env: ManagerBasedRlEnv, command_name: str, k_init: float = 8.0,
                 window_s: float = 0.48, check_every_steps: int = 1200,
                 threshold: float = 0.8, shrink: float = 0.8, k_min: float = 1.0,
                 turn_floor: float | None = None,
                 asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
        del k_init, window_s   # used only in __init__
        asset = env.scene[asset_cfg.name]
        cmd = env.command_manager.get_command(command_name)            # (N, 3): vx, vy, wz

        # ring buffer of body-frame velocities; average over the last `window` steps
        v = torch.nan_to_num(asset.data.root_link_lin_vel_b, nan=0.0)   # (N, 3)
        self.buf[:, self.pos] = v
        self.pos = (self.pos + 1) % self.window
        v_avg = self.buf.mean(dim=1)
        self.v_avg = v_avg

        # error: forward + sideways against the command, vertical speed should be 0 (as upstream, but averaged)
        err2 = torch.sum(torch.square(cmd[:, :2] - v_avg[:, :2]), dim=1) + torch.square(v_avg[:, 2])
        sigma = sigma_of(torch.linalg.norm(cmd[:, :2], dim=1), self.k)
        if turn_floor is not None:                                      # v10; None -> identical to v9
            sigma = turn_sigma_floor(sigma, cmd[:, 2], self.k, floor=turn_floor)
        r = torch.exp(-err2 / torch.square(sigma))

        # k curriculum: accumulate the score, check every check_every_steps steps
        self._sum += r.mean()
        self._n += 1
        if self._n >= check_every_steps:
            self.last_window_mean = float(self._sum / self._n)
            self.k = next_k(self.k, self.last_window_mean, threshold, shrink, k_min)
            self._sum.zero_()
            self._n = 0
        return r


class PetGaitYawTracking(ManagerTermBase):
    """v5: exp(-(turn command - 0.48 s average yaw rate)^2 / sigma^2), sigma = yaw_sigma_of(turn command, k), sharing k.

    Why: upstream track_angular_velocity uses the *instantaneous* yaw rate with a fixed sigma of 0.707 rad/s (40 deg/s).
    The body twists +-60 deg/s every step, so drifting 5 deg/s hides inside it and costs only 1.8% -> each seed drifted
    its own way. Averaged over a gait cycle the twisting cancels; what is left is the drift.
    Why an extra term instead of replacing upstream's: that term also governs the instantaneous roll / pitch rates,
    which average to zero and would no longer be controlled. Upstream's term stays; this one only checks the drift."""

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        self.window = max(1, round(cfg.params.get("window_s", 0.48) / env.step_dt))
        self.buf = torch.zeros(env.num_envs, self.window, device=env.device)
        self.pos = 0

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.buf[env_ids] = 0.0

    def __call__(self, env: ManagerBasedRlEnv, command_name: str, base: float = 0.05,
                 window_s: float = 0.48, tracking_reward_name: str = "track_linear_velocity",
                 asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
        del window_s   # used only in __init__
        asset = env.scene[asset_cfg.name]
        cmd_wz = env.command_manager.get_command(command_name)[:, 2]                 # (N,)

        wz = torch.nan_to_num(asset.data.root_link_ang_vel_b[:, 2], nan=0.0)        # yaw rate this step
        self.buf[:, self.pos] = wz
        self.pos = (self.pos + 1) % self.window
        wz_avg = self.buf.mean(dim=1)

        k = env.reward_manager.get_term_cfg(tracking_reward_name).func.k             # k shared with linear tracking
        sigma = yaw_sigma_of(cmd_wz, k, base)
        return torch.exp(-torch.square(cmd_wz - wz_avg) / torch.square(sigma))


def standing_foot_lift(env: ManagerBasedRlEnv, command_name: str, sensor_name: str,
                       tracking_reward_name: str = "track_linear_velocity",
                       cmd_zero: float = 0.01, stopped: float = 0.05) -> torch.Tensor:
    """When the robot should stand (still_gate), how many feet are in the air (0, 1 or 2). Negative weight -> penalty.
    Feet in the air, not action changes: nodding or tilting the head while standing moves neck motors but not feet,
    and must not be penalized."""
    cmd = env.command_manager.get_command(command_name)                          # (N, 3): vx, vy, wz
    cmd_size = torch.linalg.norm(cmd[:, :2], dim=1) + torch.abs(cmd[:, 2])       # turning in place counts as a command
    v_avg = env.reward_manager.get_term_cfg(tracking_reward_name).func.v_avg     # the tracking term's 0.48 s average
    avg_speed = torch.linalg.norm(v_avg[:, :2], dim=1)
    found = env.scene[sensor_name].data.found                                    # (N, 2): left / right foot touching the ground
    feet_in_air = (found <= 0).float().sum(dim=1)
    return feet_in_air * still_gate(cmd_size, avg_speed, cmd_zero, stopped).float()


def petgait_k_log(env: ManagerBasedRlEnv, env_ids, reward_name: str = "track_linear_velocity") -> dict:
    """Changes nothing; logs k and the last window's mean score (TensorBoard Curriculum/petgait_k/...)."""
    del env_ids
    term = env.reward_manager.get_term_cfg(reward_name).func
    return {"k": torch.tensor(term.k), "window_mean": torch.tensor(term.last_window_mean)}


class PetGaitVelocityCommand(VelocityCommandCommandOnly):
    """After upstream's sampling (uniform + standing + turning in place), two changes for the *moving* robots:

      1. skewed toward slow: |vx| = limit x u^p (u uniform 0-1). With p = 2, P(|vx| < 0.1) goes from 25% to 50%
      2. straight buckets: a fraction walks slowly straight (vx in slow_straight_range, vy = wz = 0), another fraction
         walks fast straight (fast_straight_range); independently drawn numbers almost never give "fast, no sidestep,
         no turn"

    "Moving" = neither standing nor turning in place. Fractions are of the moving robots: the standing fraction is raised
    from 2% to 25% by upstream's curriculum, and a fraction of all robots would drift with it."""

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        c = self.vel_command_b
        # turning in place: upstream sets vx and vy to exactly 0 (uniform sampling never hits exactly 0)
        turning = (c[env_ids, 0] == 0.0) & (c[env_ids, 1] == 0.0)
        movers = env_ids[~turning & ~self.is_standing_env[env_ids]]
        if len(movers) == 0:
            return

        p = self.cfg.slow_skew_power
        if p != 1.0:
            lo, hi = self.cfg.ranges.lin_vel_x
            backward = c[movers, 0] < 0.0                                   # keep the sampled direction
            vmax = torch.where(backward, abs(lo), hi)
            u = torch.rand(len(movers), device=self.device)
            c[movers, 0] = torch.where(backward, -1.0, 1.0) * vmax * u ** p

        # two straight buckets (vy = wz = 0), split by one random number so they never overlap
        q = self.cfg.rel_slow_straight_envs
        s = self.cfg.rel_fast_straight_envs
        r = torch.rand(len(movers), device=self.device)
        for ids, rng in ((movers[r < q], self.cfg.slow_straight_range),
                         (movers[(r >= q) & (r < q + s)], self.cfg.fast_straight_range)):
            if len(ids) > 0:
                c[ids, 0] = torch.empty(len(ids), device=self.device).uniform_(*rng)
                c[ids, 1] = 0.0
                c[ids, 2] = 0.0

        self.vel_command_w[env_ids] = self.vel_command_b[env_ids]


@_dataclass(kw_only=True)
class PetGaitVelocityCommandCfg(VelocityCommandCommandOnlyCfg):
    rel_slow_straight_envs: float = 0.15            # fraction of moving robots practicing slow straight walking
    slow_straight_range: tuple[float, float] = (0.03, 0.15)
    slow_skew_power: float = 2.0                    # 1 = upstream's uniform sampling
    rel_fast_straight_envs: float = 0.0             # fraction practicing fast straight walking (0 = off)
    fast_straight_range: tuple[float, float] = (0.15, 0.4)

    def build(self, env: ManagerBasedRlEnv) -> PetGaitVelocityCommand:
        return PetGaitVelocityCommand(self, env)
