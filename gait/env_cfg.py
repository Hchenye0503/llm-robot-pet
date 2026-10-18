"""Pet gait task: the Microduck velocity task with three changes (plus three added terms, v4-v10).

Pass criteria (fixed before training, eval.py): dead zone 0.05 m/s; tracking at five speeds within
0.02 + 10% x command (0.48 s average); 0.4 <-> 0 within 0.48 s; 0 falls in 100; standing still at a zero command;
all at both 7.4 V and 6.5 V battery.

Changes (all other rewards, domain randomization and observations are the velocity task's -> 61-dim observation,
so the policy plugs into the same planner as the official ones):
  1. track_linear_velocity -> PetGaitLinVelTracking (0.48 s average speed, sigma = k x (0.02 + 10% x command),
     k shrinks with performance)
  2. commands -> PetGaitVelocityCommandCfg (straight buckets + slow-skewed sampling)
  3. command resampling 3-8 s -> 1.5-3 s
Added: standing_foot_lift (v4), track_yaw_avg (v5), TURN_FLOOR (v10).
"""

import dataclasses

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import CurriculumTermCfg, RewardTermCfg

from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    MicroduckRlCfg,
    make_microduck_velocity_env_cfg,
)

from . import mdp_petgait

# 1: velocity tracking
K_INIT = 8.0                 # starting k: at a 0.2 command sigma = 8 x 0.04 = 0.32 ~ upstream's sqrt(0.1) = 0.316
K_THRESHOLD = 0.42           # v6: 0.5 -> 0.42. v2-v5 all stalled at k = 1.68 (score crept to 0.49 and stayed), i.e.
                             # training stayed 1.68x looser than the pass criterion
K_SHRINK = 0.8               # shrink to 80% each time -> 8 reaches 1 in 10 steps
K_CHECK_EVERY = 50 * 24      # check every 50 iterations (1 iteration = 24 steps)
WINDOW_S = 0.48              # one gait cycle (FFT of the baseline gait)
TURN_FLOOR = 0.04            # v10 (None = v9): with a turn command (|wz| > 0.01), linear sigma >= k x 0.04 (the width
                             # at 0.2 m/s). v9 lost in-place turning on 2/3 seeds: sigma was only k x 0.02 while turning,
                             # the stepping residual ~0.017 cost ~1.2 per step
TRACK_WEIGHT = 2.75          # v9 (None = upstream 2.0). v8's 3.5 reached 0.390 at 0.4 (randomized) but fell 17/1200,
                             # stepped in place at a zero command and walked less straight (overdose); 2.0 gives ~0.338.
                             # 2.75 is the middle of the dose curve: target >= 0.36 at 0.4 and <= 2/600 falls

# 2: command distribution
MAX_FORWARD = 0.4            # v5: forward command limit (m/s), as upstream; backward stays -0.4 (animals mostly turn)
SLOW_STRAIGHT_FRACTION = 0.15
SLOW_STRAIGHT_RANGE = (0.03, 0.15)
SLOW_SKEW_POWER = 2.0
FAST_STRAIGHT_FRACTION = 0.15   # enough for the fast gait to be learned
FAST_STRAIGHT_RANGE = (0.25, MAX_FORWARD)   # v7: lower bound 0.15 -> 0.25 (straight 0.35-0.4 from 3% to 5% of commands;
                                            # the fast span becomes twice the slow one). Cost: no straight bucket at 0.15-0.25

# 4 (v4): no stepping at a zero command
STILL_WEIGHT = -1.0          # per foot in the air, per step (negative). 0 = off
STILL_STOPPED = 0.05         # penalize only once the average speed is below this (the pass criterion's "stopped")

# 5 (v5): walk straight -- a second yaw score on the 0.48 s average yaw rate (upstream's term untouched)
YAW_WEIGHT = 1.0             # weight of this term (positive). 0 = off -> identical to v4
YAW_BASE = 0.05              # allowed average yaw rate walking straight (rad/s) ~ 3 deg/s

# 3: command resampling interval
RESAMPLE_S = (1.5, 3.0)


def make_microduck_petgait_env_cfg(play: bool = False, rough: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_microduck_velocity_env_cfg(play=play, rough=rough)

    # 2 + 3: replace the command generator (keep every upstream setting, add the bucket fields, change the interval)
    old = cfg.commands["twist"]
    cfg.commands["twist"] = mdp_petgait.PetGaitVelocityCommandCfg(
        **vars(old),
        rel_slow_straight_envs=SLOW_STRAIGHT_FRACTION,
        slow_straight_range=SLOW_STRAIGHT_RANGE,
        slow_skew_power=SLOW_SKEW_POWER,
        rel_fast_straight_envs=FAST_STRAIGHT_FRACTION,
        fast_straight_range=FAST_STRAIGHT_RANGE,
    )
    cfg.commands["twist"].resampling_time_range = RESAMPLE_S
    lo, _ = cfg.commands["twist"].ranges.lin_vel_x
    cfg.commands["twist"].ranges.lin_vel_x = (lo, MAX_FORWARD)

    # 1: replace velocity tracking (weight as upstream unless TRACK_WEIGHT is set)
    weight = cfg.rewards["track_linear_velocity"].weight if TRACK_WEIGHT is None else TRACK_WEIGHT
    cfg.rewards["track_linear_velocity"] = RewardTermCfg(
        func=mdp_petgait.PetGaitLinVelTracking,
        weight=weight,
        params={
            "command_name": "twist",
            "k_init": K_INIT,
            "window_s": WINDOW_S,
            "check_every_steps": K_CHECK_EVERY,
            "threshold": K_THRESHOLD,
            "shrink": K_SHRINK,
            "k_min": 1.0,
            "turn_floor": TURN_FLOOR,
        },
    )
    # 4: feet off the ground while it should stand still (STILL_WEIGHT = 0 -> term not added)
    if STILL_WEIGHT != 0.0:
        cfg.rewards["standing_foot_lift"] = RewardTermCfg(
            func=mdp_petgait.standing_foot_lift,
            weight=STILL_WEIGHT,
            params={"command_name": "twist", "sensor_name": "feet_ground_contact", "stopped": STILL_STOPPED},
        )
    # 5: average-yaw tracking (YAW_WEIGHT = 0 -> term not added)
    if YAW_WEIGHT != 0.0:
        cfg.rewards["track_yaw_avg"] = RewardTermCfg(
            func=mdp_petgait.PetGaitYawTracking,
            weight=YAW_WEIGHT,
            params={"command_name": "twist", "base": YAW_BASE, "window_s": WINDOW_S},
        )
    cfg.curriculum["petgait_k"] = CurriculumTermCfg(
        func=mdp_petgait.petgait_k_log,
        params={"reward_name": "track_linear_velocity"},
    )
    return cfg


MicroduckPetGaitRlCfg = dataclasses.replace(
    MicroduckRlCfg,
    experiment_name="petgait",
    run_name="petgait",
)
