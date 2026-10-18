"""Pet gait: the walking policy trained for this project (RL, PPO on the Microduck velocity task).

    mdp_petgait  reward terms and command generator (averaged-speed tracking with a command-scaled tolerance)
    env_cfg      the task: velocity task + three changes; every number has its version history next to it
    register     registers the task with mjlab at runtime (the upstream repo is not modified)
    train        python -m gait.train --env.scene.num-envs 2048 ...
    eval         the pass criteria C1-C7, fixed before training, at 7.4 V and 6.5 V

Needs the upstream environment (microduck_rl and its mjlab / torch install).
"""
