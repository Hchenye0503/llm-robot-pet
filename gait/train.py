"""Train the pet gait with the upstream training script, after registering the task.

    python -m gait.train --env.scene.num-envs 2048 --agent.seed 42
(run from the repo root with the microduck_rl environment; on Windows with a non-ASCII user name, point TEMP and
WARP_CACHE_PATH to ASCII paths first, or CUDA kernel compilation fails)
"""

import sys

from .register import TASK_ID


def main():
    from mjlab.scripts.train import main as mjlab_train_main
    sys.argv = [sys.argv[0], TASK_ID] + sys.argv[1:]
    return mjlab_train_main()


if __name__ == "__main__":
    sys.exit(main())
