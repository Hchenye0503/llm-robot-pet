"""Register the pet gait task with mjlab at runtime, so the upstream repo stays unmodified.

The task id is distinct from anything upstream defines; importing this module twice is harmless.
"""

import mjlab.tasks  # noqa: F401  (populates the registry with the upstream tasks first)
from mjlab.tasks.registry import list_tasks, register_mjlab_task
from mjlab_microduck.tasks import MicroduckOnPolicyRunner

from .env_cfg import MicroduckPetGaitRlCfg, make_microduck_petgait_env_cfg

TASK_ID = "LlmRobotPet-PetGait-Flat-MicroDuck"

if TASK_ID not in list_tasks():
    register_mjlab_task(
        task_id=TASK_ID,
        env_cfg=make_microduck_petgait_env_cfg(),
        play_env_cfg=make_microduck_petgait_env_cfg(play=True),
        rl_cfg=MicroduckPetGaitRlCfg,
        runner_cls=MicroduckOnPolicyRunner,
    )
