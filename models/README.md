# Policies trained in this project

| File | What it is | Validated |
|---|---|---|
| `petgait_v10.onnx` | The pet gait: walking policy used by the whole system (`pet/body/sim.py`, policy `walk`). Trained with `gait/` (PPO, seed 42). | Simulation: `python -m gait.eval` (pass criteria C1–C7 at 7.4 V and 6.5 V). Not run on hardware. |
| `spin_hop.onnx` | Spin hop in place, a by-product of an early reward experiment (feet air time weighted 5×). Used only by the `spin_hop` gesture and `zoomies`. | Simulation only, without domain randomization checks. Not run on hardware. |

Both take the same 61-dimensional observation and 14 joint targets as the official Microduck policies, so the
scheduler can switch between them and the official ones.

The official pretrained policies (velstand, sit/stand, stand, kicks) are not included: download them from
[pollen-robotics/microduck-policies](https://huggingface.co/pollen-robotics/microduck-policies) (revision d5a8b55)
and point `MICRODUCK_POLICY_DIR` at the folder.

Trained on the Microduck environment ([microduck_rl](https://github.com/pollen-robotics/microduck_rl), Apache-2.0).
