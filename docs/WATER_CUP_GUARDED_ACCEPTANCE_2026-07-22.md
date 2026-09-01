# Water Cup Guarded Closed-Loop Acceptance (2026-07-22)

## Outcome

The household `pick up the glass cup and place it in the cabinet` task passes
closed-loop RoboCasa evaluation on both the nominal seed 0 and the held-out
seed 2 when OpenVLA is deployed with the stateful Cartesian safety/recovery
supervisor.

This is deliberately reported as **OpenVLA + state supervisor**, not as a pure
OpenVLA policy result. The supervisor currently reads simulator end-effector,
object and grasp state. A physical robot must provide the same state contract
from object-pose perception and robot proprioception.

## Model and data

- Adapter: `models/openvla-water-cup-dagger-r1-e1`
- Training updates: 11,857 / 11,857
- Nominal samples: 7,635
- Unique DAgger recovery samples: 2,111
- Weighted recovery samples: 4,222
- Held-out seed: 2

## Acceptance evidence

| Seed | Result | Steps | Grasped | Object lift | Minimum EEF distance | Direct OpenVLA steps |
|---:|---|---:|---|---:|---:|---:|
| 0 | PASS | 545 | yes | 0.5307 m | 0.0097 m | 435 / 545 (79.8%) |
| 2 | PASS | 494 | yes | 0.5424 m | 0.0086 m | 239 / 494 (48.4%) |

Reports:

- `outputs/water_cup_guarded_r1_seed0/guarded_report.json`
- `outputs/water_cup_guarded_r1_seed2/guarded_report.json`

## Root cause of the unguarded failure

The unguarded DAgger LoRA completed training but failed seed 0 at 700 steps:
it never grasped, lifted the object by only `6.18e-7 m`, and stopped no closer
than `0.277 m`. On the exact nominal and DAgger training frame at index 6, the
model predicted a post-grasp close/reverse action despite the label requiring
continued open-gripper approach. The single RGB image plus constant task
instruction does not reliably encode execution phase, previous action or
proprioceptive state.

## Runtime command

Start the trained model server in the `openvla` environment:

```powershell
python -m tools.openvla_tcp_server_water_cup --port 8773 --adapter-dir C:\OpenVLA-Simulator\models\openvla-water-cup-dagger-r1-e1
```

Run guarded RoboCasa evaluation in the `robocasa` environment:

```powershell
python -m tools.openvla_gym_water_cup_guarded_rollout --seed 0 --steps 900 --port 8773 --max-translation-error 0.15 --output-dir C:\OpenVLA-Simulator\outputs\water_cup_guarded_manual_seed0
```

## Next engineering step

Replace simulator ground-truth supervisor inputs with the project perception
contract: object 6D pose, end-effector pose, gripper/contact state and a
phase/history state. Then phase-condition the VLA policy (or train a temporal
policy) and measure a decreasing supervisor intervention rate without reducing
task success.
