# RoboCasa OFT Continuous-Head Adaptation Design

## Objective

Produce the fastest technically valid path from the current RoboCasa365 data to a pure visual-language-action closed-loop policy for the approved 8-task, 60-object midterm matrix. The acceptance rollout must be simulation-only and must not use scripted actions, privileged simulator state, an oracle, or expert recovery to alter policy actions.

The immediate milestone is one successful representative closed-loop task. Expansion to 8 tasks and then 60 objects is conditional on that milestone; failed architectures are not repeated at scale.

## Evidence Driving the Design

- The environment and asset pipeline has already completed 60/60 simulator reset checks.
- The original one-step OpenVLA pipeline completed 60/60 single-action calls and 600/600 augmented-image calls, proving that inference runs, but not that tasks succeed.
- Pure one-step OpenVLA closed-loop evaluation completed 8/8 executions with 0/8 task success.
- LoRA revisions R1, R2, and R3 did not solve closed-loop control. R3 produced the exact same action for all 60 decisions on an object/task pair present in training.
- The R3 source labels are strongly modal by channel: the most common near-zero token accounts for up to 47.5% of a motion channel and closed gripper accounts for 69.1% of the gripper channel. Independent next-token cross-entropy therefore rewards the observed stationary policy.
- The local OpenVLA-OFT checkpoint has already completed 39/40 LIBERO rollouts without expert recovery. Its continuous 8-step action head is the relevant working reference.
- The local RoboCasa365 mirror includes the inputs required for OFT adaptation: agent-view video, eye-in-hand video, 16-dimensional proprioception, and 12-dimensional PandaOmron actions.

## Considered Approaches

### Selected: frozen OpenVLA-OFT backbone with a RoboCasa continuous head

Freeze the 7B vision-language backbone and train only a new continuous L1 action head and an 8-dimensional proprioception projector. This removes the discrete action-token mode-collapse failure while keeping OpenVLA visual and language features. Freezing the backbone minimizes RTX 3090 memory use and training time.

### Rejected: another one-step token LoRA

Three iterations have already failed, including an exact task/object training match. Additional sampling changes do not address the per-channel modal-token objective.

### Deferred: full OpenVLA-OFT fine-tuning

The official batch-size-1 recipe requires about 25 GB VRAM, above the available 24 GB RTX 3090. It is slower and riskier than training the small continuous components first.

## Data Contract

Each training sample contains:

- Primary image: `observation.images.robot0_agentview_left`.
- Wrist image: `observation.images.robot0_eye_in_hand`.
- Language instruction from the RoboCasa365 task table.
- Raw state: `observation.state`, ordered as base position (3), base quaternion (4), relative end-effector position (3), relative end-effector quaternion (4), and gripper joint position (2).
- OFT proprioception: relative end-effector position (state indices 7:10), relative end-effector quaternion (10:14), and the mean of the two gripper joint positions (14:16), yielding 8 values.
- Arm action: `action[5:12]`, ordered as end-effector delta position (3), delta axis-angle rotation (3), and gripper (1). Base motion and control mode are excluded.
- Target: an 8-step future arm-action chunk. Near episode end, indices are clamped to the last valid action, matching the OFT trajectory transform.

The split is episode-level so frames from one demonstration cannot occur in both training and validation. Action and proprioception statistics are calculated from the training split only using 1st/99th-percentile normalization.

## Components

1. **Dataset index and decoder** validates LeRobot parquet/video alignment and produces two images, 8-D proprioception, language, and 8x7 action chunks.
2. **OFT feature adapter** loads the existing combined OFT checkpoint with the backbone frozen and exposes the hidden action features used by the official L1 head.
3. **Continuous trainer** trains only the RoboCasa proprio projector and action head, records validation L1 error, peak GPU memory, checkpoints, and immutable run metadata.
4. **Persistent inference service** accepts two images, 8-D proprioception, and language and returns an 8x7 action chunk with RoboCasa training statistics.
5. **RoboCasa rollout adapter** reads only public observations for policy input, executes the returned chunk through the stationary-base action mapping, and reads privileged state only for the official success predicate and evidence logging.

## Resource and Failure Policy

- Run a single-batch forward/backward smoke test before full training.
- Peak allocated GPU memory must stay below 23.5 GB; otherwise reduce image/batch memory without changing the data or acceptance protocol.
- Training starts with one epoch and early stopping on validation L1. Additional epochs are allowed only while validation improves.
- Every run is resumable and writes progress at fixed intervals.
- A failed representative rollout triggers trajectory diagnosis. It does not trigger a 60-object batch.
- Expert or scripted controllers may generate separately disclosed training demonstrations, but no such controller may run during pure-policy acceptance.

## Verification Gates

1. **Data gate:** camera, state, action, episode, and instruction alignment tests pass; no train/validation episode leakage.
2. **Memory gate:** one real OFT training batch completes below 23.5 GB.
3. **Learning gate:** validation action L1 improves over the unadapted OFT head, predicted chunks contain finite values, and motion channels have non-zero variance across distinct observations.
4. **Representative gate:** one task/object pair with exact training coverage succeeds under the official RoboCasa predicate in a 300-decision maximum rollout with no expert recovery.
5. **Category gate:** one representative from each of 8 categories succeeds.
6. **Matrix gate:** all 60 selected rows execute with per-row video, first/last frames, trajectory, success predicate, and proxy disclosure. Only predicate success counts as pass.

## Outputs

- Dataset alignment and statistics manifest.
- GPU smoke-test report.
- Adapted action-head and proprio-projector checkpoints.
- Training and validation curves.
- Per-rollout JSON, MP4, first frame, last frame, and key-frame evidence.
- Final 8-category and 60-object manifests that distinguish native objects from functional proxies and never equate successful execution with task success.

## Approved Decision

Use the frozen-backbone continuous-head approach. Do not resume one-step token LoRA, do not run full-matrix tests before the representative gate, and do not count hybrid/expert recovery as pure VLA success.
