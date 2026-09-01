# Fast Joint LIBERO Diffusion Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, train, and evaluate one compact language-conditioned Diffusion Policy on the four modified LIBERO suites using at most ten demonstrations per task.

**Architecture:** Convert selected RLDS episodes into a compact cache of frozen ResNet-18 features, state, action, language, and episode boundaries. Train a byte-language-conditioned residual MLP denoiser over eight-action chunks with DDIM, then evaluate it using unmodified LIBERO success predicates and traceable per-episode artifacts.

**Tech Stack:** Python 3.10, PyTorch 2.2.2 CUDA 12.1, torchvision 0.17.2, TensorFlow/TFDS 2.15, diffusers 0.30.3, LIBERO, MuJoCo, pytest, JSON/JSONL, Matplotlib.

## Global Constraints

- Use one RTX 3090 and do not stop unrelated GPU jobs.
- Use all four suites but at most ten successful demonstrations per task.
- Use only 128×128 static third-person images; do not use wrist images.
- Freeze ImageNet ResNet-18 and cache FP16 visual features before policy training.
- Use one seed, 20,000 initial steps, checkpoints every 5,000 steps, and FP16.
- Use the exact LIBERO `done` result; no expert recovery, scripted servoing, or relaxed predicates.
- Preserve every run's logs and never replace measured values with reference values.
- The workspace root is not currently a valid Git worktree. Do not initialize or repair Git as part of this experiment; replace commit steps with an artifact manifest entry stating `git_available=false`.

---

### Task 1: Action and window contracts

**Files:**
- Create: `tools/libero_diffusion_contracts.py`
- Test: `tests/test_libero_diffusion_contracts.py`

**Interfaces:**
- Produces: `transform_dataset_gripper(raw: np.ndarray) -> np.ndarray`
- Produces: `dataset_gripper_to_env(value: np.ndarray) -> np.ndarray`
- Produces: `ActionNormalizer.fit(actions: np.ndarray) -> ActionNormalizer`
- Produces: `ActionNormalizer.normalize(actions: np.ndarray) -> np.ndarray`
- Produces: `ActionNormalizer.denormalize(actions: np.ndarray) -> np.ndarray`
- Produces: `build_window_indices(lengths: Sequence[int], obs_horizon: int, action_horizon: int) -> list[WindowIndex]`

- [ ] **Step 1: Write failing action-contract tests**

```python
import numpy as np

from tools.libero_diffusion_contracts import (
    dataset_gripper_to_env,
    transform_dataset_gripper,
)


def test_libero_gripper_matches_official_transform_and_environment_inverse():
    raw = np.array([-1.0, 1.0], dtype=np.float32)
    transformed = transform_dataset_gripper(raw)
    np.testing.assert_array_equal(transformed, np.array([1.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(
        dataset_gripper_to_env(transformed),
        np.array([-1.0, 1.0], dtype=np.float32),
    )
```

- [ ] **Step 2: Run the action test and verify RED**

Run:

```powershell
.venv\Scripts\python.exe -m pytest -q tests/test_libero_diffusion_contracts.py
```

Expected: collection fails because `tools.libero_diffusion_contracts` does not exist.

- [ ] **Step 3: Implement the minimal gripper conversion**

```python
def transform_dataset_gripper(raw):
    raw = np.asarray(raw, dtype=np.float32)
    return 1.0 - np.clip(raw, 0.0, 1.0)


def dataset_gripper_to_env(value):
    value = np.asarray(value, dtype=np.float32)
    return np.where(value >= 0.5, -1.0, 1.0).astype(np.float32)
```

- [ ] **Step 4: Add failing normalization and boundary tests**

```python
def test_action_normalizer_round_trip_uses_robust_quantiles():
    actions = np.arange(140, dtype=np.float32).reshape(20, 7)
    normalizer = ActionNormalizer.fit(actions)
    restored = normalizer.denormalize(normalizer.normalize(actions))
    np.testing.assert_allclose(restored, actions, atol=1e-4)


def test_windows_never_cross_episode_boundaries():
    rows = build_window_indices([3, 4], obs_horizon=2, action_horizon=3)
    assert all(row.episode_index in (0, 1) for row in rows)
    assert all(row.action_stop <= (3 if row.episode_index == 0 else 4) for row in rows)
```

- [ ] **Step 5: Verify the new tests fail for missing interfaces**

Run the same focused pytest command.

Expected: gripper test passes; normalization/window tests fail with missing names.

- [ ] **Step 6: Implement immutable `WindowIndex` and robust normalization**

`ActionNormalizer.fit` stores float32 `q01` and `q99`, replaces spans below
`1e-6` with one, and clips normalized values to `[-1, 1]`. Window indices pad
observations on the left and actions on the right within the same episode.

- [ ] **Step 7: Verify GREEN and record the task artifact**

Run:

```powershell
.venv\Scripts\python.exe -m pytest -q tests/test_libero_diffusion_contracts.py
```

Expected: all tests pass. Write `outputs/experiment_4_6/fast_reproduction/diffusion_policy/artifacts/task1.json` with the test command, timestamp, result, and `git_available=false`.

---

### Task 2: Deterministic RLDS selection and feature cache

**Files:**
- Create: `tools/prepare_libero_diffusion_data.py`
- Test: `tests/test_prepare_libero_diffusion_data.py`

**Interfaces:**
- Consumes: `transform_dataset_gripper`
- Produces: `EpisodeArrays` with `images`, `states`, `actions`, `instruction`, and `suite`
- Produces: `select_episodes(episodes: Iterable[EpisodeArrays], max_per_instruction: int) -> list[EpisodeArrays]`
- Produces: `validate_expanded_tfrecords(dataset_root: Path) -> dict`
- Produces CLI cache: `feature_cache/episodes/*.npz` and `dataset_manifest.json`

- [ ] **Step 1: Write failing deterministic-selection tests**

```python
def test_selection_caps_each_full_instruction_without_cross_suite_collisions():
    episodes = [
        fake_episode("libero_goal", "turn on the stove", index)
        for index in range(12)
    ] + [
        fake_episode("libero_spatial", "turn on the stove", index)
        for index in range(12)
    ]
    selected = select_episodes(episodes, max_per_instruction=10)
    assert len(selected) == 20
    assert [row.episode_index for row in selected[:10]] == list(range(10))
```

- [ ] **Step 2: Verify RED**

Run:

```powershell
.venv\Scripts\python.exe -m pytest -q tests/test_prepare_libero_diffusion_data.py
```

Expected: import failure for the missing preparation module.

- [ ] **Step 3: Implement `EpisodeArrays` and deterministic selection**

Use `(suite, instruction)` as the grouping key, preserve source order, reject
empty instructions, and require image/state/action lengths to match.

- [ ] **Step 4: Add a failing LFS-pointer validation test**

```python
def test_tfrecord_validation_rejects_unexpanded_lfs_pointer(tmp_path):
    shard = tmp_path / "suite" / "1.0.0" / "data.tfrecord-00000-of-00001"
    shard.parent.mkdir(parents=True)
    shard.write_text("version https://git-lfs.github.com/spec/v1\n")
    with pytest.raises(RuntimeError, match="Git LFS pointer"):
        validate_expanded_tfrecords(tmp_path)
```

- [ ] **Step 5: Implement TFRecord validation and RLDS iteration**

Use `tfds.builder_from_directory(builder_dir=...)`, iterate the `train` split,
decode each episode to NumPy, select `observation.image`, `observation.state`,
`action`, and `language_instruction`, and apply the official gripper transform.
Require all four builder directories and exactly 96 expanded TFRecord shards.

- [ ] **Step 6: Add a failing synthetic feature-cache test**

Inject a tiny encoder callable returning a known `(T, 4)` feature matrix. Assert
that cache NPZ files contain FP16 features, float32 state/action arrays, UTF-8
instruction text, and exact episode boundaries in the manifest.

- [ ] **Step 7: Implement frozen ResNet-18 extraction and cache writing**

Load `ResNet18_Weights.DEFAULT`, remove the final classifier, set every parameter
to `requires_grad=False`, preprocess images at 128×128 with ImageNet
normalization, and write one compressed NPZ per episode. Refuse to overwrite a
cache whose source commit or selection configuration differs.

- [ ] **Step 8: Verify GREEN and run the real-data parser smoke**

Run:

```powershell
C:\Users\sjtu101\miniconda3\envs\openvla-libero\python.exe -m pytest -q tests/test_prepare_libero_diffusion_data.py
C:\Users\sjtu101\miniconda3\envs\openvla-libero\python.exe -m tools.prepare_libero_diffusion_data --dataset-root datasets/modified_libero_rlds --output-dir outputs/experiment_4_6/fast_reproduction/diffusion_policy --max-per-instruction 10 --image-size 128 --smoke-episodes 1
```

Expected: tests pass; smoke manifest contains one real decoded episode with
nonzero frames and a `(frames, 512)` feature array.

---

### Task 3: Compact language-conditioned diffusion model

**Files:**
- Create: `tools/libero_diffusion_model.py`
- Test: `tests/test_libero_diffusion_model.py`

**Interfaces:**
- Produces: `DiffusionPolicyConfig`
- Produces: `ByteInstructionEncoder.forward(tokens, lengths) -> Tensor[B, 128]`
- Produces: `CompactDiffusionPolicy.training_loss(batch) -> Tensor[]`
- Produces: `CompactDiffusionPolicy.sample_actions(condition, num_inference_steps=10) -> Tensor[B, 8, 7]`

- [ ] **Step 1: Write failing shape and conditioning tests**

```python
def test_training_loss_and_sampling_shapes():
    config = DiffusionPolicyConfig(
        visual_dim=512, state_dim=8, obs_horizon=2,
        action_dim=7, action_horizon=8,
    )
    model = CompactDiffusionPolicy(config)
    batch = synthetic_batch(batch_size=3, config=config)
    loss = model.training_loss(batch)
    assert loss.ndim == 0
    sampled = model.sample_actions(batch["condition"], num_inference_steps=2)
    assert sampled.shape == (3, 8, 7)


def test_language_bytes_change_the_condition():
    model = CompactDiffusionPolicy(DiffusionPolicyConfig())
    a = model.encode_instruction(["put the bowl on the stove"])
    b = model.encode_instruction(["turn on the stove"])
    assert not torch.allclose(a, b)
```

- [ ] **Step 2: Verify RED**

Run:

```powershell
C:\Users\sjtu101\miniconda3\envs\openvla-libero\python.exe -m pytest -q tests/test_libero_diffusion_model.py
```

Expected: import failure for the missing model module.

- [ ] **Step 3: Implement byte encoding and the condition projector**

Use byte IDs `0..255`, reserve `256` for padding, embedding dimension 32, a
single-layer GRU with hidden size 128, and a two-layer condition projector to
256 dimensions. Concatenate two visual features, two states, and language state.

- [ ] **Step 4: Implement the residual noise predictor and DDIM contract**

Flatten noisy actions to 56 values, concatenate 64-dimensional sinusoidal time
embedding and the 256-dimensional condition, use four 512-wide residual MLP
blocks, and predict 56 noise values. Use
`DDIMScheduler(num_train_timesteps=50, beta_schedule="squaredcos_cap_v2",
prediction_type="epsilon")`.

- [ ] **Step 5: Implement training loss and deterministic sampling**

Training samples one timestep per batch row and minimizes MSE against sampled
Gaussian noise. Sampling initializes Gaussian actions with a passed generator,
uses ten DDIM inference steps by default, and clips final normalized actions to
`[-1, 1]`.

- [ ] **Step 6: Verify GREEN, CPU determinism, and CUDA autocast**

Run:

```powershell
C:\Users\sjtu101\miniconda3\envs\openvla-libero\python.exe -m pytest -q tests/test_libero_diffusion_model.py
```

Expected: shape, conditioning, deterministic-generator, and FP16 autocast tests
all pass.

---

### Task 4: Training, checkpoints, and process evidence

**Files:**
- Create: `tools/train_libero_diffusion_policy.py`
- Test: `tests/test_train_libero_diffusion_policy.py`

**Interfaces:**
- Consumes: feature cache, `ActionNormalizer`, `CompactDiffusionPolicy`
- Produces: `FeatureWindowDataset`
- Produces: `save_checkpoint(...)`, `load_checkpoint(...)`
- Produces CLI artifacts: `checkpoints/step_*.pt`, `train_metrics.jsonl`, `run_manifest.json`, `training_process.png`

- [ ] **Step 1: Write failing dataset-boundary and checkpoint tests**

Create two synthetic cached episodes with distinct constant actions. Assert no
window mixes the constants. Train two optimizer steps, save, load into a fresh
model/optimizer, and assert step, parameters, optimizer state, normalizer, seed,
and data-manifest hash are identical.

- [ ] **Step 2: Verify RED**

Run:

```powershell
C:\Users\sjtu101\miniconda3\envs\openvla-libero\python.exe -m pytest -q tests/test_train_libero_diffusion_policy.py
```

Expected: import failure for the missing training module.

- [ ] **Step 3: Implement the memory-mapped feature window dataset**

Index windows once from NPZ metadata, load only the requested episode, maintain
a bounded per-worker episode cache, repeat boundary frames inside the episode,
and return normalized action chunks plus visual/state/text condition inputs.

- [ ] **Step 4: Implement atomic checkpoints and strict resume**

Write to `step_N.pt.tmp`, fsync, then replace `step_N.pt`. Reject resume when the
dataset-manifest SHA-256, model config, or normalization stats differ. Never
delete older 5k checkpoints.

- [ ] **Step 5: Implement the training CLI**

Defaults: seed 42, 20,000 steps, batch 256, learning rate `1e-4`, AdamW weight
decay `1e-4`, gradient norm 1.0, FP16 GradScaler, validation every 1,000 steps,
checkpoint every 5,000 steps, eight DataLoader workers. On CUDA OOM, exit with
an explicit recommendation to rerun at batch 128; do not silently alter the run.

- [ ] **Step 6: Implement metric logging and chart rendering**

Append one JSON object per metric event with UTC timestamp, step, train loss,
validation loss, learning rate, examples/sec, GPU name, allocated VRAM, and
peak VRAM. Render only actual logged points into `training_process.png`.

- [ ] **Step 7: Verify GREEN and run a two-step CUDA smoke**

Run:

```powershell
C:\Users\sjtu101\miniconda3\envs\openvla-libero\python.exe -m pytest -q tests/test_train_libero_diffusion_policy.py
C:\Users\sjtu101\miniconda3\envs\openvla-libero\python.exe -m tools.train_libero_diffusion_policy --experiment-dir outputs/experiment_4_6/fast_reproduction/diffusion_policy --steps 2 --batch-size 2 --checkpoint-every 1 --run-name cuda_smoke
```

Expected: tests pass; the smoke writes two finite loss rows and a reloadable
step-2 checkpoint.

---

### Task 5: Pure LIBERO evaluation wrapper

**Files:**
- Create: `tools/eval_libero_diffusion_policy.py`
- Test: `tests/test_eval_libero_diffusion_policy.py`

**Interfaces:**
- Consumes: trained checkpoint, normalizer, frozen ResNet-18, LIBERO benchmark
- Produces: `PolicyState`, `policy_action_chunk(...)`
- Produces: `evaluate_suite(...) -> dict`
- Produces CLI artifacts under `evaluation/<suite>/seed_<seed>/`

- [ ] **Step 1: Write failing action-contract and state-history tests**

Use a fake policy that emits known dataset-space gripper values. Assert the
environment receives `-1` for open and `+1` for close, the first observation is
duplicated to fill history, and exactly eight actions are queued before another
policy query.

- [ ] **Step 2: Verify RED**

Run:

```powershell
C:\Users\sjtu101\miniconda3\envs\openvla-libero\python.exe -m pytest -q tests/test_eval_libero_diffusion_policy.py
```

Expected: import failure for the missing evaluator.

- [ ] **Step 3: Implement checkpoint loading and policy preprocessing**

Require the checkpoint's config and normalizer, use the same 128×128 ImageNet
preprocessing and instruction byte encoding as training, and maintain exactly
two observation states.

- [ ] **Step 4: Implement pure LIBERO rollouts**

Use suite horizons 220/280/300/520 plus ten settling steps, fixed official init
states, and only `done` for success. Save every episode's instruction, initial
state index, success, steps, inference latency, error, action summary, and
representative first/middle/final PNG frames. Save MP4 when imageio/ffmpeg is
available; a video failure must not change episode success.

- [ ] **Step 5: Implement suite summaries and resumability**

Write an episode JSON immediately after each rollout. On resume, skip only an
episode whose JSON parses, matches checkpoint SHA-256 and seed, and contains a
boolean success. Compute suite success from concrete episode booleans.

- [ ] **Step 6: Verify GREEN and run one real episode**

Run:

```powershell
C:\Users\sjtu101\miniconda3\envs\openvla-libero\python.exe -m pytest -q tests/test_eval_libero_diffusion_policy.py
C:\Users\sjtu101\miniconda3\envs\openvla-libero\python.exe -m tools.eval_libero_diffusion_policy --checkpoint outputs/experiment_4_6/fast_reproduction/diffusion_policy/checkpoints/best.pt --suite libero_goal --task-ids 1 --trials-per-task 1 --seed 7 --output-dir outputs/experiment_4_6/fast_reproduction/diffusion_policy/evaluation_smoke
```

Expected: tests pass and one real episode JSON plus PNG evidence is written,
regardless of whether its measured success is true or false.

---

### Task 6: Formal runs and report evidence

**Files:**
- Create: `tools/build_fast_4_6_2_report.py`
- Test: `tests/test_build_fast_4_6_2_report.py`
- Create: `outputs/experiment_4_6/fast_reproduction/summary.json`
- Create: `outputs/experiment_4_6/fast_reproduction/4.6.2_缩减复现实验.md`

**Interfaces:**
- Consumes: baseline/OFT/Diffusion Policy episode artifacts
- Produces: `aggregate_method_suite(episode_paths) -> EvidenceRecord`
- Produces: a Chinese report chapter and comparison/process figures

- [ ] **Step 1: Write failing evidence aggregation tests**

Assert aggregation rejects missing/non-boolean success fields, counts only real
episode files, computes unrounded rates before display rounding, and labels each
row `local_measured`.

- [ ] **Step 2: Verify RED**

Run:

```powershell
.venv\Scripts\python.exe -m pytest -q tests/test_build_fast_4_6_2_report.py
```

Expected: import failure for the missing report module.

- [ ] **Step 3: Implement strict aggregation and provenance**

Record method, suite, successes, episodes, rate, checkpoint SHA-256, seed,
source paths, start/end timestamps, GPU, and protocol deviations. Call the
fourth suite `LIBERO-10（长时序任务）`; do not claim it is a mobile navigation
benchmark.

- [ ] **Step 4: Run bounded formal evaluations**

Run baseline, OFT, and Diffusion Policy for ten tasks per suite and one trial per
task first. Preserve the already valid OFT 39/40 smoke as a separately identified
local run; execute new trials only where a method/suite lacks ten concrete
episodes. If time allows, increase Diffusion Policy to three trials per task
without overwriting the first run.

- [ ] **Step 5: Generate real figures**

Create a three-method/four-suite success-rate chart from `summary.json`, a
training-process chart from `train_metrics.jsonl`, and a process montage using
representative PNG frames from at least one real rollout in each suite. Include
captions explaining source run, task, episode, and timestamp.

- [ ] **Step 6: Render the Chinese report chapter**

State the reduced protocol, exact trial counts, actual rates, confidence limits
for small samples, deviations from the paper, model/checkpoint identities,
training duration, and image insertion positions. Do not use official reported
rates as substitutes for missing local results.

- [ ] **Step 7: Run full verification**

Run:

```powershell
.venv\Scripts\python.exe -m pytest -q tests/test_libero_diffusion_contracts.py tests/test_prepare_libero_diffusion_data.py tests/test_libero_diffusion_model.py tests/test_train_libero_diffusion_policy.py tests/test_eval_libero_diffusion_policy.py tests/test_build_fast_4_6_2_report.py
.venv\Scripts\python.exe -m compileall -q tools tests
```

Expected: all focused tests pass and compileall exits zero. Then validate that
every figure path in the report exists and every displayed success rate can be
recomputed from concrete boolean episode files.

