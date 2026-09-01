# RoboCasa OFT Continuous-Head Adaptation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and evaluate a pure RoboCasa visual-language-action policy by freezing the existing OpenVLA-OFT backbone and adapting only an 8-D proprioception projector and continuous 8x7 action head.

**Architecture:** A deterministic data layer converts the local RoboCasa365 LeRobot mirror into aligned agent-view, wrist-view, 8-D proprioception, and 8-step arm-action chunks. The existing combined OpenVLA-OFT checkpoint supplies frozen image-language features; only the continuous head and proprio projector receive gradients. A persistent inference service feeds action chunks to a pure-policy RoboCasa rollout and records official predicate evidence.

**Tech Stack:** Python 3.10/3.11, PyTorch, Transformers/OpenVLA-OFT, PyArrow/Pandas, NumPy, FFmpeg/ImageIO, Gymnasium, RoboCasa, MuJoCo, pytest.

## Global Constraints

- Acceptance is simulation-only and uses no scripted action, privileged-state action, oracle, or expert recovery.
- Use `agentview_left` as the primary image and `eye_in_hand` as the wrist image.
- Convert `observation.state[7:10] + observation.state[10:14] + mean(observation.state[14:16])` to 8-D proprioception.
- Convert `action[5:12]` to the 7-D stationary-base arm action.
- Predict 8 future actions and clamp chunk indices at the final episode frame.
- Split by episode; statistics come from training episodes only.
- Peak allocated GPU memory must remain below 23.5 GB.
- Do not execute the 60-object matrix until one representative and all 8 category representatives pass.
- The workspace has no `.git` directory. Each task writes a verification artifact with file hashes instead of creating a commit.

---

### Task 1: RoboCasa OFT data contract

**Files:**
- Create: `tools/robocasa_oft_contract.py`
- Create: `tools/tests/test_robocasa_oft_contract.py`

**Interfaces:**
- Consumes: raw 16-D states and 12-D actions from RoboCasa365.
- Produces: `extract_proprio(state) -> np.ndarray`, `extract_arm_action(action) -> np.ndarray`, `future_action_chunk(actions, index, chunk_size=8) -> np.ndarray`, and `episode_split(episode_ids, seed=82) -> dict[str, list[int]]`.

- [ ] **Step 1: Write the failing contract tests**

```python
import numpy as np
from tools.robocasa_oft_contract import (
    episode_split,
    extract_arm_action,
    extract_proprio,
    future_action_chunk,
)


def test_extracts_oft_proprio_from_pandaomron_state():
    state = np.arange(16, dtype=np.float32)
    np.testing.assert_allclose(
        extract_proprio(state),
        np.array([7, 8, 9, 10, 11, 12, 13, 14.5], dtype=np.float32),
    )


def test_extracts_stationary_base_arm_action():
    action = np.arange(12, dtype=np.float32)
    np.testing.assert_array_equal(extract_arm_action(action), action[5:12])


def test_future_chunk_clamps_at_episode_end():
    actions = np.arange(21, dtype=np.float32).reshape(3, 7)
    chunk = future_action_chunk(actions, 1, chunk_size=4)
    np.testing.assert_array_equal(chunk, actions[[1, 2, 2, 2]])


def test_episode_split_has_no_leakage_and_is_deterministic():
    first = episode_split(list(range(20)), seed=82)
    second = episode_split(list(range(20)), seed=82)
    assert first == second
    assert not (set(first["train"]) & set(first["val"]))
    assert not (set(first["train"]) & set(first["test"]))
    assert not (set(first["val"]) & set(first["test"]))
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tools/tests/test_robocasa_oft_contract.py -q`

Expected: collection fails because `tools.robocasa_oft_contract` does not exist.

- [ ] **Step 3: Implement the minimal contract**

```python
from __future__ import annotations

import random
from typing import Sequence

import numpy as np


def extract_proprio(state: Sequence[float]) -> np.ndarray:
    values = np.asarray(state, dtype=np.float32)
    if values.shape != (16,):
        raise ValueError("RoboCasa365 state must contain 16 values")
    return np.concatenate((values[7:14], [values[14:16].mean()])).astype(np.float32)


def extract_arm_action(action: Sequence[float]) -> np.ndarray:
    values = np.asarray(action, dtype=np.float32)
    if values.shape != (12,):
        raise ValueError("RoboCasa365 action must contain 12 values")
    return values[5:12].copy()


def future_action_chunk(actions: np.ndarray, index: int, chunk_size: int = 8) -> np.ndarray:
    values = np.asarray(actions, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 7 or not 0 <= index < len(values):
        raise ValueError("actions must be non-empty Nx7 and index must be valid")
    indices = np.minimum(np.arange(index, index + chunk_size), len(values) - 1)
    return values[indices].copy()


def episode_split(episode_ids: Sequence[int], seed: int = 82) -> dict[str, list[int]]:
    ids = sorted({int(value) for value in episode_ids})
    if len(ids) < 3:
        raise ValueError("at least three episodes are required")
    random.Random(seed).shuffle(ids)
    train_end = max(1, round(0.8 * len(ids)))
    held = ids[train_end:]
    val_end = max(1, len(held) // 2)
    return {"train": sorted(ids[:train_end]), "val": sorted(held[:val_end]), "test": sorted(held[val_end:])}
```

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest tools/tests/test_robocasa_oft_contract.py -q`

Expected: `4 passed`.

- [ ] **Step 5: Record immutable verification**

Run: `python -m pytest tools/tests/test_robocasa_oft_contract.py -q > outputs/midterm_testing_vla82/oft_contract_test.txt`

Expected artifact: `outputs/midterm_testing_vla82/oft_contract_test.txt` contains `4 passed`.

---

### Task 2: Dual-view episode cache

**Files:**
- Create: `tools/prepare_robocasa_oft_cache.py`
- Create: `tools/tests/test_prepare_robocasa_oft_cache.py`
- Reuse: `tools/robocasa_episode_archive_v2.py`
- Reuse: `datasets/vla82_robocasa365_19class/training_manifest.json`

**Interfaces:**
- Consumes: the selected 95-episode manifest and flat LeRobot v3 source.
- Produces: one directory per episode containing `agentview_left.mp4`, `eye_in_hand.mp4`, `states.npy`, `actions.npy`, plus `datasets/vla82_robocasa365_oft/cache_manifest.json`.

- [ ] **Step 1: Write failing cache-record tests**

```python
import numpy as np
from tools.prepare_robocasa_oft_cache import build_aligned_arrays, validate_lengths


def test_build_aligned_arrays_extracts_expected_shapes():
    states = np.arange(48, dtype=np.float32).reshape(3, 16)
    actions = np.arange(36, dtype=np.float32).reshape(3, 12)
    result = build_aligned_arrays(states, actions)
    assert result["proprio"].shape == (3, 8)
    assert result["actions"].shape == (3, 7)
    assert result["action_chunks"].shape == (3, 8, 7)


def test_validate_lengths_rejects_camera_mismatch():
    try:
        validate_lengths(10, 10, 9, 10)
    except ValueError as error:
        assert "alignment" in str(error)
    else:
        raise AssertionError("camera mismatch was accepted")
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tools/tests/test_prepare_robocasa_oft_cache.py -q`

Expected: collection fails because the cache module does not exist.

- [ ] **Step 3: Implement aligned array construction and the CLI**

```python
def validate_lengths(states: int, actions: int, primary_frames: int, wrist_frames: int) -> None:
    if len({states, actions, primary_frames, wrist_frames}) != 1:
        raise ValueError(
            f"episode alignment mismatch: states={states}, actions={actions}, "
            f"primary={primary_frames}, wrist={wrist_frames}"
        )


def build_aligned_arrays(states: np.ndarray, actions: np.ndarray) -> dict[str, np.ndarray]:
    proprio = np.stack([extract_proprio(row) for row in states])
    arm = np.stack([extract_arm_action(row) for row in actions])
    chunks = np.stack([future_action_chunk(arm, index) for index in range(len(arm))])
    return {"proprio": proprio, "actions": arm, "action_chunks": chunks}
```

The CLI must reuse `episode_assets`, `chunk_relative_rows`, and `_cut_video` for both camera keys, read `observation.state` and `action` from the episode's source parquet slice, validate all four lengths, and write a manifest entry containing task class, instruction, episode index, split, frame count, absolute artifact paths, and SHA-256 values.

- [ ] **Step 4: Verify unit tests and a three-episode smoke cache**

Run: `python -m pytest tools/tests/test_prepare_robocasa_oft_cache.py -q`

Expected: `2 passed`.

Run: `python -m tools.prepare_robocasa_oft_cache --limit 3 --output datasets/vla82_robocasa365_oft_smoke`

Expected: three entries with equal video/state/action lengths and no alignment failure.

- [ ] **Step 5: Build and record the full selected cache**

Run: `python -m tools.prepare_robocasa_oft_cache --output datasets/vla82_robocasa365_oft`

Expected artifact: `datasets/vla82_robocasa365_oft/cache_manifest.json` reports 95 episodes, disjoint splits, and zero alignment failures.

---

### Task 3: Normalization and frozen OFT model adapter

**Files:**
- Create: `tools/robocasa_oft_model.py`
- Create: `tools/tests/test_robocasa_oft_model.py`
- Reuse: `third_party/openvla-oft/prismatic/models/action_heads.py`
- Reuse: `third_party/openvla-oft/prismatic/models/projectors.py`

**Interfaces:**
- Consumes: training-split proprio/action chunks and the combined OFT checkpoint.
- Produces: `quantile_stats`, `normalize`, `unnormalize`, `install_robocasa_stats`, and `build_trainable_components`.

- [ ] **Step 1: Write failing normalization tests**

```python
import numpy as np
from tools.robocasa_oft_model import normalize, quantile_stats, unnormalize


def test_quantile_round_trip():
    values = np.linspace(-2, 2, 500, dtype=np.float32).reshape(100, 5)
    stats = quantile_stats(values)
    in_range = values[1:-1]
    restored = unnormalize(normalize(in_range, stats), stats)
    np.testing.assert_allclose(restored, in_range, atol=1e-5)


def test_normalization_is_finite_for_constant_channel():
    values = np.ones((10, 3), dtype=np.float32)
    assert np.isfinite(normalize(values, quantile_stats(values))).all()
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tools/tests/test_robocasa_oft_model.py -q`

Expected: collection fails because `tools.robocasa_oft_model` does not exist.

- [ ] **Step 3: Implement statistics and component construction**

```python
def quantile_stats(values: np.ndarray) -> dict[str, list[float]]:
    data = np.asarray(values, dtype=np.float32)
    low = np.quantile(data, 0.01, axis=0).astype(np.float32)
    high = np.quantile(data, 0.99, axis=0).astype(np.float32)
    return {"q01": low.tolist(), "q99": high.tolist(), "mask": ((high - low) > 1e-6).tolist()}


def normalize(values: np.ndarray, stats: dict) -> np.ndarray:
    data = np.asarray(values, dtype=np.float32)
    low, high = np.asarray(stats["q01"]), np.asarray(stats["q99"])
    mask = np.asarray(stats["mask"], dtype=bool)
    result = np.zeros_like(data)
    result[..., mask] = np.clip(2 * (data[..., mask] - low[mask]) / (high[mask] - low[mask]) - 1, -1, 1)
    return result


def unnormalize(values: np.ndarray, stats: dict) -> np.ndarray:
    data = np.asarray(values, dtype=np.float32)
    low, high = np.asarray(stats["q01"]), np.asarray(stats["q99"])
    mask = np.asarray(stats["mask"], dtype=bool)
    result = np.zeros_like(data)
    result[..., mask] = 0.5 * (data[..., mask] + 1) * (high[mask] - low[mask]) + low[mask]
    result[..., ~mask] = low[~mask]
    return result
```

`build_trainable_components` must instantiate `L1RegressionActionHead(input_dim=vla.llm_dim, hidden_dim=vla.llm_dim, action_dim=7)` and `ProprioProjector(llm_dim=vla.llm_dim, proprio_dim=8)`, freeze every VLA parameter, and return only the two trainable modules.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest tools/tests/test_robocasa_oft_model.py -q`

Expected: `2 passed`.

- [ ] **Step 5: Record model-contract verification**

Run: `python -m pytest tools/tests/test_robocasa_oft_model.py -q > outputs/midterm_testing_vla82/oft_model_contract_test.txt`

Expected artifact contains `2 passed`.

---

### Task 4: One-batch memory smoke and continuous-head training

**Files:**
- Create: `tools/train_robocasa_oft_head.py`
- Create: `tools/tests/test_train_robocasa_oft_head.py`
- Create at runtime: `models/openvla-oft-robocasa365-head-r1/`

**Interfaces:**
- Consumes: Task 2 cache and Task 3 adapters.
- Produces: `action_head.pt`, `proprio_projector.pt`, `robocasa_oft_stats.json`, `training_report.json`, and `memory_smoke.json`.

- [ ] **Step 1: Write failing metric and early-stop tests**

```python
from tools.train_robocasa_oft_head import EarlyStop, improvement


def test_improvement_requires_lower_validation_loss():
    assert improvement(0.19, 0.20, minimum=0.005)
    assert not improvement(0.198, 0.20, minimum=0.005)


def test_early_stop_counts_non_improving_epochs():
    stop = EarlyStop(patience=2, minimum=0.005)
    assert not stop.update(0.20)
    assert not stop.update(0.198)
    assert stop.update(0.197)
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tools/tests/test_train_robocasa_oft_head.py -q`

Expected: collection fails because the trainer does not exist.

- [ ] **Step 3: Implement the trainer**

The trainer must load the OFT checkpoint in bfloat16, set `requires_grad=False` on all VLA parameters, place only the action head and proprio projector in training mode, compute normalized target chunks, call the official multimodal forward path with two images and normalized proprioception, minimize L1 loss, and save the lowest-validation-loss checkpoint. `--smoke-batches 1` must execute one real backward/optimizer step and report `torch.cuda.max_memory_allocated()`.

```python
def improvement(value: float, best: float, minimum: float) -> bool:
    return value <= best - minimum


@dataclass
class EarlyStop:
    patience: int
    minimum: float
    best: float = float("inf")
    misses: int = 0

    def update(self, value: float) -> bool:
        if improvement(value, self.best, self.minimum):
            self.best, self.misses = value, 0
        else:
            self.misses += 1
        return self.misses >= self.patience
```

- [ ] **Step 4: Verify tests and memory gate**

Run: `python -m pytest tools/tests/test_train_robocasa_oft_head.py -q`

Expected: `2 passed`.

Run: `C:/Users/sjtu101/miniconda3/envs/openvla-libero/python.exe -m tools.train_robocasa_oft_head --cache datasets/vla82_robocasa365_oft/cache_manifest.json --output models/openvla-oft-robocasa365-head-r1 --smoke-batches 1`

Expected: one optimizer step completes and `peak_allocated_gb < 23.5` in `memory_smoke.json`.

- [ ] **Step 5: Train to validation early stop**

Run: `C:/Users/sjtu101/miniconda3/envs/openvla-libero/python.exe -m tools.train_robocasa_oft_head --cache datasets/vla82_robocasa365_oft/cache_manifest.json --output models/openvla-oft-robocasa365-head-r1 --epochs 5 --patience 2`

Expected: validation L1 is below the unadapted head baseline, checkpoints exist, and the report records finite non-zero prediction variance.

---

### Task 5: Persistent OFT inference and pure rollout

**Files:**
- Create: `tools/robocasa_oft_tcp_server.py`
- Create: `tools/robocasa_oft_rollout.py`
- Create: `tools/tests/test_robocasa_oft_rollout.py`
- Reuse: `tools/openvla_tcp_protocol.py`
- Reuse: `tools/openvla_simulator_adapter.py`

**Interfaces:**
- Server request: primary image path, wrist image path, 8-D proprioception, instruction, relation key.
- Server response: finite 8x7 raw action chunk.
- Rollout output: report JSON, MP4, first/last frames, trajectory, official success predicate, and disclosure flags.

- [ ] **Step 1: Write failing chunk-policy tests**

```python
import numpy as np
from tools.robocasa_oft_rollout import validate_action_chunk


def test_validate_action_chunk_accepts_finite_8_by_7():
    result = validate_action_chunk(np.zeros((8, 7), dtype=np.float32))
    assert result.shape == (8, 7)


def test_validate_action_chunk_rejects_non_finite_values():
    chunk = np.zeros((8, 7), dtype=np.float32)
    chunk[0, 0] = np.nan
    try:
        validate_action_chunk(chunk)
    except ValueError as error:
        assert "finite" in str(error)
    else:
        raise AssertionError("non-finite chunk was accepted")
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tools/tests/test_robocasa_oft_rollout.py -q`

Expected: collection fails because the rollout module does not exist.

- [ ] **Step 3: Implement validation, server, and rollout**

```python
def validate_action_chunk(action_chunk: np.ndarray) -> np.ndarray:
    values = np.asarray(action_chunk, dtype=np.float32)
    if values.shape != (8, 7):
        raise ValueError(f"action chunk must have shape (8, 7), got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("action chunk must be finite")
    return values
```

The rollout must request a new chunk after executing at most 8 actions, map each 7-D row with `to_robocasa_action`, prohibit oracle replacement, permit simulator state only in `_check_success`, and store every predicted and executed action.

- [ ] **Step 4: Verify unit and service smoke tests**

Run: `python -m pytest tools/tests/test_robocasa_oft_rollout.py -q`

Expected: `2 passed`.

Run the server with the R1 continuous checkpoint and request chunks for at least eight distinct cached validation observations.

Expected: all responses are finite 8x7 arrays and the across-observation motion variance is greater than zero.

- [ ] **Step 5: Run the representative gate**

Run: `C:/Users/sjtu101/miniconda3/envs/robocasa/python.exe -m tools.robocasa_oft_rollout --selection-id VLA82-014 --max-decisions 300 --output outputs/midterm_testing_vla82/oft_r1_representative`

Expected: official `success=true`, `pure_autonomous_vla=true`, `oracle_or_scripted_action_used=false`, and complete image/video evidence. On failure, inspect this one trajectory before any expansion.

---

### Task 6: Eight-category and 60-object gated evaluation

**Files:**
- Create: `tools/run_robocasa_oft_midterm_matrix.py`
- Create: `tools/tests/test_robocasa_oft_midterm_matrix.py`
- Create at runtime: `outputs/midterm_testing_vla82/oft_final/`

**Interfaces:**
- Consumes: successful Task 5 representative model/service and `simulator_mapping_plan.json`.
- Produces: category manifest, 60-object manifest, evidence index, and summary images.

- [ ] **Step 1: Write failing manifest-gate tests**

```python
from tools.run_robocasa_oft_midterm_matrix import can_expand, summarize


def test_matrix_expansion_requires_every_prior_success():
    assert can_expand([{"success": True}, {"success": True}])
    assert not can_expand([{"success": True}, {"success": False}])


def test_summary_never_counts_execution_as_success():
    report = summarize([{"status": "TASK_FAILED", "success": False}])
    assert report["completed_count"] == 1
    assert report["success_count"] == 0
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tools/tests/test_robocasa_oft_midterm_matrix.py -q`

Expected: collection fails because the matrix runner does not exist.

- [ ] **Step 3: Implement gated orchestration and summary**

```python
def can_expand(results: list[dict]) -> bool:
    return bool(results) and all(item.get("success") is True for item in results)


def summarize(results: list[dict]) -> dict:
    return {
        "completed_count": len(results),
        "success_count": sum(item.get("success") is True for item in results),
        "failure_count": sum(item.get("success") is not True for item in results),
        "all_successful": can_expand(results),
        "results": results,
    }
```

The runner must execute the fixed 8 category representatives first. It may execute all 60 rows only when all 8 pass. Every proxy row must retain `proxy_disclosed=true`; proxy success must not be presented as native-object recognition.

- [ ] **Step 4: Verify GREEN and run eight-category gate**

Run: `python -m pytest tools/tests/test_robocasa_oft_midterm_matrix.py -q`

Expected: `2 passed`.

Run: `python -m tools.run_robocasa_oft_midterm_matrix --stage categories --output outputs/midterm_testing_vla82/oft_final`

Expected: 8/8 official predicate successes before matrix expansion.

- [ ] **Step 5: Run matrix and evidence audit**

Run: `python -m tools.run_robocasa_oft_midterm_matrix --stage matrix --output outputs/midterm_testing_vla82/oft_final`

Expected: 60 completed rows, per-row reports and videos, accurate success/failure counts, SHA-256 evidence index, and no undisclosed proxy claims.

---

## Plan Self-Review

- Spec coverage: data, memory, learning, representative, category, matrix, evidence, and purity gates are assigned to Tasks 1–6.
- Gate mapping: Data gate = Tasks 1–2; Memory gate = Task 4 smoke; Learning gate = Task 4 validation and Task 5 variance probe; representative gate = Task 5; 8 category gate and 60-object matrix gate = Task 6.
- Type consistency: all model-facing actions are 8x7 chunks; all proprioception values are 8-D; source actions remain 12-D only at the dataset boundary.
- Scope: the plan adapts only continuous components and does not add full-backbone fine-tuning or expert-controlled acceptance.
- Repository constraint: verification artifacts replace unavailable Git commits because `C:/OpenVLA-Simulator` is not a Git repository.
