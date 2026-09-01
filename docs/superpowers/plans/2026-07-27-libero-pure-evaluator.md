# LIBERO Pure-VLA Evaluator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run selected official LIBERO-Goal tasks through the downloaded OpenVLA checkpoint without importing training-only OpenVLA dependencies.

**Architecture:** A new standalone runner will reproduce the official evaluator's prompt, image preprocessing, action un-normalization, gripper transform, settling steps, and simulator success condition. It will import LIBERO directly and use Hugging Face AutoClasses for the downloaded checkpoint, avoiding the upstream utility's unused RLDS training imports.

**Tech Stack:** Python 3.10, PyTorch 2.2.2 CUDA, Transformers 4.40.1, TensorFlow 2.15.1, LIBERO, MuJoCo.

## Global Constraints

- Evaluate direct OpenVLA predictions only; do not add expert recovery or servo control.
- Keep the existing RoboCasa hybrid results separate from this pure-VLA report.
- Run a one-episode smoke rollout before any ten-episode task evaluation.
- Use the local checkpoint at `models/openvla-7b-finetuned-libero-goal` and `libero_goal` action normalization.

---

### Task 1: Standalone official-protocol helpers

**Files:**
- Create: `tools/openvla_libero_goal_pure_eval.py`
- Test: `tests/test_openvla_libero_goal_pure_eval.py`

**Interfaces:**
- Produces: `build_openvla_prompt(task: str) -> str`
- Produces: `parse_task_ids(value: str) -> list[int]`
- Produces: `run_subset(model_dir: Path, task_ids: list[int], trials_per_task: int, seed: int, output_dir: Path) -> dict[str, Any]`

- [ ] **Step 1: Write the failing test**

```python
from tools.openvla_libero_goal_pure_eval import build_openvla_prompt


def test_build_openvla_prompt_uses_official_instruction_template():
    assert build_openvla_prompt("Put the bowl on the stove") == (
        "In: What action should the robot take to put the bowl on the stove?\\nOut:"
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/test_openvla_libero_goal_pure_eval.py`

Expected: `ModuleNotFoundError: No module named 'tools.openvla_libero_goal_pure_eval'`.

- [ ] **Step 3: Write minimal implementation**

```python
def build_openvla_prompt(task: str) -> str:
    return f"In: What action should the robot take to {task.lower()}?\\nOut:"
```

Implement the direct LIBERO environment construction and exact official image/action transformations in the same file. Write a JSON report recording task ID, instruction, episode result, decision count, inference mean, and any exception.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/test_openvla_libero_goal_pure_eval.py`

Expected: `1 passed`.

### Task 2: One-episode pure-VLA smoke

**Files:**
- Create: `outputs/experiment_4_2_5/libero_goal_pure_smoke/report.json`

- [ ] **Step 1: Execute task 1 for one fixed initial state**

Run: `python -m tools.openvla_libero_goal_pure_eval --model-dir models/openvla-7b-finetuned-libero-goal --task-ids 1 --trials-per-task 1 --seed 7 --output-dir outputs/experiment_4_2_5/libero_goal_pure_smoke`

- [ ] **Step 2: Verify report fields**

Confirm `evaluation_kind` is `official_libero_goal_pure_vla`, `uses_expert_recovery` is false, and exactly one episode has a concrete success boolean.

### Task 3: Bounded formal evaluation

**Files:**
- Create: `outputs/experiment_4_2_5/libero_goal_pure_four_tasks/report.json`

- [ ] **Step 1: Run task IDs 1, 2, 5, and 7 for ten fixed initial states each**

Run: `python -m tools.openvla_libero_goal_pure_eval --model-dir models/openvla-7b-finetuned-libero-goal --task-ids 1,2,5,7 --trials-per-task 10 --seed 7 --output-dir outputs/experiment_4_2_5/libero_goal_pure_four_tasks`

- [ ] **Step 2: Verify evidence**

Confirm 40 episodes, four task summaries, no recovery field, and success rates computed from the recorded episode booleans.
