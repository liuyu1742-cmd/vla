# Water-Cup OpenVLA Zero-Shot Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute and report one genuine OpenVLA-controlled RoboCasa water-cup episode.

**Architecture:** A dedicated runner creates `PickPlaceCounterToCabinet`, obtains `robot0_agentview_left_image`, predicts one seven-element Bridge action per step, maps it with `to_robocasa_action`, and writes an auditable JSON report. It never reads BridgeData V2.

**Tech Stack:** Python 3.10 `openvla` environment; Transformers; MuJoCo/RoboCasa; NumPy; JSON.

## Global Constraints

- Use actual RoboCasa camera observations, never the placeholder smoke-test image.
- Do not access `datasets/oxe/bridge_orig`.
- Bound the episode to 10 control steps, seed 0, and report simulator success separately from runner completion.
- Reject a model action whose length is not 7 before calling `env.step`.

---

### Task 1: Implement report and action validation helpers

**Files:**

- Create: `tools/run_openvla_robocasa_water_cup.py`
- Create: `tests/test_run_openvla_robocasa_water_cup.py`

**Interfaces:**

- Produces `validate_model_action(action: Sequence[float]) -> list[float]`.
- Produces `build_report(seed: int, instruction: str) -> dict[str, object]` with `steps`, `runner_completed`, and `simulator_success`.

- [ ] **Step 1: Write failing tests**

```python
def test_rejects_non_seven_dimensional_action():
    with self.assertRaisesRegex(ValueError, "7"):
        validate_model_action([0.0] * 6)

def test_report_separates_runner_and_simulator_status():
    report = build_report(0, "pick up the glass cup and place it in the cabinet")
    self.assertFalse(report["runner_completed"])
    self.assertIsNone(report["simulator_success"])
```

- [ ] **Step 2: Verify RED**

Run: `C:\\Users\\sjtu101\\miniconda3\\envs\\openvla\\python.exe -m unittest tests.test_run_openvla_robocasa_water_cup -v`

Expected: import failure because the runner module does not exist.

- [ ] **Step 3: Implement helpers**

```python
def validate_model_action(action):
    values = [float(value) for value in action]
    if len(values) != 7:
        raise ValueError(f"OpenVLA action must have 7 elements, got {len(values)}")
    return values
```

`build_report` initializes `seed`, `task`, `instruction`, `camera`, empty `steps`, `runner_completed=False`, `simulator_success=None`, and `error=None`.

- [ ] **Step 4: Verify GREEN**

Run the Task 1 command again. Expected: PASS.

### Task 2: Run the bounded real-camera zero-shot episode

**Files:**

- Modify: `tools/run_openvla_robocasa_water_cup.py`
- Modify: `tests/test_run_openvla_robocasa_water_cup.py`
- Create: `outputs/openvla_robocasa_water_cup_zeroshot.json`

**Interfaces:**

- Consumes `--seed`, `--max-steps`, and `--report`.
- Produces one report step containing `raw_action`, `robocasa_action`, `reward`, `terminated`, and `truncated`.

- [ ] **Step 1: Add a failing report-shape test**

```python
def test_step_report_keeps_raw_and_adapted_actions():
    step = make_step_record([0.0] * 7, to_robocasa_action([0.0] * 7), 0.0, False, False)
    self.assertEqual(step["raw_action"], [0.0] * 7)
    self.assertIn("action.end_effector_position", step["robocasa_action"])
```

- [ ] **Step 2: Verify RED**

Run the Task 1 command. Expected: failure for missing `make_step_record`.

- [ ] **Step 3: Implement runner**

Add local RoboCasa source to `sys.path`; create `gym.make("robocasa/PickPlaceCounterToCabinet", seed=seed)`; reset it; obtain `robot0_agentview_left_image`; convert it to RGB PIL; run the already-verified local OpenVLA model with prompt `pick up the glass cup and place it in the cabinet`; validate and map its action; execute `env.step`; append JSON-safe action groups and transition flags. Catch exceptions into `error`, always close the environment, then write the report.

- [ ] **Step 4: Verify GREEN and real episode**

Run the Task 1 test command. Expected: PASS.

Run: `C:\\Users\\sjtu101\\miniconda3\\envs\\openvla\\python.exe tools\\run_openvla_robocasa_water_cup.py --seed 0 --max-steps 10 --report outputs\\openvla_robocasa_water_cup_zeroshot.json`

Expected: a report with at least one camera-derived action, or a captured simulator/model error; no BridgeData access.
