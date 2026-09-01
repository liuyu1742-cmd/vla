# VLA82 Manual Teleoperation Launcher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a keyboard-driven VLA82 launcher that loads any authoritative VLA82 scene and saves a manual simulator episode.

**Architecture:** A new standalone module resolves a selection through the existing runtime spec and scene builders, then owns only CLI parsing, keyboard-to-public-action translation, rendering, and episode serialization. It reuses RoboSuite `Keyboard` and preserves the existing VLA82 physics scene and friction setup.

**Tech Stack:** Python 3.11, NumPy, RoboCasa, RoboSuite, MuJoCo, OpenCV, pytest.

## Global Constraints

- Support every identifier present in the compiled VLA82 registry, currently `VLA82-001` through `VLA82-060`.
- Use `make_environment` and public `environment.step` only; do not write MuJoCo state directly.
- Persist manual output as `manual_unverified`; do not claim strict PASS.
- Close the simulator and retain a non-empty capture after user interrupt.
- The GUI must run in the user’s visible Windows desktop session.

---

### Task 1: Create pure selection and metadata helpers

**Files:**
- Create: `tools/vla82_manual_teleop.py`
- Test: `tests/test_vla82_manual_teleop.py`

**Interfaces:**
- Produces: `resolve_selection(selection_id: str, specs: Sequence[OperationSpec]) -> OperationSpec`
- Produces: `build_episode_metadata(...) -> dict[str, object]`

- [ ] **Step 1: Write the failing tests**

```python
def test_resolve_selection_returns_requested_runtime_spec():
    spec = SimpleNamespace(selection_id="VLA82-017")
    assert teleop.resolve_selection("VLA82-017", [spec]) is spec

def test_resolve_selection_rejects_unknown_identifier():
    with pytest.raises(ValueError, match="unknown VLA82 selection"):
        teleop.resolve_selection("VLA82-999", [])

def test_episode_metadata_marks_capture_manual_and_unverified(tmp_path):
    metadata = teleop.build_episode_metadata(
        selection_id="VLA82-017", seed=820017, task_class="PickPlaceCounterToCabinet",
        camera="robot0_agentview_left", action_count=2, request=SimpleNamespace(
            manipulated_objects=("bottle",), source_fixture="counter",
            target_fixture="cabinet", target_relation="inside",
        ), fingerprint={"sha256": "scene-hash"}, friction_evidence={"material": "plastic"},
    )
    assert metadata["status"] == "manual_unverified"
    assert metadata["action_count"] == 2
```

- [ ] **Step 2: Run the tests and verify they fail because the module is absent**

Run: `python -m pytest tests/test_vla82_manual_teleop.py -q`

- [ ] **Step 3: Implement the smallest helpers**

```python
def resolve_selection(selection_id, specs):
    normalized = str(selection_id).strip().upper()
    for spec in specs:
        if spec.selection_id == normalized:
            return spec
    raise ValueError(f"unknown VLA82 selection: {selection_id}")
```

`build_episode_metadata` must return the specified status, request semantics, scene hash, friction evidence, selected camera, and action count.

- [ ] **Step 4: Run the helper tests and verify they pass**

Run: `python -m pytest tests/test_vla82_manual_teleop.py -q`

### Task 2: Implement public keyboard action collection and persistence

**Files:**
- Modify: `tools/vla82_manual_teleop.py`
- Modify: `tests/test_vla82_manual_teleop.py`

**Interfaces:**
- Produces: `save_episode(directory: Path, actions: Sequence[np.ndarray], metadata: Mapping[str, object], model_xml: str) -> Path`
- Produces: `run_manual_episode(...) -> Path`

- [ ] **Step 1: Write failing persistence tests**

```python
def test_save_episode_writes_native_actions_and_metadata(tmp_path):
    output = teleop.save_episode(
        tmp_path, [np.zeros(12, dtype=np.float32)], {"status": "manual_unverified"}, "<mujoco/>",
    )
    assert (output / "actions.npz").is_file()
    assert json.loads((output / "episode.json").read_text())["status"] == "manual_unverified"
    with np.load(output / "actions.npz") as saved:
        assert saved["actions"].shape == (1, 12)
```

- [ ] **Step 2: Run the persistence test and verify it fails**

Run: `python -m pytest tests/test_vla82_manual_teleop.py -q`

- [ ] **Step 3: Implement persistence and the interactive loop**

Use `robosuite.devices.Keyboard`, `device.input2action(mirror_actions=True)`, and the existing PandaOmron action-vector construction path. In base mode, preserve RoboSuite’s standard `base`, `base_mode`, torso, arm, and gripper channels. Render with the live MuJoCo viewer, optionally write camera frames using OpenCV, and save episode files in `finally` if at least one action was collected.

- [ ] **Step 4: Run persistence tests and Python compilation**

Run: `python -m pytest tests/test_vla82_manual_teleop.py -q`

Run: `python -m py_compile tools/vla82_manual_teleop.py`

### Task 3: Add CLI and verify executable surface

**Files:**
- Modify: `tools/vla82_manual_teleop.py`
- Modify: `tests/test_vla82_manual_teleop.py`

**Interfaces:**
- Produces: `main(argv: Sequence[str] | None = None) -> int`

- [ ] **Step 1: Write failing CLI parser tests**

```python
def test_parser_requires_selection_and_accepts_record_options():
    parser = teleop.build_parser()
    args = parser.parse_args(["--selection-id", "VLA82-017", "--record-video"])
    assert args.selection_id == "VLA82-017"
    assert args.record_video is True
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `python -m pytest tests/test_vla82_manual_teleop.py -q`

- [ ] **Step 3: Implement parser and CLI help**

Expose `--selection-id`, `--seed`, `--record-dir`, `--camera`, `--record-video`, `--max-steps`, `--pos-sensitivity`, and `--rot-sensitivity`. `main` must resolve the VLA82 scene using `_load_compiled_specs` and `_scene_for_spec`, then call the interactive session.

- [ ] **Step 4: Run all launcher checks**

Run: `python -m pytest tests/test_vla82_manual_teleop.py -q`

Run: `python tools/vla82_manual_teleop.py --help`

Run: `python -m py_compile tools/vla82_manual_teleop.py`

### Task 4: Document direct usage

**Files:**
- Modify: `docs/superpowers/specs/2026-08-18-vla82-manual-teleop-design.md`

- [ ] **Step 1: Add the verified Windows command**

```powershell
python tools\vla82_manual_teleop.py --selection-id VLA82-017 --record-video
```

- [ ] **Step 2: Re-run help command and retain its output as verification**

Run: `python tools/vla82_manual_teleop.py --help`
