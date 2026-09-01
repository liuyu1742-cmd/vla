# OpenVLA / OpenVLA-OFT Experiment Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a traceable, simulation-only reproduction of Section 4.6 using project-compatible OpenVLA evidence and the official OpenVLA-OFT implementation.

**Architecture:** Keep evidence collection, model-configuration profiling, result aggregation, and Chinese chapter generation separate. Official OpenVLA-OFT code and checkpoints remain third-party inputs; project wrappers normalize their logs into `outputs/experiment_4_6/` without changing policy actions or success predicates.

**Tech Stack:** Python 3.10, PyTorch 2.2.x CUDA, Transformers 4.40.1, PEFT, LIBERO/MuJoCo, pytest, JSON, Markdown.

## Global Constraints

- Use only simulation; exclude ALOHA and every real-robot result.
- Use effective batch size 16 through gradient accumulation where physical batch 16 does not fit.
- Preserve Full Fine-tuning, LoRA rank 32, Last Layer Only, and Frozen Vision as the four comparison modes.
- OpenVLA-OFT must use parallel decoding, action chunking, continuous actions, and L1 regression.
- Never add expert recovery, rule-based servo control, or relaxed success predicates.
- The acceptance floor for each requested success rate is the supplied value minus 5 percentage points.
- Label every result as `local_measured`, `project_historical`, or `official_reported`.

---

### Task 1: Evidence schema and aggregation

**Files:**
- Create: `tools/experiment_4_6_evidence.py`
- Test: `tests/test_experiment_4_6_evidence.py`

**Interfaces:**
- Produces: `EvidenceKind`, `EvidenceRecord`, `load_libero_report(path: Path, suite: str) -> EvidenceRecord`
- Produces: `mean_success(records: Sequence[EvidenceRecord]) -> float`

- [ ] **Step 1: Write failing tests**

```python
from pathlib import Path
import json

from tools.experiment_4_6_evidence import load_libero_report, mean_success


def test_load_libero_report_counts_episode_booleans(tmp_path: Path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"episodes": [{"success": True}, {"success": False}]}))
    row = load_libero_report(path, "libero_goal")
    assert row.success_rate == 50.0
    assert row.trials == 2
    assert row.evidence_kind == "local_measured"


def test_mean_success_is_trial_weighted():
    assert mean_success([
        {"success_rate": 100.0, "trials": 1},
        {"success_rate": 50.0, "trials": 3},
    ]) == 62.5
```

- [ ] **Step 2: Run tests and confirm import failure**

Run: `.venv\Scripts\python.exe -m pytest -q tests/test_experiment_4_6_evidence.py`

Expected: `ModuleNotFoundError: No module named 'tools.experiment_4_6_evidence'`.

- [ ] **Step 3: Implement immutable evidence records and strict parsing**

```python
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Mapping, Sequence
import json

EvidenceKind = Literal["local_measured", "project_historical", "official_reported"]


@dataclass(frozen=True)
class EvidenceRecord:
    name: str
    success_rate: float
    trials: int
    evidence_kind: EvidenceKind
    source: str

    def to_dict(self) -> dict:
        return asdict(self)


def load_libero_report(path: Path, suite: str) -> EvidenceRecord:
    payload = json.loads(path.read_text(encoding="utf-8"))
    episodes = payload["episodes"]
    if not episodes or any(type(row.get("success")) is not bool for row in episodes):
        raise ValueError("report must contain concrete episode success booleans")
    successes = sum(row["success"] for row in episodes)
    return EvidenceRecord(suite, 100.0 * successes / len(episodes), len(episodes), "local_measured", str(path))


def mean_success(records: Sequence[EvidenceRecord | Mapping]) -> float:
    total = sum(row.trials if isinstance(row, EvidenceRecord) else row["trials"] for row in records)
    wins = sum(
        (row.success_rate if isinstance(row, EvidenceRecord) else row["success_rate"])
        * (row.trials if isinstance(row, EvidenceRecord) else row["trials"])
        for row in records
    )
    return wins / total
```

- [ ] **Step 4: Run the focused tests**

Run: `.venv\Scripts\python.exe -m pytest -q tests/test_experiment_4_6_evidence.py`

Expected: `2 passed`.

### Task 2: Fine-tuning mode parameter and memory profiler

**Files:**
- Create: `tools/profile_openvla_finetune_modes.py`
- Test: `tests/test_profile_openvla_finetune_modes.py`

**Interfaces:**
- Produces: `classify_parameter(name: str, mode: str) -> bool`
- Produces: `summarize_mode(named_parameters, mode: str) -> dict`
- CLI writes: `outputs/experiment_4_6/finetune_modes/profile.json`

- [ ] **Step 1: Write tests for the four exact trainability policies**

```python
from tools.profile_openvla_finetune_modes import classify_parameter


def test_full_mode_trains_every_parameter():
    assert classify_parameter("vision_backbone.layer.weight", "full")


def test_frozen_vision_excludes_vision_parameters():
    assert not classify_parameter("vision_backbone.layer.weight", "frozen_vision")
    assert classify_parameter("language_model.layers.0.weight", "frozen_vision")


def test_last_layer_only_is_narrow():
    assert classify_parameter("language_model.layers.31.weight", "last_layer_only")
    assert classify_parameter("language_model.lm_head.weight", "last_layer_only")
    assert not classify_parameter("language_model.layers.30.weight", "last_layer_only")
```

- [ ] **Step 2: Run tests and verify failure**

Run: `.venv\Scripts\python.exe -m pytest -q tests/test_profile_openvla_finetune_modes.py`

Expected: import failure.

- [ ] **Step 3: Implement mode classification, parameter counts, CUDA peak recording, and OOM capture**

The CLI must load `models/openvla-7b` once per subprocess, apply the selected `requires_grad` policy, inject PEFT LoRA rank 32 for `lora_r32`, execute one forward/backward optimizer step with physical batch 1 and accumulation 16, and write:

```json
{
  "mode": "lora_r32",
  "effective_batch_size": 16,
  "physical_batch_size": 1,
  "gradient_accumulation_steps": 16,
  "trainable_parameters": 0,
  "total_parameters": 0,
  "peak_vram_gb": null,
  "status": "measured|oom|unsupported",
  "error": null
}
```

OOM must be captured after `torch.cuda.reset_peak_memory_stats()` and must not be converted to a projected success result.

- [ ] **Step 4: Run unit tests**

Run: `.venv\Scripts\python.exe -m pytest -q tests/test_profile_openvla_finetune_modes.py`

Expected: all tests pass.

- [ ] **Step 5: Run each mode in an isolated subprocess**

Run:

```powershell
.venv\Scripts\python.exe -m tools.profile_openvla_finetune_modes --model-dir models/openvla-7b --modes full,lora_r32,last_layer_only,frozen_vision --effective-batch-size 16 --output-dir outputs/experiment_4_6/finetune_modes
```

Expected: one JSON record per mode plus `profile.json`; OOM is an accepted measured outcome for modes exceeding 24 GB.

### Task 3: Official OpenVLA-OFT source and environment validation

**Files:**
- External source: `third_party/openvla-oft/`
- Create: `tools/check_openvla_oft_environment.py`
- Test: `tests/test_check_openvla_oft_environment.py`

**Interfaces:**
- Produces: `required_checkpoint(task_suite_name: str) -> str`
- CLI writes: `outputs/experiment_4_6/environment.json`

- [ ] **Step 1: Test the four official suite/checkpoint mappings**

```python
from tools.check_openvla_oft_environment import required_checkpoint


def test_official_suite_checkpoint_mapping():
    assert required_checkpoint("libero_spatial").endswith("libero-spatial")
    assert required_checkpoint("libero_object").endswith("libero-object")
    assert required_checkpoint("libero_goal").endswith("libero-goal")
    assert required_checkpoint("libero_10").endswith("libero-10")
```

- [ ] **Step 2: Clone the official source at a recorded commit**

Run: `git clone https://github.com/moojink/openvla-oft.git third_party/openvla-oft`

Expected: `third_party/openvla-oft/LIBERO.md` exists. Record `git -C third_party/openvla-oft rev-parse HEAD`.

- [ ] **Step 3: Implement the environment checker**

The checker must record Python, PyTorch, CUDA, GPU, Transformers, LIBERO import state, source commit, checkpoint cache state, and the official method flags:

```python
OFFICIAL_FLAGS = {
    "use_l1_regression": True,
    "use_diffusion": False,
    "use_film": False,
    "num_images_in_input": 2,
    "use_proprio": True,
    "center_crop": True,
    "num_open_loop_steps": 8,
}
```

- [ ] **Step 4: Verify the checker**

Run: `C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe -m tools.check_openvla_oft_environment --source-dir third_party/openvla-oft --output outputs/experiment_4_6/environment.json`

Expected: JSON identifies RTX 3090 and reports every missing dependency explicitly.

### Task 4: Official OFT LIBERO evaluation wrapper

**Files:**
- Create: `tools/run_openvla_oft_libero_4_6.py`
- Test: `tests/test_run_openvla_oft_libero_4_6.py`

**Interfaces:**
- Consumes: official `run_libero_eval.py` and suite/checkpoint mapping from Task 3
- Produces: `build_command(...) -> list[str]`
- Writes: `outputs/experiment_4_6/oft/<suite>/<run_id>/`

- [ ] **Step 1: Test that commands preserve official OFT settings**

```python
from pathlib import Path
from tools.run_openvla_oft_libero_4_6 import build_command


def test_build_command_uses_pure_official_policy():
    cmd = build_command(Path("python.exe"), Path("third_party/openvla-oft"), "libero_goal", 1, 7)
    joined = " ".join(cmd)
    assert "run_libero_eval.py" in joined
    assert "--task_suite_name libero_goal" in joined
    assert "--num_trials_per_task 1" in joined
    assert "--center_crop True" in joined
    assert "expert" not in joined.lower()
```

- [ ] **Step 2: Implement subprocess logging and immutable run manifests**

Each run manifest must include command, start/end time, return code, stdout/stderr paths, source commit, checkpoint ID, GPU, seed, requested trials, and parsed task success counters.

- [ ] **Step 3: Run one smoke episode per suite**

Run:

```powershell
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe -m tools.run_openvla_oft_libero_4_6 --source-dir third_party/openvla-oft --suites libero_spatial,libero_object,libero_goal,libero_10 --trials-per-task 1 --seed 7 --output-dir outputs/experiment_4_6/oft_smoke
```

Expected: all four processes return zero and produce concrete task counters.

- [ ] **Step 4: Run bounded formal evaluations**

Run the same wrapper with `--trials-per-task 10` and seeds `7,17,27`. Preserve every failed attempt. If a suite is below its acceptance floor, verify checkpoint, center crop, action normalization key, and open-loop steps before rerunning.

Expected: 100 trials per suite per seed, unless runtime requires the documented one-trial-per-task fallback.

### Task 5: Result synthesis and chapter generation

**Files:**
- Create: `tools/generate_experiment_4_6_report.py`
- Test: `tests/test_generate_experiment_4_6_report.py`
- Create: `outputs/experiment_4_6/summary.json`
- Create: `outputs/experiment_4_6/4.6_实验验证_复现实验.md`

**Interfaces:**
- Consumes: profiler JSON, local LIBERO report, OFT manifests, and a versioned official reference-results constant
- Produces: `build_summary(...) -> dict`
- Produces: `render_chapter(summary: dict) -> str`

- [ ] **Step 1: Test provenance labels and suite naming**

```python
from tools.generate_experiment_4_6_report import render_chapter


def test_report_uses_libero_long_and_provenance():
    text = render_chapter({
        "finetune_rows": [],
        "oft_rows": [{"suite": "LIBERO-Long", "oft": 94.5, "oft_source": "official_reported"}],
        "limitations": [],
    })
    assert "LIBERO-Long" in text
    assert "LIBERO-Navigation" not in text
    assert "官方报告" in text
```

- [ ] **Step 2: Implement strict synthesis**

The generator must reject rows with no provenance, compute means from unrounded values, include trial counts for local results, and never substitute RoboCasa hybrid-control runs for pure VLA or OFT results.

- [ ] **Step 3: Generate final evidence**

Run:

```powershell
.venv\Scripts\python.exe -m tools.generate_experiment_4_6_report --experiment-dir outputs/experiment_4_6 --output outputs/experiment_4_6/4.6_实验验证_复现实验.md
```

Expected: Markdown contains Sections 4.6, 4.6.1, and 4.6.2; Tables 4-7 and 4-8; experimental setup; provenance notes; limitations; and only evidence-backed numbers.

- [ ] **Step 4: Run verification**

Run:

```powershell
.venv\Scripts\python.exe -m pytest -q tests/test_experiment_4_6_evidence.py tests/test_profile_openvla_finetune_modes.py tests/test_check_openvla_oft_environment.py tests/test_run_openvla_oft_libero_4_6.py tests/test_generate_experiment_4_6_report.py
```

Expected: all tests pass.

Run: `.venv\Scripts\python.exe -m compileall -q tools tests`

Expected: exit code 0.

