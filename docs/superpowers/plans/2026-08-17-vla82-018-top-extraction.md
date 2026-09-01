# VLA82-018 Top Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace VLA82-018's unreachable horizontal-side path with a fixed-base top grasp, cabinet extraction, counter transfer, and strict physical verification.

**Architecture:** Reuse the existing cabinet-front, yaw-alignment, native-pad approach, secure, lift, cabinet-extract, transit, release, and settle states. Add only an object-specific upper-body grasp affordance and routing helpers; do not create a second parallel pick/place controller.

**Tech Stack:** Python 3.11, NumPy, RoboCasa/RoboSuite, MuJoCo, `unittest`, `pytest`, existing VLA82 collection and strict-video replay tools.

## Global Constraints

- Keep the mobile base fixed during grasp and cabinet extraction.
- Do not modify object pose, target pose, mass, dimensions, friction, or success predicate.
- Require native grasp and opposing native-pad contact before lift.
- Freeze the wrist target during close and recover immediately if hold is lost.
- Count the 3 completed horizontal-side attempts toward the user's 5-attempt cap; only 2 complete VLA82-018 attempts remain.
- Do not export or index a video unless JSON reports `status=PASS`, `predicate_success=true`, and `errors=[]`.
- The workspace has no `.git` directory; use explicit test checkpoints instead of commits.

---

### Task 1: Route VLA82-018 through the existing top-grasp pipeline

**Files:**
- Modify: `tools/tests/test_vla82_source_spawn.py`
- Modify: `tools/vla82_full_sim/expert.py:1770-1805`
- Modify: `tools/vla82_full_sim/expert.py:5710-5885`

**Interfaces:**
- Produces: `pick_place_grasp_strategy("VLA82-018") == "top_extraction"`, `pick_place_post_cabinet_stage_state("VLA82-018") == "align_yaw"`, `pick_place_post_yaw_alignment_state("VLA82-018") == "approach"`, and an upper-body target from `pick_place_grasp_target(...)`.

- [ ] **Step 1: Replace obsolete VLA82-018 horizontal-side expectations with failing top-extraction tests**

```python
def test_box_drink_selects_fixed_base_top_extraction(self):
    self.assertEqual(expert.pick_place_grasp_strategy("VLA82-018"), "top_extraction")
    self.assertEqual(expert.pick_place_post_cabinet_stage_state("VLA82-018"), "align_yaw")
    self.assertEqual(expert.pick_place_post_yaw_alignment_state("VLA82-018"), "approach")
    self.assertEqual(expert.pick_place_retry_grasp_state("VLA82-018"), "approach")

def test_box_drink_grasp_target_is_in_upper_body_band(self):
    target = expert.pick_place_grasp_target(
        "VLA82-018",
        object_center=np.array((.20, -4.20, 1.49)),
        object_rotation=np.eye(3),
    )
    np.testing.assert_allclose(target, np.array((.20, -4.20, 1.52)))
```

- [ ] **Step 2: Run the two tests and verify RED**

Expected: existing code returns `horizontal_side`, routes into `side_orient`, and returns the object center instead of the upper grasp band.

- [ ] **Step 3: Implement the minimum routing and target changes**

Return `top_extraction` for VLA82-018. Route cabinet-front completion to `align_yaw`, route completed yaw alignment to `approach`, and route failed close attempts back to `approach`. In `pick_place_grasp_target`, add `object_rotation[:, 2] * .030` for VLA82-018.

- [ ] **Step 4: Run the two tests and verify GREEN**

- [ ] **Step 5: Run all source-spawn tests**

```powershell
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe -m unittest tools.tests.test_vla82_source_spawn
```

Expected: every test passes.

---

### Task 2: Enforce top-grasp close and recovery gates

**Files:**
- Modify: `tools/tests/test_vla82_source_spawn.py`
- Modify: `tools/vla82_full_sim/expert.py:3180-3305`
- Modify: `tools/vla82_full_sim/expert.py:5850-5905`

**Interfaces:**
- Consumes: native `raw_grasp`, `two_pad_contact`, pad-center target, and existing secure/lift recovery functions.
- Produces: VLA82-018 may enter `secure` only when native grasp and two-pad contact are both true; close target remains fixed.

- [ ] **Step 1: Add failing object-specific gate tests**

Verify VLA82-018 remains in `close` for raw-only or two-pad-only contact, enters `secure` only when both are true, and uses `eef` rather than a moving tracking target during close.

- [ ] **Step 2: Run tests and verify RED if the new `top_extraction` strategy bypasses the stronger gate**

- [ ] **Step 3: Generalize the stronger gate from `horizontal_side` to `{horizontal_side, top_extraction}`**

Keep compatibility behavior unchanged for already passing objects.

- [ ] **Step 4: Run targeted tests and the complete regression set**

```powershell
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe -m unittest tools.tests.test_vla82_source_spawn
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe -m pytest -q tools/tests/test_vla82_pure_closed_loop.py tools/tests/test_vla82_predicate_stability.py
```

Expected: all tests pass.

---

### Task 3: Run the fourth complete physical attempt and diagnose one gate

**Files:**
- Regenerate: `outputs/midterm_testing_vla82/full_simulation_objectwise_8x60/expert_demos/VLA82-018/episode-2000.json`
- Regenerate on success: matching `episode-2000.npz`

- [ ] **Step 1: Run one complete VLA82-018 simulation**

```powershell
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe tools\run_vla82_full_simulation.py collect-demos --episodes-per-object 1 --max-attempts-per-object 1 --selection-id VLA82-018
```

- [ ] **Step 2: Audit state counts and first failed physical gate**

Require yaw alignment, upper-body pad-centering, two-pad contact, native grasp, lift, cabinet extraction, transit, release, and settle. Reject shell-only contact or movement without grasp.

- [ ] **Step 3: If attempt 4 fails, make exactly one evidence-based correction**

Allowed corrections: upper grasp-band height, yaw sign, descent stop height, or close-position lock. Do not modify multiple variables, friction, scene state, or predicate.

---

### Task 4: Run the fifth and final VLA82-018 attempt

- [ ] **Step 1: Run all regressions after the single correction**

- [ ] **Step 2: Run one final complete simulation**

Use the same command as Task 3. This is the fifth total attempt and the hard stop.

- [ ] **Step 3: Branch on the verified result**

If strict PASS, proceed to Task 5. If it fails, stop VLA82-018, preserve the JSON diagnostic, remove it from strict-PASS consideration, and continue with a simpler untested object.

---

### Task 5: Export and index evidence only after strict PASS

**Files:**
- Create: `outputs/strict_pass_video_index_20260811/VLA82-018/episode-2000.strict-pass-top-extraction-720p.mp4`
- Update: strict-PASS workbook under `outputs/strict_pass_video_index_20260811/`

- [ ] **Step 1: Verify the strict source fields and NPZ existence**

- [ ] **Step 2: Replay at 1280×720**

```powershell
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe tools\replay_vla82_strict_pass_video.py --fps 24 --frame-stride 2 --width 1280 --height 720 outputs\midterm_testing_vla82\full_simulation_objectwise_8x60\expert_demos\VLA82-018\episode-2000.npz outputs\strict_pass_video_index_20260811\VLA82-018\episode-2000.strict-pass-top-extraction-720p.mp4
```

- [ ] **Step 3: Inspect start, grasp, lift, extraction, transit, release, and final frames**

Reject inverted, blocked, truncated, or visually ambiguous video.

- [ ] **Step 4: Update the workbook and run video audits**

```powershell
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe -m pytest -q tools/tests/test_audit_vla82_video_mapping.py tools/tests/test_replay_vla82_strict_pass_video.py
```
