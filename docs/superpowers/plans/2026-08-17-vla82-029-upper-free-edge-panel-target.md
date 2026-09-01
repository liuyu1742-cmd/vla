# VLA82-029 Upper Free-Edge Panel Target Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the upper-panel target 65% of panel half-width toward the live free edge while preserving all safety margins and fixed-base post-raise control.

**Architecture:** Extend the existing pure upper-panel waypoint helper with hinge-aware horizontal targeting and reachability validation. Pass the live hinge anchor from `DrawerClosePrimitive`; no other controller architecture changes.

**Tech Stack:** Python 3.11, NumPy, `unittest`, `pytest`, RoboCasa, robosuite, MuJoCo.

## Global Constraints

- Use `horizontal_free_edge_fraction=.65`, minimum edge clearance `.075 m`, and maximum raise-to-precontact distance `.20 m`.
- Preserve vertical fraction `.45`, handle clearance `.07 m`, top clearance `.08 m`, and precontact clearance `.06 m`.
- Preserve the validated base-assist cap and lock the base after raise.
- Do not accept handle/frame contact, change strict predicates, write state, or increase the free-edge fraction after failure.
- Enable only for VLA82-029 microwave upper-panel targeting.
- Workspace is not a Git repository; use atomic patches and test checkpoints.

---

### Task 1: Hinge-aware free-edge target geometry

**Files:**
- Modify: `tools/vla82_full_sim/expert.py:850-1050`
- Test: `tools/tests/test_vla82_source_spawn.py:1490-1580`

- [ ] **Step 1: Write failing tests and update existing calls**

Pass `hinge_anchor=(.234, 0., 1.3)` to identity-panel tests and the rotated equivalent to rotation tests. Assert `free_edge_direction=(-1,0,0)`, horizontal offset magnitude `.65*.234`, face X near `-.1521`, edge clearance `.35*.234`, and raise-to-precontact distance below `.20`. Add an invalid case with fraction `.80`, whose remaining edge clearance is below `.075`, and expect `ValueError`.

- [ ] **Step 2: Verify RED**

Run the upper-panel waypoint tests. Expected: unexpected `hinge_anchor` keyword or missing new result keys.

- [ ] **Step 3: Extend the pure helper**

Add required `hinge_anchor`, optional fraction/minimum-edge/maximum-path parameters, identify the horizontal in-plane axis, choose the edge farther from the hinge, add the 65% horizontal offset to `face`, validate remaining edge clearance and `norm(precontact - raised) <= .20`, and return `free_edge_direction`, `horizontal_offset`, and `edge_clearance`.

- [ ] **Step 4: Verify targeted and full unit suites**

```powershell
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe -m unittest tools.tests.test_vla82_source_spawn
```

Expected: all tests pass; baseline is 131 tests.

---

### Task 2: Live hinge integration and trace

**Files:**
- Modify: `tools/vla82_full_sim/expert.py:1400-1750`

- [ ] **Step 1: Pass live hinge anchor**

At upper waypoint creation, pass `raw.sim.data.xanchor[joint_index]`. Keep all existing base-assist, vertical, contact, and state-machine logic unchanged.

- [ ] **Step 2: Record horizontal evidence**

Add returned free-edge direction, horizontal offset, and edge clearance to each trace row.

- [ ] **Step 3: Run regression verification**

```powershell
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe -m compileall -q tools/vla82_full_sim/expert.py tools/tests/test_vla82_source_spawn.py
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe -m unittest tools.tests.test_vla82_source_spawn
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe -m pytest -q tools/tests/test_vla82_pure_closed_loop.py tools/tests/test_vla82_predicate_stability.py
```

---

### Task 3: Final physical round for this path

- [ ] **Step 1: Run one fresh VLA82-029 attempt**

```powershell
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe tools\run_vla82_full_simulation.py collect-demos --episodes-per-object 1 --max-attempts-per-object 1 --selection-id VLA82-029
```

- [ ] **Step 2: Audit in order**

Verify raise completion, fixed base after raise, traverse distance entering `.025 m`, transition to `seat_upper_panel`, broad-panel contact continuity, and hinge motion toward zero.

- [ ] **Step 3: Apply the explicit stop condition**

If the 65% target remains unreachable, do not raise the fraction or continue retrying VLA82-029; record the final evidence and proceed to easier remaining object operations. Export and index video only if full strict PASS succeeds.
