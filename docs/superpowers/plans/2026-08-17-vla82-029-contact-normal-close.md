# VLA82-029 Contact-Normal Close Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve continuous physical contact while VLA82-029 pushes the microwave door along its closing hinge tangent.

**Architecture:** Add pure helpers for contact-normal filtering and bounded tangent/normal direction blending. Feed the deepest live target-door contact into the microwave-only branch of `DrawerClosePrimitive`; use the blended direction during contact and a 6 mm last-normal reseat step after brief contact loss.

**Tech Stack:** Python 3.11, NumPy, `unittest`, `pytest`, RoboCasa, robosuite, MuJoCo.

## Global Constraints

- Use only real MuJoCo contacts between Panda gripper/hand collision geoms and geoms bound to the selected microwave door joint.
- Do not write joint state, teleport bodies, disable collision, alter the strict close threshold, or edit result status manually.
- Filter normals with `normalize(0.65 * previous + 0.35 * current)` after hemisphere alignment.
- Use `minimum_inward=0.08`, `max_normal_bias=0.45`, and a contact-loss reseat distance of `0.006 m`.
- After 20 consecutive no-contact frames, stop using the stale normal and use the current door geometry's short-range seat target without returning to the wrong side.
- Enable this behavior only for the microwave free-edge bypass branch; preserve all other fixture-close behavior.
- Export video and update the strict-PASS index only after fresh physical, predicate, trace, and visual verification.
- The workspace is not a Git repository; use atomic `apply_patch` edits and passing-test checkpoints instead of commits.

---

### Task 1: Pure contact-normal control helpers

**Files:**
- Modify: `tools/vla82_full_sim/expert.py:493-520,765-890`
- Test: `tools/tests/test_vla82_source_spawn.py:1170-1430`

**Interfaces:**
- Consumes: previous/current normals, closing tangent, minimum inward component, and maximum normal bias.
- Produces: `fixture_close_filtered_contact_normal(previous, current) -> np.ndarray` and `fixture_close_contact_preserving_direction(tangent, contact_normal, minimum_inward=0.08, max_normal_bias=0.45) -> tuple[np.ndarray, float]`.

- [ ] **Step 1: Write failing tests for filtering and direction blending**

Add tests with these assertions:

```python
def test_fixture_close_contact_normal_aligns_and_smooths(self):
    from tools.vla82_full_sim import expert
    filtered = expert.fixture_close_filtered_contact_normal(
        previous=(1., 0., 0.), current=(-.8, -.6, 0.),
    )
    self.assertGreater(filtered[0], .9)
    self.assertGreater(filtered[1], 0.)
    self.assertAlmostEqual(float(np.linalg.norm(filtered)), 1., places=6)

def test_fixture_close_direction_adds_only_bounded_inward_bias(self):
    from tools.vla82_full_sim import expert
    tangent = np.array((.05, -.99875, 0.))
    normal = np.array((.95, .31225, 0.))
    direction, bias = expert.fixture_close_contact_preserving_direction(
        tangent=tangent, contact_normal=normal,
        minimum_inward=.08, max_normal_bias=.45,
    )
    self.assertGreaterEqual(float(np.dot(direction, normal)), .079)
    self.assertGreater(float(np.dot(direction, tangent)), .8)
    self.assertGreater(bias, 0.)
    self.assertLessEqual(bias, .45)

def test_fixture_close_direction_does_not_bias_an_inward_tangent(self):
    from tools.vla82_full_sim import expert
    direction, bias = expert.fixture_close_contact_preserving_direction(
        tangent=(1., 0., 0.), contact_normal=(1., 0., 0.),
    )
    np.testing.assert_allclose(direction, (1., 0., 0.), atol=1e-8)
    self.assertEqual(bias, 0.)
```

Add error assertions for zero-length current normal and zero-length tangent.

- [ ] **Step 2: Run the new tests and verify RED**

Run the exact new unittest methods. Expected: errors naming the two missing helper functions.

- [ ] **Step 3: Implement minimal pure helpers**

Implement hemisphere alignment, the specified 65/35 normalized filter, degenerate-vector validation, `bias = clip(minimum_inward - dot(tangent, normal), 0, max_normal_bias)`, and normalized `tangent + bias * normal`. Return the final direction and scalar bias.

- [ ] **Step 4: Run targeted tests and the full source-spawn suite**

Run:

```powershell
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe -m unittest tools.tests.test_vla82_source_spawn
```

Expected: all tests pass; the current baseline is 118 tests, so the new total must be at least 121.

---

### Task 2: Microwave controller integration and short reseat

**Files:**
- Modify: `tools/vla82_full_sim/expert.py:493-520,1200-1460`
- Test: `tools/tests/test_vla82_source_spawn.py`

**Interfaces:**
- Consumes: `selected_gripper_contact_details`, Task 1 helpers, existing `push_direction`, `contact_reference`, and microwave bypass state.
- Produces: live `raw_contact_normal`, `filtered_contact_normal`, `normal_bias`, `contact_preserving_direction`, `continuous_contact_steps`, and `contact_loss_reseat_steps` trace fields.

- [ ] **Step 1: Write failing tests for contact selection and reseat target**

Add pure helpers and wished-for tests:

```python
def test_fixture_close_selects_deepest_target_contact_normal(self):
    from tools.vla82_full_sim import expert
    normal = expert.fixture_close_deepest_contact_normal((
        {"normal_gripper_to_object": (1., 0., 0.), "distance": -.0001},
        {"normal_gripper_to_object": (0., 1., 0.), "distance": -.0006},
    ))
    np.testing.assert_allclose(normal, (0., 1., 0.))

def test_fixture_close_reseat_uses_small_last_normal_step(self):
    from tools.vla82_full_sim import expert
    target = expert.fixture_close_contact_reseat_target(
        contact_reference=(1., 2., 3.), last_contact_normal=(0., 1., 0.),
        distance=.006,
    )
    np.testing.assert_allclose(target, (1., 2.006, 3.))
```

Assert that empty contact details return `None`, and a degenerate reseat normal raises `ValueError`.

- [ ] **Step 2: Run these tests and verify RED**

Expected: errors because `fixture_close_deepest_contact_normal` and `fixture_close_contact_reseat_target` are missing.

- [ ] **Step 3: Implement the two pure helpers**

`fixture_close_deepest_contact_normal(details)` selects the row with the smallest `distance`, validates its three-vector, normalizes it, and returns `None` for no rows. `fixture_close_contact_reseat_target` returns `contact_reference + normalize(last_contact_normal) * distance`.

- [ ] **Step 4: Integrate live contact-normal state**

In the microwave branch of `DrawerClosePrimitive`:

1. Read `contact_details = selected_gripper_contact_details(raw, contact_geoms)` once per control cycle and reuse it for both control and trace.
2. When exact contact exists, select the deepest normal, update the 65/35 filtered normal, set `continuous_contact_steps += 1`, and reset `contact_loss_reseat_steps = 0`.
3. During `push_close`, replace the pure hinge tangent with the Task 1 blended direction and retain the existing `0.01 m` hinge step.
4. After a previously valid correct-side contact is lost, set `seat_contact` target to the current contact reference plus filtered normal times `0.006 m`; increment `contact_loss_reseat_steps` and reset `continuous_contact_steps`.
5. Once loss exceeds 20 frames, stop using the stale normal and use the live selected door geom target through the existing short-range seating path.
6. Keep the base fixed throughout bypass, push, and reseat.

- [ ] **Step 5: Record the diagnostic fields**

Add the exact raw/filtered normal arrays, original tangent, scalar bias, blended direction, continuous-contact counter, loss-reseat counter, and reused `contact_details` to each microwave trace row.

- [ ] **Step 6: Run compile and regression verification**

Run:

```powershell
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe -m compileall -q tools/vla82_full_sim/expert.py tools/tests/test_vla82_source_spawn.py
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe -m unittest tools.tests.test_vla82_source_spawn
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe -m pytest -q tools/tests/test_vla82_pure_closed_loop.py tools/tests/test_vla82_predicate_stability.py
```

Expected: source-spawn suite passes and the pure closed-loop/predicate suites report at least `5 passed`.

---

### Task 3: Fresh physical run and conditional evidence publication

**Files:**
- Regenerate: `outputs/midterm_testing_vla82/full_simulation_objectwise_8x60/expert_demos/VLA82-029/episode-2000.json`
- Regenerate on collection success: matching `episode-2000.npz`
- Create only on strict success: `outputs/strict_pass_video_index_20260811/VLA82-029/episode-2000.strict-pass-microwave-close-720p.mp4`
- Update only on strict success: existing strict-PASS workbook under `outputs/strict_pass_video_index_20260811/`

**Interfaces:**
- Consumes: the verified Task 2 controller.
- Produces: fresh physical evidence, then a video and workbook entry only if all strict conditions pass.

- [ ] **Step 1: Run one fresh VLA82-029 attempt**

```powershell
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe tools\run_vla82_full_simulation.py collect-demos --episodes-per-object 1 --max-attempts-per-object 1 --selection-id VLA82-029
```

- [ ] **Step 2: Audit the fresh trace**

Require ordered bypass stages, correct-side contact, a substantial increase beyond the previous maximum one-frame contact segment, a closing-directed blended vector, and hinge motion toward zero. A process exit without these properties is not success.

- [ ] **Step 3: Decide the next cause from evidence**

If contact continuity improves but hinge closure stalls, investigate contact point leverage and reachability. If contact remains one-frame despite correct logged blend vectors, stop coefficient tuning and return to a redesigned broad-panel contact path. Do not stack unrelated parameter changes.

- [ ] **Step 4: Export only after strict PASS**

After JSON reports `status: PASS`, `predicate_success: true`, no errors, continuous real contact, and stable closure, run:

```powershell
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe tools\replay_vla82_strict_pass_video.py --fps 24 --frame-stride 2 --width 1280 --height 720 outputs\midterm_testing_vla82\full_simulation_objectwise_8x60\expert_demos\VLA82-029\episode-2000.npz outputs\strict_pass_video_index_20260811\VLA82-029\episode-2000.strict-pass-microwave-close-720p.mp4
```

- [ ] **Step 5: Visually and programmatically verify publication**

Inspect upright, unobstructed start/bypass/contact/mid-close/final frames. Verify source hashes and workbook absolute path, then rerun all source-spawn, pure closed-loop, predicate-stability, replay, and video-mapping tests before reporting strict PASS.
