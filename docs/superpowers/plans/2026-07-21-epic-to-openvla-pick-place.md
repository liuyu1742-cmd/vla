# EPIC-KITCHENS to OpenVLA Pick-and-Place Plan

**Goal:** Produce an auditable human-video skill manifest that maps one EPIC pick-and-place demonstration to a RoboCasa household task.

### Task 1: Skill-manifest contract

- Create `tools/epic_skill_manifest.py` and `tests/test_epic_skill_manifest.py`.
- Test first: reject a phase without `verb`, `object`, or `target`.
- Implement JSON manifest validation and a CLI that accepts clip metadata only; it must not require loading all video frames.

### Task 2: Semantic mapping

- Create `tools/epic_to_robocasa_mapping.py` and `tests/test_epic_to_robocasa_mapping.py`.
- Test first: map `cup` pick-and-place only to the existing `PickPlaceCounterToCabinet` / `glass_cup` target; reject unknown objects.
- Produce an integrated report linking source clip, ordered phases, simulator task, and OpenVLA phase instruction.

### Task 3: First clip selection

- Read only the user-supplied EPIC clips/annotations needed to select one suitable pick-and-place clip.
- Write `outputs/epic_to_openvla_pick_place_manifest.json` and verify it passes the manifest contract.
