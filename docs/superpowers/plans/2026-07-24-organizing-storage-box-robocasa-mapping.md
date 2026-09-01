# Organizing Storage Box RoboCasa Mapping Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and verify an auditable RoboCasa mapping for `organizing::storage_box` using the native `tupperware` object group.

**Architecture:** Extend the existing formal relation registry without changing the stable toy mapping. Add a small real-environment audit command that writes a mapping report but never claims execution success.

**Tech Stack:** Python 3.11, unittest, Gymnasium, RoboCasa, robosuite, MuJoCo.

## Global Constraints

- Preserve `organizing::toy` behavior.
- Use `tupperware` only as a disclosed semantic asset proxy.
- Do not use held-out seeds `201`, `202`, or `203` for reset tuning, training, or DAgger.
- Environment reset evidence is not L3 execution evidence.

---

### Task 1: Formal relation registry

**Files:**
- Modify: `tools/skill_transfer/robocasa_envs/__init__.py`
- Test: `tests/test_organizing_storage_box_robocasa_mapping.py`

**Interfaces:**
- Consumes: `formal_env_spec(relation_key: str)`.
- Produces: `ORGANIZING_STORAGE_BOX` and a relation-aware `create_formal_env`.

- [ ] Write tests asserting the exact storage-box mapping and Gym keyword arguments.
- [ ] Run `python -m unittest -v tests.test_organizing_storage_box_robocasa_mapping` and confirm failure because the relation is unsupported.
- [ ] Add `ORGANIZING_STORAGE_BOX` with `object_group="tupperware"`, `semantic_asset_proxy=True`, and registry `("lightwheel",)`.
- [ ] Select registries per mapping while leaving toy registries unchanged.
- [ ] Re-run the test and confirm all cases pass.

### Task 2: Real environment audit

**Files:**
- Create: `tools/check_organizing_storage_box_env.py`
- Test: `tests/test_check_organizing_storage_box_env.py`

**Interfaces:**
- Consumes: `create_formal_env("organizing::storage_box", seed=seed)`.
- Produces: `audit_environment(seed: int) -> dict[str, object]` and a JSON report.

- [ ] Write tests for report validation using a minimal fake environment.
- [ ] Confirm failure because the audit module is absent.
- [ ] Implement reset, camera/action/object/cabinet checks, and atomic JSON output.
- [ ] Confirm the unit tests pass.
- [ ] Run real reset audits for seeds `0`, `1`, and `2`.
- [ ] Verify every report says `simulator_mapping_ready=true` and `task_success_at_reset=false`.

### Task 3: Regression and handoff

**Files:**
- Create: `docs/STORAGE_BOX_HUMAN_VIDEO_L2_STATUS_2026-07-24.md`

**Interfaces:**
- Consumes: L2 Skill IR manifest and three simulator reset reports.
- Produces: evidence-level status and exact expert-collection next steps.

- [ ] Run all storage-box unit tests and Python compile checks.
- [ ] Re-run the existing toy environment mapping tests.
- [ ] Document the proxy limitation, three L2 records, reset results, and L3 acceptance gates.
