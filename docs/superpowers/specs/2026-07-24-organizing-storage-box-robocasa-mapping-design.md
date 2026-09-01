# Organizing Storage Box RoboCasa Mapping Design

## Goal

Promote the three audited `organizing::storage_box` human-video records from
L2 parsing evidence to a simulator-ready relation without claiming robot
execution success.

## Constraints

- The relation remains exactly `organizing::storage_box`.
- The Task-2 source videos and annotations remain read-only.
- Held-out seeds `201`, `202`, and `203` must never enter training or DAgger.
- A simulator asset proxy must be disclosed; it cannot be presented as an
  exact physical copy of the human-video storage box.
- L3 is awarded only after a robot rollout passes all native success
  predicates. Environment reset or expert success alone is not L3.
- Hybrid/final-contact assistance must remain separate from pure autonomous
  VLA results.

## Considered Approaches

### A. Native `tupperware` proxy in `PickPlaceCounterToCabinet` — selected

RoboCasa already ships multiple graspable, receptacle-like Lightwheel
`tupperware` assets. The atomic task opens the cabinet, samples the object on
the counter, and defines native success as the object being inside the cabinet
with the gripper far from the object. This preserves the canonical
`locate -> grasp -> move(storage) -> place` sequence and avoids new physics
assets before the closed loop works.

Trade-off: the asset is a household storage-container proxy, not an exact
human-video mesh. Every mapping and report must set
`semantic_asset_proxy=true`.

### B. Custom local storage-box mesh — deferred

This gives the best visual match and supports dimensions/material
randomization, but requires mesh acquisition, collision simplification,
inertial tuning, registration, and grasp validation. It is appropriate after
the native proxy establishes the L3 pipeline.

### C. Re-label the existing toy environment — rejected

This would reuse a stable controller but would make the object evidence
semantically false. A toy cannot count as a storage box merely because the
motion family is the same.

## Mapping Contract

```text
relation_key:              organizing::storage_box
robocasa_task:             PickPlaceCounterToCabinet
gym_id:                    robocasa/PickPlaceCounterToCabinet
object_group:              tupperware
target_object:             storage_box
simulator_object_proxy:    tupperware
semantic_asset_proxy:      true
target_region:             cabinet
canonical_target:          storage
success_predicate_version: organizing_storage_box_v1
```

The environment uses only the standard `lightwheel` registry. The reset audit
must confirm:

1. the sampled target is a tupperware asset;
2. observation contains the configured camera image;
3. the native action space is 12-dimensional;
4. the object starts on the counter and outside the cabinet;
5. the native success predicate is false at reset;
6. object/cabinet pose data needed by expert collection is available.

## Evidence and Error Handling

- Unsupported relations fail before environment construction.
- Missing `tupperware` assets or reset errors fail the mapping audit.
- Mapping output stores relation, seed, sampled model, object extent, action
  shape, initial predicates, and camera shape.
- Human-video L2 records keep `execution_evidence=null`.
- Reset success is reported as `simulator_mapping_ready`, never as task
  success.

## Testing

- Unit tests verify the exact mapping contract and Gym arguments.
- A real RoboCasa reset audit runs seeds `0`, `1`, and `2`.
- Existing `organizing::toy` mappings must remain unchanged.
- The next phase collects at least 30 successful non-held-out expert seeds,
  with held-out overlap equal to zero.

## Approval Basis

The user previously approved `organizing::storage_box` as the next formal
relation and explicitly instructed execution to continue without per-stage
confirmation. This design implements that standing approval while retaining
the evidence boundaries above.
