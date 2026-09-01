# Whole-Home VLA Coverage Design

## Purpose

Replace the kitchen-biased metadata mapping with a reproducible whole-home
household-robot coverage ledger. The final 15 task classes and 120 object
classes must be drawn from real robot datasets and must distinguish catalog
existence from training readiness.

## Decisions

- RoboCasa remains the kitchen simulation and action-execution backbone; it
  cannot alone define the final whole-home taxonomy.
- `arranging_buffet` is removed from the headline taxonomy because it overlaps
  with `setting_the_table` at household-service level.
- Each task-object row has one of three evidence states:
  `ready` (paired image-language-action demonstrations verified),
  `generatable` (a simulator task and expert collection route verified), or
  `catalog_only` (not eligible for the 15x120 training claim).
- The final coverage count includes only `ready` and `generatable` rows.
- Source data is kept separate by dataset and normalized into one manifest;
  original trajectories are never overwritten.

## Deliverables

1. A source manifest describing local and downloaded dataset metadata.
2. A normalized task-object evidence ledger, with room zone and training
   evidence for every row.
3. A workbook with 15 non-overlapping household-service task categories,
   120 eligible objects, source provenance, and coverage summaries.
4. Validation that no category is catalog-only, that every selected object has
   a concrete data-generation or demonstration path, and that the selected
   categories cover multiple household zones rather than only the kitchen.

## Boundaries

This phase establishes the truthful coverage contract and data-acquisition
targets. It does not claim that an uncollected simulator route has already
trained OpenVLA; collection and training begin only after the ledger marks the
route as `generatable` and records its collector command.
