# Household VLA Data Audit and Coverage Design

## Purpose

Replace the previous object-first, overlap-prone household taxonomy with an evidence-first plan for an indoor household robot. The plan must use only public robot demonstrations that contain visual observations, language/task text, and executable robot action labels (or a documented conversion path). It must not use self-collected data.

## Scope

- Indoor household work only.
- A final category describes a distinct household service outcome, not an atomic motion, fixture type, or isolated object.
- A category must contain at least five ordinary household objects with demonstrated operations.
- No category may be added merely to reach fifteen. The audit will state any verified shortfall.
- Do not create separate categories for power/water controls, loose-trash handling, leisure-item cleaning, storage fixtures, or duplicate organization: those are operations within broader service outcomes.

## Data hierarchy

1. **RoboCasa-365** is the primary source for broad simulated household VLA training because it provides household scenes, language tasks, visual observations, and robot actions.
2. **BEHAVIOR-1K** augments long-horizon service tasks where the downloaded local pairs contain task text, RGB video, and action/state Parquet. Its 23-D R1Pro control must be converted before use with the project robot.
3. **BridgeData V2 / RT-1 / DROID** provide general robot manipulation pretraining and must not alone define household-service categories.
4. **AIRoA-MoMa / HSR household teleoperation** is the candidate source to close gaps in indoor delivery and daily service, subject to an audit of its downloadable task metadata and action schema.
5. Local **AgiBotWorld Alpha** metadata can be used only after its actual task, visual stream, and action files are verified; supermarket and non-household scenes are excluded.

## Audit procedure

For every candidate episode, record: dataset, exact instruction, environment, RGB stream, action representation, task/object names, and conversion path. A task/object entry becomes *training-ready* only after all of these are present. A category enters the final workbook only when it has at least five non-contrived household objects backed by qualifying episodes.

## Non-overlap rules

- Appliance operation includes appliance controls; no separate switch/water/electric category.
- Indoor hygiene includes cleaning supplies, waste disposal, and ordinary household-item cleaning; no separate loose-debris or leisure-item-cleaning category.
- Retrieval, delivery, placement, and opening storage fixtures belong to one indoor item-service category.
- Organization and return-to-place is one category only.
- Food is limited to two outcomes: storage/preservation and preparation/cooking. Meal cleanup is distinct because it begins after eating and targets used tableware/leftovers.

## Deliverables

1. A source audit covering local RoboCasa, BEHAVIOR-1K, Bridge/RT-1, AgiBot metadata, and candidate HSR data.
2. A minimal download plan capped by available disk space, with each selected shard tied to a coverage gap.
3. A replacement Excel workbook containing only evidence-backed broad tasks, at least 120 real everyday objects, each object's demonstrated operation, source task, dataset, and training/conversion status.
4. A separate gap sheet listing any category that lacks qualifying data rather than inventing coverage.

## Validation

- Verify sampled episodes have matching RGB, language/task text, and action records.
- Verify all listed objects arise from included source instructions.
- Verify five-or-more distinct ordinary objects per final category.
- Detect duplicate category purposes before workbook generation.
- Do not claim direct OpenVLA training for data requiring an action-space conversion.

## Constraints and risks

The current local BEHAVIOR subset is valuable but cannot independently justify fifteen non-overlapping service categories. RoboCasa-365 and an HSR dataset are therefore audited before any final taxonomy is frozen. The project has approximately 267 GB free disk; downloads are selective and reversible.
