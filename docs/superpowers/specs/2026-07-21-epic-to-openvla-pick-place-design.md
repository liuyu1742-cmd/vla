# EPIC-KITCHENS to OpenVLA Pick-and-Place Design

## Goal

Establish the first end-to-end human-demonstration skill-transfer chain: parse an EPIC-KITCHENS household video into a high-level pick-and-place skill, instantiate the equivalent RoboCasa task, and use OpenVLA for closed-loop robot action prediction.

## Boundaries

- EPIC-KITCHENS supplies human-video semantics: hands, interacted objects, verb/noun labels, and event order.
- It does not supply robot actions and will not be used as direct OpenVLA action supervision.
- RoboCasa supplies robot-camera observations, robot actions, and task-success signals.
- OpenVLA consumes the current robot-camera frame plus a phase instruction, then predicts one 7D robot action.
- BridgeData V2 remains optional general robot-action data; it is not read in this first chain.

## Pipeline

1. Select one EPIC clip whose annotation expresses `take/pick up <object>` followed by `put/place <object> <target>`.
2. Extract a compact skill manifest containing clip ID, source object, destination, ordered phases, and confidence values.
3. Map the semantic object and phases to a RoboCasa `PickPlaceCounterToCabinet` task instance using a supported cup-like object.
4. RoboCasa renders its virtual robot camera; an IPC OpenVLA worker predicts actions without sharing Python dependencies with RoboCasa.
5. Record phase transitions, actions, and task outcome. The first deliverable is truthful integration evidence, not a claim of transfer success.

## Extension Model

The skill manifest is deliberately independent of EPIC and RoboCasa. Future task types add verb templates and simulator adapters; future objects add a semantic-to-RoboCasa object mapping. This permits gradual coverage toward 15 household task types and 120 objects without redesigning the pipeline.

## Acceptance Criteria

- A manifest validates that every phase has a verb, object, and target state.
- The pick-and-place mapping resolves only supported simulator objects and fails clearly otherwise.
- The final report links the EPIC clip manifest, the RoboCasa task, and all OpenVLA action records.
- No physical camera, physical robot, or direct human-video-to-robot-joint mapping is used.
