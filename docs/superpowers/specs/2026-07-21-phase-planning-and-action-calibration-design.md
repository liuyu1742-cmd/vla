# OpenVLA Phase Planning and RoboCasa Action Calibration Design

## Goal

Turn the existing EPIC-derived `pick(cup) → place(cup, cabinet)` skill manifest into a bounded RoboCasa control policy with measurable, calibrated action translation.

## Phase Planner

The planner is a two-state finite-state machine. In `pick`, OpenVLA receives `pick up the glass cup`; in `place`, it receives `place the glass cup in the cabinet`. A phase transition requires a simulator-visible predicate, never merely a fixed number of actions. The first deliverable records predicates and leaves a failed predicate in the current phase.

## Action Calibration

RoboCasa receives native 12D actions: the seven OpenVLA dimensions followed by four fixed base-motion zeros and one fixed control-mode zero. Calibration runs fixed one-axis probes from a reset state and records the observed end-effector delta. It derives a per-axis safe scale and records the gripper sign/threshold from the simulator action convention. Until calibration exists, no claim is made that Bridge and RoboCasa action units agree.

## Boundaries

- EPIC contributes only phase semantics.
- OpenVLA remains in its CUDA/Python 3.10 environment; RoboCasa remains in its separate environment.
- The phase planner and calibrator do not train model weights or read BridgeData V2.
- All probes run in simulation only and use deterministic seed 0.

## Acceptance

- A plan report lists current phase, prompt, predicate outcome, and transition reason.
- Calibration report lists all seven source dimensions, corresponding observed simulator response, and chosen safe scales.
- The next micro-step uses the calibrated 12D action conversion.
