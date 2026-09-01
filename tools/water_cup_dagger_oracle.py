"""Recoverable simulator-state oracle used only to label DAgger samples.

The oracle deliberately has no RoboCasa imports so its state machine and safety
contract can be unit tested in the lightweight OpenVLA environment. Simulator
integration supplies world positions and the controller's world-to-origin
coordinate transform.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


VectorTransform = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class OracleSnapshot:
    eef_position: np.ndarray
    object_position: np.ndarray
    grasped: bool
    success: bool = False


@dataclass(frozen=True)
class OracleDecision:
    action: np.ndarray
    phase: str
    force_expert: bool


class WaterCupDaggerOracle:
    """Top-grasp, place-in-cabinet oracle with failed-grasp recovery."""

    APPROACH_HEIGHT = 0.18
    GRASP_HEIGHT_OFFSET = 0.002
    LIFT_HEIGHT = 0.22
    APPROACH_TOLERANCE = 0.025
    CONTACT_TOLERANCE = 0.010
    CLOSE_HOLD_STEPS = 35
    RELEASE_HOLD_STEPS = 35
    SETTLE_STEPS = 20

    def __init__(
        self,
        destination_front: np.ndarray,
        destination_center: np.ndarray,
        destination_retreat: np.ndarray,
        world_to_origin: VectorTransform | None = None,
    ) -> None:
        self.destination_front = self._vector(destination_front)
        self.destination_center = self._vector(destination_center)
        self.destination_retreat = self._vector(destination_retreat)
        self.world_to_origin = world_to_origin or (lambda value: value)
        self.phase = "approach_object"
        self.phase_steps = 0
        self.grasp_attempts = 0
        self.lift_target: np.ndarray | None = None

    @staticmethod
    def _vector(value: np.ndarray) -> np.ndarray:
        vector = np.asarray(value, dtype=float)
        if vector.shape != (3,):
            raise ValueError(f"expected a 3-vector, got {vector.shape}")
        return vector.copy()

    def _distance(self, current: np.ndarray, target: np.ndarray) -> float:
        error = self.world_to_origin(target) - self.world_to_origin(current)
        return float(np.linalg.norm(error))

    def _motion(
        self,
        snapshot: OracleSnapshot,
        target: np.ndarray,
        *,
        closed: bool,
        force_expert: bool = False,
    ) -> OracleDecision:
        error = self.world_to_origin(target) - self.world_to_origin(
            self._vector(snapshot.eef_position)
        )
        limit = 0.20 if closed else 1.0
        translation = np.clip(error / 0.05, -limit, limit)
        action = np.r_[translation, np.zeros(3), float(closed)].astype(np.float32)
        return OracleDecision(action, self.phase, force_expert)

    def _hold(self, *, closed: bool, force_expert: bool = True) -> OracleDecision:
        action = np.r_[np.zeros(6), float(closed)].astype(np.float32)
        return OracleDecision(action, self.phase, force_expert)

    def _enter(self, phase: str) -> None:
        self.phase = phase
        self.phase_steps = 0

    def _restart_grasp(self) -> None:
        self.grasp_attempts += 1
        self.lift_target = None
        self._enter("approach_object")

    def decide(self, snapshot: OracleSnapshot) -> OracleDecision:
        eef = self._vector(snapshot.eef_position)
        obj = self._vector(snapshot.object_position)

        # Re-evaluate transitions within one call so reaching a waypoint does not
        # inject an unlabeled no-op between phases.
        while True:
            if snapshot.success:
                self._enter("done")

            if self.phase == "approach_object":
                target = obj + np.array([0.0, 0.0, self.APPROACH_HEIGHT])
                if self._distance(eef, target) <= self.APPROACH_TOLERANCE:
                    self._enter("descend_to_object")
                    continue
                return self._motion(snapshot, target, closed=False)

            if self.phase == "descend_to_object":
                target = obj + np.array([0.0, 0.0, self.GRASP_HEIGHT_OFFSET])
                if self._distance(eef, target) <= self.CONTACT_TOLERANCE:
                    self._enter("close_gripper")
                    continue
                return self._motion(snapshot, target, closed=False)

            if self.phase == "close_gripper":
                if self.phase_steps < self.CLOSE_HOLD_STEPS:
                    self.phase_steps += 1
                    return self._hold(closed=True)
                if not snapshot.grasped:
                    self._restart_grasp()
                    continue
                self.lift_target = eef + np.array([0.0, 0.0, self.LIFT_HEIGHT])
                self._enter("lift_object")
                continue

            if self.phase in {
                "lift_object",
                "approach_cabinet",
                "move_inside_cabinet",
            } and not snapshot.grasped:
                self._restart_grasp()
                continue

            if self.phase == "lift_object":
                assert self.lift_target is not None
                if self._distance(eef, self.lift_target) <= self.APPROACH_TOLERANCE:
                    self._enter("approach_cabinet")
                    continue
                return self._motion(snapshot, self.lift_target, closed=True)

            if self.phase == "approach_cabinet":
                if self._distance(eef, self.destination_front) <= self.APPROACH_TOLERANCE:
                    self._enter("move_inside_cabinet")
                    continue
                return self._motion(snapshot, self.destination_front, closed=True)

            if self.phase == "move_inside_cabinet":
                if self._distance(eef, self.destination_center) <= self.APPROACH_TOLERANCE:
                    self._enter("release_object")
                    continue
                return self._motion(snapshot, self.destination_center, closed=True)

            if self.phase == "release_object":
                if self.phase_steps < self.RELEASE_HOLD_STEPS:
                    self.phase_steps += 1
                    return self._hold(closed=False)
                self._enter("retreat_from_cabinet")
                continue

            if self.phase == "retreat_from_cabinet":
                if self._distance(eef, self.destination_retreat) <= self.APPROACH_TOLERANCE:
                    self._enter("settle")
                    continue
                return self._motion(snapshot, self.destination_retreat, closed=False)

            if self.phase == "settle":
                if self.phase_steps < self.SETTLE_STEPS:
                    self.phase_steps += 1
                    return self._hold(closed=False, force_expert=False)
                self._enter("done")
                continue

            if self.phase == "done":
                return self._hold(closed=False, force_expert=True)

            raise RuntimeError(f"unknown oracle phase: {self.phase}")
