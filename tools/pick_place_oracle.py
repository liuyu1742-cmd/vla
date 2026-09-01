"""Relation-agnostic top-grasp and place expert with failed-grasp recovery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


VectorTransform = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class PickPlaceSnapshot:
    eef_position: np.ndarray
    object_position: np.ndarray
    grasped: bool
    success: bool = False


@dataclass(frozen=True)
class PickPlaceDecision:
    action: np.ndarray
    phase: str
    canonical_action: str
    force_expert: bool


class PickPlaceOracle:
    """Execute locate/grasp/move/place using generic destination waypoints."""

    APPROACH_HEIGHT = 0.18
    GRASP_HEIGHT_OFFSET = 0.002
    LIFT_HEIGHT = 0.22
    APPROACH_TOLERANCE = 0.025
    CONTACT_TOLERANCE = 0.010
    CLOSE_HOLD_STEPS = 35
    RELEASE_HOLD_STEPS = 35
    SETTLE_STEPS = 20

    CANONICAL_BY_PHASE = {
        "approach_object": "locate",
        "descend_to_object": "locate",
        "close_gripper": "grasp",
        "lift_object": "move",
        "approach_destination": "move",
        "move_inside_destination": "move",
        "release_object": "place",
        "retreat_from_destination": "place",
        "settle": "place",
        "done": "place",
    }

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

    def _decision(
        self,
        action: np.ndarray,
        *,
        force_expert: bool,
    ) -> PickPlaceDecision:
        return PickPlaceDecision(
            np.asarray(action, dtype=np.float32),
            self.phase,
            self.CANONICAL_BY_PHASE[self.phase],
            force_expert,
        )

    def _motion(
        self,
        snapshot: PickPlaceSnapshot,
        target: np.ndarray,
        *,
        closed: bool,
    ) -> PickPlaceDecision:
        error = self.world_to_origin(target) - self.world_to_origin(
            self._vector(snapshot.eef_position)
        )
        translation = np.clip(error / 0.05, -0.20 if closed else -1.0, 0.20 if closed else 1.0)
        action = np.r_[translation, np.zeros(3), float(closed)]
        return self._decision(action, force_expert=False)

    def _hold(self, *, closed: bool, force_expert: bool = True) -> PickPlaceDecision:
        return self._decision(
            np.r_[np.zeros(6), float(closed)], force_expert=force_expert
        )

    def _enter(self, phase: str) -> None:
        self.phase = phase
        self.phase_steps = 0

    def _restart_grasp(self) -> None:
        self.grasp_attempts += 1
        self.lift_target = None
        self._enter("approach_object")

    def decide(self, snapshot: PickPlaceSnapshot) -> PickPlaceDecision:
        eef = self._vector(snapshot.eef_position)
        obj = self._vector(snapshot.object_position)
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
                "approach_destination",
                "move_inside_destination",
            } and not snapshot.grasped:
                self._restart_grasp()
                continue

            if self.phase == "lift_object":
                assert self.lift_target is not None
                if self._distance(eef, self.lift_target) <= self.APPROACH_TOLERANCE:
                    self._enter("approach_destination")
                    continue
                return self._motion(snapshot, self.lift_target, closed=True)

            if self.phase == "approach_destination":
                if self._distance(eef, self.destination_front) <= self.APPROACH_TOLERANCE:
                    self._enter("move_inside_destination")
                    continue
                return self._motion(snapshot, self.destination_front, closed=True)

            if self.phase == "move_inside_destination":
                if self._distance(eef, self.destination_center) <= self.APPROACH_TOLERANCE:
                    self._enter("release_object")
                    continue
                return self._motion(snapshot, self.destination_center, closed=True)

            if self.phase == "release_object":
                if self.phase_steps < self.RELEASE_HOLD_STEPS:
                    self.phase_steps += 1
                    return self._hold(closed=False)
                self._enter("retreat_from_destination")
                continue

            if self.phase == "retreat_from_destination":
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
                return self._hold(closed=False)

            raise RuntimeError(f"unknown pick/place oracle phase: {self.phase}")


__all__ = ["PickPlaceDecision", "PickPlaceOracle", "PickPlaceSnapshot"]
