"""Authoritative household skill-coverage registry utilities."""

from .contracts import (
    Applicability,
    ValidationState,
    validate_matrix_entry,
    validate_object_candidate,
)

__all__ = [
    "Applicability",
    "ValidationState",
    "validate_matrix_entry",
    "validate_object_candidate",
]
