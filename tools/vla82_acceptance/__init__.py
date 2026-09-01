"""Strict evidence tooling for the VLA82 midterm acceptance completion."""

from .contracts import (
    REQUIRED_EVIDENCE,
    build_acceptance_summary,
    validate_item_report,
)

__all__ = [
    "REQUIRED_EVIDENCE",
    "build_acceptance_summary",
    "validate_item_report",
]
