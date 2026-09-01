"""Source-backed operation specifications for the VLA82 full simulation."""

from .annotations import OperationSpec, compile_all, compile_operation_spec
from .contracts import validate_fixed_scope, validate_no_shortcuts

__all__ = [
    "OperationSpec",
    "compile_all",
    "compile_operation_spec",
    "validate_fixed_scope",
    "validate_no_shortcuts",
]
