"""Compile the fixed VLA82 registry into traceable operation specifications."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tools.vla82_acceptance.demo_evidence import load_registry


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY_PATH = (
    PROJECT_ROOT / "outputs" / "midterm_testing_vla82" / "vla82_midterm_registry.json"
)

# These phrases are intentionally source-language first.  The parser finds every
# phrase in source order, so a multi-operation annotation remains a multi-phase
# specification rather than being collapsed to a generic label.
PHRASE_PHASES = {
    "擦拭": "wipe",
    "刷洗": "scrub",
    "喷洒": "spray",
    "投放": "deposit",
    "放入": "insert",
    "叠放": "stack",
    "并排": "arrange",
    "整理": "arrange",
    "拉开": "pull",
    "拉出": "pull",
    "推入": "push",
    "开门": "open",
    "关门": "close",
    "关闭": "close",
    "打开": "open",
    "旋钮": "turn_knob",
    "按压": "press",
    "合盖": "close_lid",
    "抓取": "grasp",
    "放回": "return",
    "摆放": "place",
    "放置": "place",
    "放至": "place",
    "移至": "place",
    "取用": "grasp",
    "装入": "insert",
    "收纳": "insert",
    "pick": "grasp",
    "place": "place",
    "close": "close",
    "open": "open",
    "push": "push",
    "pull": "pull",
    "turn": "turn_knob",
    "press": "press",
}

_ENGLISH_PHRASES = frozenset(
    {"pick", "place", "close", "open", "push", "pull", "turn", "press"}
)


@dataclass(frozen=True)
class OperationSpec:
    selection_id: str
    task: str
    object_name: str
    operation_text: str
    source_kind: str
    source_path: str
    phases: tuple[str, ...]
    manipulated_objects: tuple[str, ...]
    predicate_names: tuple[str, ...]
    source_sha256: str
    source_table: str = ""
    source_differences: tuple[tuple[str, str], ...] = ()


def sha256_file(path: Path) -> str:
    """Return the SHA-256 fingerprint of a source artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _first_text(value: Any) -> str:
    """Extract a stable non-empty instruction from supported JSON payloads."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Mapping):
        for key in ("instruction", "operation", "text", "description", "label", "value"):
            if key in value:
                text = _first_text(value[key])
                if text:
                    return text
        # Recurse through JSON containers but do not mistake scalar metadata
        # (for example, an object's name) for an executable instruction.
        for key in sorted(value, key=str):
            nested = value[key]
            if not isinstance(nested, (Mapping, list, tuple)):
                continue
            text = _first_text(nested)
            if text:
                return text
    if isinstance(value, (list, tuple)):
        for item in value:
            text = _first_text(item)
            if text:
                return text
    return ""


def _read_instruction(path: Path) -> str:
    if not path.is_file():
        return ""
    try:
        return _first_text(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return ""


def parse_ordered_phases(operation_text: str) -> tuple[str, ...]:
    """Map all recognized Chinese/English phrases in deterministic source order."""
    matches: list[tuple[int, int, int, str]] = []
    for phrase, phase in PHRASE_PHASES.items():
        if phrase in _ENGLISH_PHRASES:
            pattern = re.compile(rf"\b{re.escape(phrase)}\b", re.IGNORECASE)
            found = pattern.finditer(operation_text)
        else:
            found = re.finditer(re.escape(phrase), operation_text)
        matches.extend(
            (match.start(), match.end(), -len(phrase), phase) for match in found
        )
    matches.sort(key=lambda item: (item[0], item[2]))
    ordered: list[str] = []
    occupied_until = -1
    for start, end, _, phase in matches:
        # A compound verb such as “摆放至” contains the overlapping aliases
        # “摆放” and “放至”, but it describes one physical action.  Keep the
        # earliest/longest match and preserve genuinely separate occurrences.
        if start < occupied_until:
            continue
        # Each match is a source action occurrence.  Keeping adjacent repeats
        # is essential: two annotated wipes/placements need two independent
        # physical event windows during evaluation.
        ordered.append(phase)
        occupied_until = end
    return tuple(ordered) if ordered else ("composite",)


def select_source_annotation(*, operation: str, instruction: str) -> tuple[str, str]:
    """Choose the most specific executable annotation from one recorded episode.

    A slash-separated operation label such as ``open/close`` is a catalogue
    capability, not evidence that both motions occurred.  When its paired
    instruction specifies one motion, execute only that concrete instruction.
    """
    # Source JSON was produced by multiple encoders; the slash itself is the
    # reliable capability-list marker even where Chinese action glyphs arrive
    # mojibaked.  A non-empty concrete instruction still supplies its verb.
    generic_pair = bool(re.search(r"[/／]", operation))
    if generic_pair and instruction and parse_ordered_phases(instruction) != ("composite",):
        return instruction, "instruction_json_specific"
    if operation:
        return operation, "operation_json"
    if instruction:
        return instruction, "instruction_json"
    return "", ""


def predicates_for_phases(phases: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(f"{phase}_completed" for phase in phases) or ("composite_completed",)


def parse_manipulated_objects(text: str, selection: Mapping[str, Any]) -> tuple[str, ...]:
    """Anchor object attribution to the authoritative selected registry row."""
    object_name = str(selection.get("object", selection.get("object_name", ""))).strip()
    return (object_name,) if object_name else ("unspecified_object",)


def compile_operation_spec(selection: Mapping[str, Any]) -> OperationSpec:
    """Compile one registry selection using the required source precedence."""
    data_path = Path(str(selection["data_path"]))
    operation_path = data_path / "operation.json"
    instruction_path = data_path / "instruction.json"
    registry_path = Path(str(selection.get("_registry_path", DEFAULT_REGISTRY_PATH)))
    operation = _read_instruction(operation_path)
    instruction = _read_instruction(instruction_path)
    selected_text, selected_kind = select_source_annotation(operation=operation, instruction=instruction)
    selected_path = instruction_path if selected_kind.startswith("instruction") else operation_path
    candidates = (
        (selected_text, selected_kind, selected_path),
        (str(selection.get("real_video_annotation", "")).strip(), "ledger_real_video_annotation", registry_path),
        (str(selection.get("operation_label", "")).strip(), "ledger_operation_label", registry_path),
    )
    text, source_kind, source_path = next(
        (candidate for candidate in candidates if candidate[0]),
        ("unannotated operation", "synthetic_composite", registry_path),
    )
    phases = parse_ordered_phases(text)
    source_differences = tuple(
        (kind, candidate_text)
        for candidate_text, kind, _ in candidates
        if candidate_text and candidate_text != text
    )
    return OperationSpec(
        selection_id=str(selection["selection_id"]),
        task=str(selection["task"]),
        object_name=str(selection.get("object", selection.get("object_name", ""))),
        operation_text=text,
        source_kind=source_kind,
        source_path=str(source_path.resolve()),
        phases=phases,
        manipulated_objects=parse_manipulated_objects(text, selection),
        predicate_names=predicates_for_phases(phases),
        source_sha256=sha256_file(source_path) if source_path.is_file() else "",
        source_table=str(selection.get("source_table", "")),
        source_differences=source_differences,
    )


def compile_all(registry_path: Path) -> list[OperationSpec]:
    """Compile every selected registry object in the registry's declared order."""
    registry = load_registry(registry_path)
    selections = registry.get("selected_objects", [])
    if not isinstance(selections, list):
        raise ValueError("registry selected_objects must be a list")
    return [
        compile_operation_spec({**selection, "_registry_path": str(registry_path)})
        for selection in selections
    ]
