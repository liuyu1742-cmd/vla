"""Read a simple XLSX worksheet using only the Python standard library."""

from __future__ import annotations

import posixpath
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
OFFICE_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PACKAGE_REL = "http://schemas.openxmlformats.org/package/2006/relationships"


def _column_index(cell_ref: str) -> int:
    match = re.match(r"[A-Z]+", cell_ref.upper())
    if match is None:
        raise ValueError(f"invalid cell reference: {cell_ref}")
    result = 0
    for letter in match.group(0):
        result = result * 26 + ord(letter) - ord("A") + 1
    return result - 1


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    return [
        "".join(node.text or "" for node in item.findall(f".//{{{MAIN}}}t"))
        for item in root.findall(f"{{{MAIN}}}si")
    ]


def _sheet_target(archive: zipfile.ZipFile, sheet_name: str) -> str:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    sheet = next(
        (
            node
            for node in workbook.findall(f".//{{{MAIN}}}sheet")
            if node.attrib.get("name") == sheet_name
        ),
        None,
    )
    if sheet is None:
        raise ValueError(f"sheet not found: {sheet_name}")
    relation_id = sheet.attrib[f"{{{OFFICE_REL}}}id"]
    relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    relation = next(
        (
            node
            for node in relationships.findall(f"{{{PACKAGE_REL}}}Relationship")
            if node.attrib.get("Id") == relation_id
        ),
        None,
    )
    if relation is None:
        raise ValueError(f"relationship not found: {relation_id}")
    target = relation.attrib["Target"].replace("\\", "/").lstrip("/")
    return target if target.startswith("xl/") else posixpath.normpath(f"xl/{target}")


def _cell_text(cell: ET.Element, strings: list[str]) -> str:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(
            node.text or "" for node in cell.findall(f".//{{{MAIN}}}t")
        )
    value = cell.find(f"{{{MAIN}}}v")
    raw = "" if value is None or value.text is None else value.text
    if cell_type == "s" and raw:
        return strings[int(raw)]
    return raw


def read_sheet_table(path: Path, sheet_name: str) -> list[dict[str, str]]:
    """Return non-empty data rows keyed by the first worksheet row."""
    with zipfile.ZipFile(path, "r") as archive:
        strings = _shared_strings(archive)
        root = ET.fromstring(archive.read(_sheet_target(archive, sheet_name)))

    rows: list[list[str]] = []
    for row in root.findall(f".//{{{MAIN}}}row"):
        values: list[str] = []
        for cell in row.findall(f"{{{MAIN}}}c"):
            index = _column_index(cell.attrib["r"])
            values.extend([""] * (index + 1 - len(values)))
            values[index] = _cell_text(cell, strings).strip()
        rows.append(values)

    if not rows:
        return []
    headers = rows[0]
    return [
        {
            header: values[index] if index < len(values) else ""
            for index, header in enumerate(headers)
            if header
        }
        for values in rows[1:]
        if any(values)
    ]
