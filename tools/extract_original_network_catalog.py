"""Extract the network-control object tables from the read-only original report."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from docx import Document


NETWORK_TABLES = {
    "table_5_2_household_appliance_management": 18,
    "table_5_3_environment_regulation": 19,
    "table_5_4_home_security": 20,
    "table_5_7_maintenance_and_care": 23,
}


def clean_cell(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def extract_catalog(doc_path: Path) -> dict:
    document = Document(doc_path)
    catalog = {}
    for name, index in NETWORK_TABLES.items():
        table = document.tables[index]
        rows = []
        for row in table.rows:
            values = [clean_cell(cell.text) for cell in row.cells]
            if any(values):
                rows.append(values)
        catalog[name] = {"source_table_index": index, "rows": rows}
    return {"source_document": str(doc_path), "catalog": catalog}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-doc", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(extract_catalog(args.input_doc), ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
