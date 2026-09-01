"""Filter the prior 122-object review sheet using the user's household rules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import openpyxl


EXCLUDE = {
    "boxing_glove", "rose", "trumpet", "bacon", "pizza_dough", "popcorn_bag", "chips", "oatmeal",
    "easter_egg", "pumpkin", "wreath", "candy_cane", "gift_box", "christmas_tree", "car_trunk",
    "driveway", "lawn", "tree", "pesticide_atomizer", "axe", "log", "firewood", "cauldron",
}
CANONICAL = {"sandal": "shoe", "gym_shoe": "shoe", "tennis_racket": "tennis_equipment", "tennis_ball": "tennis_equipment"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    book = openpyxl.load_workbook(args.workbook, read_only=True, data_only=True)
    sheet = book.worksheets[2]
    headers = [cell.value for cell in next(sheet.iter_rows(min_row=1, max_row=1))]
    rows = []
    seen = set()
    for values in sheet.iter_rows(min_row=2, values_only=True):
        if not values or not values[0]:
            continue
        row = dict(zip(headers, values))
        object_id = str(row["object"])
        if object_id in EXCLUDE:
            continue
        canonical = CANONICAL.get(object_id, object_id)
        if canonical in seen:
            continue
        row["object"] = canonical
        seen.add(canonical)
        rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"retained": len(rows), "excluded": 122 - len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    raise SystemExit(main())
