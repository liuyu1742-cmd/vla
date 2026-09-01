import hashlib
import html
import json
import tempfile
import unittest
import zipfile
from pathlib import Path


HEADERS = [
    "task_id",
    "task_name",
    "object",
    "物体中文名称",
    "images",
    "videos",
    "annotations",
    "metadata",
    "object_dir",
]


def inline_cell(column: str, row: int, value: str) -> str:
    return (
        f'<c r="{column}{row}" t="inlineStr"><is><t>'
        f"{html.escape(value)}</t></is></c>"
    )


def write_registry_workbook(path: Path) -> None:
    rows = [
        HEADERS,
        [
            "appliance_management",
            "家电综合管理",
            "remote_control",
            "遥控器",
            "10",
            "0",
            "10",
            "0",
            "C:\\dataset\\appliance\\remote_control",
        ],
        [
            "object_fetching",
            "物品取送",
            "remote_control",
            "遥控器",
            "20",
            "1",
            "21",
            "0",
            "C:\\dataset\\fetching\\remote_control",
        ],
    ]
    sheet_rows = []
    for row_number, values in enumerate(rows, start=1):
        cells = "".join(
            inline_cell(chr(ord("A") + index), row_number, value)
            for index, value in enumerate(values)
        )
        sheet_rows.append(f'<row r="{row_number}">{cells}</row>')

    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "xl/workbook.xml",
            '<?xml version="1.0"?><workbook '
            'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            '<sheets><sheet name="任务物体数据表" sheetId="1" r:id="rId1"/></sheets>'
            "</workbook>",
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            '<?xml version="1.0"?><Relationships '
            'xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Target="worksheets/sheet1.xml"/>'
            "</Relationships>",
        )
        archive.writestr(
            "xl/worksheets/sheet1.xml",
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            f"<sheetData>{''.join(sheet_rows)}</sheetData></worksheet>",
        )


class BuildAuthoritativeCoverageTests(unittest.TestCase):
    def test_builds_deterministic_outputs_without_changing_source(self):
        from tools.build_authoritative_skill_coverage import main

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "scope.xlsx"
            output = root / "generated"
            write_registry_workbook(source)
            before = hashlib.sha256(source.read_bytes()).hexdigest()
            exit_code = main(
                [
                    "--xlsx",
                    str(source),
                    "--output-dir",
                    str(output),
                    "--expected-dataset-tasks",
                    "2",
                    "--expected-relations",
                    "2",
                    "--expected-candidates",
                    "1",
                ]
            )
            after = hashlib.sha256(source.read_bytes()).hexdigest()
            summary = json.loads(
                (output / "coverage_summary.json").read_text(encoding="utf-8")
            )
            matrix = json.loads(
                (output / "task_object_matrix.json").read_text(encoding="utf-8")
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(before, after)
        self.assertEqual(summary["dataset_task_count"], 2)
        self.assertEqual(summary["source_relation_count"], 2)
        self.assertEqual(summary["candidate_object_count"], 1)
        self.assertEqual(summary["validated_object_count"], 0)
        self.assertEqual(len(matrix), 15)
        self.assertEqual(summary["source"]["sha256"], before)

    def test_rejects_authoritative_count_mismatch(self):
        from tools.build_authoritative_skill_coverage import main

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "scope.xlsx"
            write_registry_workbook(source)
            with self.assertRaisesRegex(ValueError, "candidate objects"):
                main(
                    [
                        "--xlsx",
                        str(source),
                        "--output-dir",
                        str(root / "generated"),
                        "--expected-dataset-tasks",
                        "2",
                        "--expected-relations",
                        "2",
                        "--expected-candidates",
                        "2",
                    ]
                )


if __name__ == "__main__":
    unittest.main()
