import hashlib
import tempfile
import unittest
import zipfile
from pathlib import Path


def write_minimal_workbook(path: Path) -> None:
    strings = [
        "task_id",
        "task_name",
        "object",
        "物体中文名称",
        "images",
        "food_serving",
        "送餐饮水",
        "water_cup",
        "水杯",
    ]
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
            "xl/sharedStrings.xml",
            '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            + "".join(f"<si><t>{value}</t></si>" for value in strings)
            + "</sst>",
        )
        archive.writestr(
            "xl/worksheets/sheet1.xml",
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            "<sheetData>"
            '<row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c>'
            '<c r="C1" t="s"><v>2</v></c><c r="D1" t="s"><v>3</v></c>'
            '<c r="E1" t="s"><v>4</v></c></row>'
            '<row r="2"><c r="A2" t="s"><v>5</v></c><c r="B2" t="s"><v>6</v></c>'
            '<c r="C2" t="s"><v>7</v></c><c r="D2" t="s"><v>8</v></c>'
            '<c r="E2"><v>300</v></c></row>'
            "</sheetData></worksheet>",
        )


class XlsxReaderTests(unittest.TestCase):
    def test_reads_shared_strings_and_preserves_source(self):
        from tools.skill_coverage.xlsx_reader import read_sheet_table

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture.xlsx"
            write_minimal_workbook(path)
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            rows = read_sheet_table(path, "任务物体数据表")
            after = hashlib.sha256(path.read_bytes()).hexdigest()

        self.assertEqual(rows[0]["object"], "water_cup")
        self.assertEqual(rows[0]["物体中文名称"], "水杯")
        self.assertEqual(rows[0]["images"], "300")
        self.assertEqual(before, after)

    def test_missing_sheet_is_rejected(self):
        from tools.skill_coverage.xlsx_reader import read_sheet_table

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture.xlsx"
            write_minimal_workbook(path)
            with self.assertRaisesRegex(ValueError, "sheet not found"):
                read_sheet_table(path, "不存在")


if __name__ == "__main__":
    unittest.main()
