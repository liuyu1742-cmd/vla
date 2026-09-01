import tempfile
import unittest
from pathlib import Path

from PIL import Image

from tools.build_closed_loop_contact_sheet import build_contact_sheet


class ClosedLoopContactSheetTests(unittest.TestCase):
    def test_contact_sheet_contains_source_image_pixels(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "first.png"
            last = root / "last.png"
            output = root / "sheet.png"
            Image.new("RGB", (32, 32), "#ff0000").save(first)
            Image.new("RGB", (32, 32), "#0000ff").save(last)

            build_contact_sheet(
                [{"selection_id": "VLA82-TEST", "first_frame": str(first), "last_frame": str(last)}],
                output,
            )

            rendered = Image.open(output).convert("RGB")
            colors = set(rendered.getdata())
            self.assertIn((255, 0, 0), colors)
            self.assertIn((0, 0, 255), colors)


if __name__ == "__main__":
    unittest.main()
