import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image


MODULE = Path(__file__).parents[1] / "update_chapter5_midterm_simplified.py"
SPEC = importlib.util.spec_from_file_location("update_chapter5_midterm_simplified", MODULE)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def test_standardize_figure_image_writes_docx_compatible_png():
    with TemporaryDirectory() as temp_dir:
        src = Path(temp_dir) / "frame.jpg"
        Image.new("RGB", (32, 24), (30, 60, 90)).save(src)

        out = MOD.standardize_figure_image(src)

        assert out.suffix == ".png"
        assert Image.open(out).format == "PNG"
