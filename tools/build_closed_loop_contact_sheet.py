"""Build a pixel-verifiable contact sheet from real closed-loop PNG frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    ROOT
    / "outputs"
    / "midterm_testing_vla82"
    / "closed_loop_evidence"
    / "manifest_8category_closed_loop_final.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs"
    / "midterm_testing_vla82"
    / "closed_loop_evidence"
    / "evidence_closed_loop_real_frames.png"
)


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def build_contact_sheet(results: Sequence[dict[str, Any]], output: Path) -> Path:
    if not results:
        raise ValueError("results must not be empty")
    panel_width, panel_height = 620, 360
    columns = 2
    rows = (len(results) + columns - 1) // columns
    title_height = 72
    canvas = Image.new(
        "RGB", (panel_width * columns, title_height + panel_height * rows), "#EEF2F6"
    )
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas.width, title_height), fill="#16324F")
    draw.text(
        (canvas.width // 2, title_height // 2),
        "8类纯OpenVLA真实闭环：首帧 / 末帧证据（0/8成功）",
        font=_font(26, bold=True),
        fill="white",
        anchor="mm",
    )
    for index, result in enumerate(results):
        row, column = divmod(index, columns)
        left = column * panel_width
        top = title_height + row * panel_height
        draw.rectangle(
            (left + 8, top + 8, left + panel_width - 8, top + panel_height - 8),
            fill="white",
            outline="#A9B7C5",
            width=2,
        )
        title = (
            f"{result['selection_id']} | {result.get('source_table', '未标注表号')} | "
            f"{result.get('task_class', '未标注任务类')} | 任务失败"
        )
        draw.text(
            (left + panel_width // 2, top + 30),
            title,
            font=_font(16, bold=True),
            fill="#B42318",
            anchor="mm",
        )
        for image_index, (label, key) in enumerate(
            (("首帧", "first_frame"), ("末帧", "last_frame"))
        ):
            source = Image.open(result[key]).convert("RGB")
            source.thumbnail((270, 270), Image.Resampling.LANCZOS)
            image_left = left + 28 + image_index * 300
            image_top = top + 68
            canvas.paste(source, (image_left, image_top))
            draw.rectangle(
                (
                    image_left - 1,
                    image_top - 1,
                    image_left + source.width,
                    image_top + source.height,
                ),
                outline="#71869A",
                width=1,
            )
            draw.text(
                (image_left + source.width // 2, top + panel_height - 24),
                label,
                font=_font(15, bold=True),
                fill="#31465A",
                anchor="mm",
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    path = build_contact_sheet(manifest["results"], args.output)
    print(path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
