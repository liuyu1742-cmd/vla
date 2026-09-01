"""Derive strict manifests and visual summaries from verified item reports."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps

from .contracts import build_acceptance_summary


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_entry(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "size": path.stat().st_size,
        "sha256": _sha256(path),
    }


def aggregate(
    task_reports: Sequence[Mapping[str, Any]],
    object_reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    summary = build_acceptance_summary(task_reports, object_reports)
    return {
        "schema_version": "vla82_acceptance_completion_v1",
        "generated_at": datetime.now().astimezone().isoformat(),
        **summary,
        "task_results": [dict(item) for item in task_reports],
        "object_results": [dict(item) for item in object_reports],
    }


def write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        Path("C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def _fit(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as source:
        return ImageOps.fit(source.convert("RGB"), size, Image.Resampling.LANCZOS)


def draw_contact_sheet(
    reports: Sequence[Mapping[str, Any]], output: Path
) -> None:
    columns = 2
    panel_width = 720
    panel_height = 230
    title_height = 90
    rows = max(1, (len(reports) + columns - 1) // columns)
    canvas = Image.new(
        "RGB", (panel_width * columns, title_height + rows * panel_height), "#EEF3F8"
    )
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas.width, title_height), fill="#123B5D")
    draw.text(
        (canvas.width // 2, 35),
        "VLA82 补测首帧/末帧证据",
        font=_font(30, bold=True),
        fill="white",
        anchor="mm",
    )
    draw.text(
        (canvas.width // 2, 68),
        f"项目数：{len(reports)}",
        font=_font(18),
        fill="#D8EAF6",
        anchor="mm",
    )
    for index, report in enumerate(reports):
        row, column = divmod(index, columns)
        left = column * panel_width
        top = title_height + row * panel_height
        draw.rounded_rectangle(
            (left + 10, top + 10, left + panel_width - 10, top + panel_height - 10),
            radius=14,
            fill="white",
            outline="#AEBECB",
            width=2,
        )
        status = str(report.get("acceptance_status", "FAIL"))
        status_color = "#087A49" if status == "PASS" else "#B42318"
        title = f"{report.get('selection_id')}  {report.get('object', '')}  {status}"
        draw.text((left + 24, top + 24), title, font=_font(17, bold=True), fill=status_color)
        bundle = Path(str(report["bundle_dir"]))
        for x_offset, name, label in (
            (24, "first_frame.png", "首帧"),
            (370, "last_frame.png", "末帧"),
        ):
            frame = _fit(bundle / name, (320, 160))
            canvas.paste(frame, (left + x_offset, top + 56))
            draw.rectangle(
                (left + x_offset, top + 56, left + x_offset + 55, top + 82),
                fill="#123B5D",
            )
            draw.text(
                (left + x_offset + 27, top + 69),
                label,
                font=_font(13, bold=True),
                fill="white",
                anchor="mm",
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
