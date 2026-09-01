import json

from PIL import Image

from tools.openvla_4l_evaluator import render_rollout_evidence


def test_render_rollout_evidence_uses_saved_simulation_frames(tmp_path):
    frame_paths = []
    for index, color in enumerate(("navy", "orange", "green")):
        path = tmp_path / f"frame_{index:04d}.png"
        Image.new("RGB", (128, 96), color).save(path)
        frame_paths.append(str(path))
    report = {
        "method_label": "Last-Layer-Only",
        "checkpoint": "models/last-layer",
        "uses_expert_recovery": False,
        "episodes": [
            {
                "seed": 0,
                "success": True,
                "steps": 3,
                "frame_paths": frame_paths,
                "records": [
                    {"step": 0, "action": [0.0] * 7},
                    {"step": 1, "action": [0.1] * 7},
                    {"step": 2, "action": [0.2] * 7},
                ],
            }
        ],
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    output = tmp_path / "rollout.png"
    render_rollout_evidence(report_path, output)
    with Image.open(output) as rendered:
        assert rendered.width >= 900
        assert rendered.height >= 500

