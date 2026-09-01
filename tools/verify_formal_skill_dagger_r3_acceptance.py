"""Verify DAgger-r3 training isolation and hybrid held-out acceptance evidence."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from tools.formal_skill_rollout import adapter_fingerprint


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    ROOT
    / "datasets"
    / "formal_skills"
    / "organizing_toy"
    / "training_manifest_dagger_r2.json"
)
ADAPTER = ROOT / "models" / "openvla-organizing-toy-lora-dagger-r3"
TRAINING_REPORT = ADAPTER / "training_report.json"
EVAL_ROOT = ROOT / "outputs" / "formal_skill_dagger_r3_hybrid_v3_eval"
SUMMARY = EVAL_ROOT / "acceptance_summary.json"
HELDOUT_SEEDS = {101, 102, 103}
RECOVERY_SEEDS = set(range(42, 50))
PREDICATES = ("simulator_success", "placed_in_storage", "gripper_released")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify() -> dict:
    manifest = _load(MANIFEST)
    training = _load(TRAINING_REPORT)
    summary = _load(SUMMARY)

    assert manifest["relation_key"] == "organizing::toy"
    train = manifest["train"]
    held_out = manifest["held_out"]
    train_seeds = {int(item["seed"]) for item in train}
    heldout_seeds = {int(item["seed"]) for item in held_out}
    assert heldout_seeds == HELDOUT_SEEDS
    assert not train_seeds & heldout_seeds
    assert len(train) == int(manifest["training_episode_count"]) == 52
    assert sum(int(item["samples"]) for item in train) == int(
        manifest["training_sample_count"]
    ) == 19261

    recovery = [item for item in train if item.get("source") == "dagger_recovery"]
    assert {int(item["seed"]) for item in recovery} == RECOVERY_SEEDS
    assert len(recovery) == 16
    repeat_counts = Counter((int(item["seed"]), item["episode"]) for item in recovery)
    assert set(repeat_counts.values()) == {2}
    assert manifest["dagger"]["recovery_seeds"] == sorted(RECOVERY_SEEDS)
    assert int(manifest["dagger"]["unique_recovery_episodes"]) == 8
    assert int(manifest["dagger"]["unique_recovery_samples"]) == 3522
    assert int(manifest["dagger"]["weighted_recovery_samples"]) == 7044

    recovery_samples = 0
    for episode in sorted({str(item["episode"]) for item in recovery}):
        expected = next(
            int(item["samples"]) for item in recovery if str(item["episode"]) == episode
        )
        with np.load(episode) as data:
            actions = np.asarray(data["actions"])
            oracle = np.asarray(data["oracle_actions"])
            assert actions.shape == oracle.shape == (expected, 7)
            assert np.isfinite(actions).all()
            assert np.array_equal(actions, oracle)
            recovery_samples += expected
    assert recovery_samples == 3522

    assert int(training["epochs"]) == 1
    assert int(training["completed_updates"]) == int(training["planned_updates"]) == 19261
    assert Path(training["initial_adapter"]).resolve() == (
        ROOT / "models" / "openvla-organizing-toy-lora-augmented-r2"
    ).resolve()
    assert set(map(int, training["heldout_seeds"])) == HELDOUT_SEEDS
    assert not set(map(int, training["train_seeds"])) & HELDOUT_SEEDS
    assert training["manifest_sha256"] == _sha256(MANIFEST)

    adapter_sha256 = adapter_fingerprint(ADAPTER)
    manifest_sha256 = _sha256(MANIFEST)
    results = []
    total_decisions = 0
    for seed in sorted(HELDOUT_SEEDS):
        report_path = EVAL_ROOT / f"seed_{seed}" / "report.json"
        report = _load(report_path)
        assert report["schema_version"] == "formal_skill_hybrid_model_evaluation_v1"
        assert report["evidence_scope"] == "formal_skill_hybrid_model_evaluation"
        assert report["relation_key"] == "organizing::toy"
        assert report["split"] == "held_out"
        assert int(report["seed"]) == seed
        assert int(report["max_decision_steps"]) == 300
        assert report["success"] is True
        assert all(report["success_predicates"].get(name) is True for name in PREDICATES)
        assert report["ever_grasped"] is True
        assert report["ever_inside"] is True
        assert report["counts_toward_task2_coverage"] is True
        assert report["counts_toward_autonomous_vla_acceptance"] is False
        assert report["adapter_sha256"] == adapter_sha256
        assert report["manifest_sha256"] == manifest_sha256

        executed = int(report["executed_decision_steps"])
        trajectory = report["trajectory"]
        assert 1 <= executed <= 300
        assert len(trajectory) == executed
        modes = Counter(str(step["execution_mode"]) for step in trajectory)
        assert dict(modes) == report["execution_mode_counts"]
        assert modes["final_contact_servo"] > 0
        assert sum(
            count
            for mode, count in modes.items()
            if mode.startswith("state_aware_locate_recovery_")
        ) > 0
        for step in trajectory:
            action = np.asarray(step["executed_action"], dtype=np.float64)
            assert action.shape == (7,)
            assert np.isfinite(action).all()
            assert np.allclose(action[3:6], 0.0, atol=1e-7)
        total_decisions += executed
        results.append(
            {
                "seed": seed,
                "executed_decision_steps": executed,
                "success": True,
                "native_success_predicates": True,
            }
        )

    assert summary["aggregate"]["successful_seeds"] == 3
    assert summary["aggregate"]["evaluated_seeds"] == 3
    assert summary["aggregate"]["success_rate"] == 1.0
    assert summary["aggregate"]["total_executed_decision_steps"] == total_decisions
    assert summary["aggregate"]["counts_toward_task2_coverage"] is True
    assert summary["aggregate"]["counts_toward_autonomous_vla_acceptance"] is False
    assert summary["evaluation_mode"]["pure_autonomous_vla"] is False
    assert summary["evaluation_mode"]["state_aware_locate_recovery"] is True
    assert summary["evaluation_mode"]["final_contact_servo"] is True
    assert summary["model"]["adapter_sha256"] == adapter_sha256
    assert summary["data_isolation"]["manifest_sha256"] == manifest_sha256

    return {
        "status": "PASS",
        "relation_key": "organizing::toy",
        "training": {
            "epochs": 1,
            "completed_updates": 19261,
            "planned_updates": 19261,
            "train_heldout_overlap": [],
            "recovery_seeds": sorted(RECOVERY_SEEDS),
            "oracle_labeled_recovery_samples": recovery_samples,
        },
        "evaluation": {
            "mode": "hybrid_closed_loop",
            "pure_autonomous_vla": False,
            "state_aware_locate_recovery": True,
            "final_contact_servo": True,
            "successful_seeds": 3,
            "evaluated_seeds": 3,
            "total_executed_decision_steps": total_decisions,
            "results": results,
        },
        "fingerprints": {
            "manifest_sha256": manifest_sha256,
            "adapter_sha256": adapter_sha256,
        },
    }


def main() -> int:
    print(json.dumps(verify(), ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
