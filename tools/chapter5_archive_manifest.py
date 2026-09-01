"""Create an evidence-bounded archive manifest for Chapter 5 sections 5.2.2–5.2.15."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.build_chapter5_midterm_revision_v2 import TASKS, VLA


NETWORK_TASKS = {"5.2.2", "5.2.3", "5.2.4"}
ALIASES = {
    "桌面": ["tabletop", "table"], "置物架": ["shelf"], "地面": ["floor"], "抹布": ["cloth"],
    "海绵": ["sponge"], "喷雾瓶": ["spray bottle"], "扫帚": ["broom"], "清洁刷": ["brush"],
    "碗": ["bowl"], "砧板": ["chopping board", "cutting board"], "餐刀": ["knife"], "锅": ["pot"],
    "平底锅": ["frying pan"], "炉灶": ["stove"], "烤箱": ["oven"], "烹饪托盘": ["tray"],
    "电钻": ["drill"], "钳子": ["pliers"], "螺丝刀": ["screwdriver"], "内六角扳手": ["allen wrench"],
    "手电筒": ["flashlight"], "工具箱": ["toolbox"], "海报": ["poster"], "墙钉": ["nail"],
    "水杯": ["cup"], "马克杯": ["mug"], "水瓶": ["water bottle"], "罐装物": ["can"],
    "书籍": ["book"], "托盘": ["tray"], "收纳容器": ["container", "tupperware"], "递送盒": ["box"],
    "鞋子": ["gym shoes", "sandals", "shoe"], "鞋架": ["shoe rack", "hallstand"], "书架": ["bookcase"],
    "文件夹": ["folder"], "笔记本": ["notebook"], "钢笔": ["pen"], "笔盒": ["pencil case"],
    "收纳盒": ["box"], "橱柜门": ["cabinet door"], "抽屉": ["drawer"], "冰箱门": ["fridge door"],
    "冰箱抽屉": ["fridge drawer"], "微波炉门": ["microwave door"], "烤箱门": ["oven door"],
    "洗碗机门": ["dishwasher door"], "烤面包机烤箱门": ["toaster oven door"],
    "微波炉启动键": ["microwave"], "微波炉停止键": ["microwave"], "烤面包机压杆": ["toaster"],
    "电热水壶开关": ["kettle"], "搅拌机电源键": ["blender"], "炉灶旋钮": ["stove"],
    "烤面包机烤箱定时器": ["toaster oven"], "水龙头手柄": ["faucet"], "盘子": ["plate"],
    "滤盆": ["colander"], "汤勺": ["ladle"], "木勺": ["wooden spoon"], "披萨刀": ["pizza cutter"],
    "洗碗刷": ["dish brush"], "夹子": ["tongs"], "铝箔纸": ["aluminum foil"], "棒球帽": ["baseball cap"],
    "运动裤": ["training pants"], "毛巾": ["towel"], "洗衣篮": ["basket"], "洗衣机": ["washer"],
    "拖鞋": ["slipper"], "电脑": ["computer"], "显示器": ["monitor"], "键盘": ["keyboard"],
    "鼠标": ["mouse"], "笔记本电脑": ["laptop"], "订书机": ["stapler"], "铅笔": ["pencil"],
    "数据线": ["cable"], "洗涤剂": ["detergent"], "卫生巾盒": ["sanitary napkin"],
    "皂液器": ["soap dispenser"], "漱口杯": ["cup"], "牙膏": ["toothpaste"], "牙刷": ["toothbrush"],
    "卫浴置物架": ["bathroom shelf"], "捕鼠器": ["mousetrap"],
}


def _network_objects(object_text: str) -> list[str]:
    return [item.strip() for item in object_text.split("、")]


def _match_candidate(object_name: str, candidates: list[dict[str, Any]], used_episodes: set[int]) -> dict[str, Any] | None:
    aliases = ALIASES.get(object_name, [])
    for alias in aliases:
        for row in candidates:
            if row.get("dataset") != "RoboCasa365 official" or row.get("episode_index") is None:
                continue
            episode = int(row["episode_index"])
            if episode in used_episodes:
                continue
            if alias.lower() in str(row.get("instruction", "")).lower():
                used_episodes.add(episode)
                return row
    return None


def build_manifest(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Return 14 task records and 112 evidence-separated object records."""
    tasks = [row for row in TASKS if row[0] != "5.2.1"]
    used_episodes: set[int] = set()
    objects: list[dict[str, Any]] = []
    for code, task_name, route, object_text, boundary, difficulty in tasks:
        if code in NETWORK_TASKS:
            for name in _network_objects(object_text):
                objects.append({
                    "section": code, "task_name": task_name, "object_name": name,
                    "archive_kind": "network_interface", "source_evidence": None,
                    "conversion_status": "not_vla_training", "acceptance_boundary": boundary,
                })
            continue
        for name, object_type, operation in VLA[code]:
            evidence = _match_candidate(name, candidates, used_episodes)
            objects.append({
                "section": code, "task_name": task_name, "object_name": name,
                "object_type": object_type, "operation": operation,
                "archive_kind": "vla_training" if evidence else "pending_source_evidence",
                "source_evidence": evidence, "conversion_status": "ready_to_derive" if evidence else "not_archived",
                "acceptance_boundary": boundary,
            })
    return {
        "scope": "Chapter 5 sections 5.2.2–5.2.15", "tasks": [
            {"section": r[0], "task_name": r[1], "route": r[2], "difficulty": r[5]} for r in tasks
        ], "objects": objects,
    }


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_manifest(json.loads(args.candidates.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"tasks": len(manifest["tasks"]), "objects": len(manifest["objects"]), "ready": sum(x["archive_kind"] == "vla_training" for x in manifest["objects"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
