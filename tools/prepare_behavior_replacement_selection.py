"""Select 44 non-duplicate, indoor BEHAVIOR-1K object samples for the strict 120 registry."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


SPECS = [
    ("chapter5_5_2_5_室内卫生清洁", "垃圾桶", 1, "将垃圾投放至垃圾桶"),
    ("chapter5_5_2_5_室内卫生清洁", "饮料罐", 1, "拾取并分类投放"),
    ("chapter5_5_2_6_烹饪与加热辅助", "砧板", 42, "取用并放入水槽"),
    ("chapter5_5_2_6_烹饪与加热辅助", "削皮刀", 42, "取用并放入水槽"),
    ("chapter5_5_2_6_烹饪与加热辅助", "餐刀", 42, "取用并放入水槽"),
    ("chapter5_5_2_6_烹饪与加热辅助", "平底锅", 41, "在炉灶上烹饪"),
    ("chapter5_5_2_6_烹饪与加热辅助", "混合碗", 20, "分类盛放食材"),
    ("chapter5_5_2_6_烹饪与加热辅助", "玻璃罐", 4, "开合并收纳食物"),
    ("chapter5_5_2_6_烹饪与加热辅助", "午餐盒", 12, "装入并收纳餐食"),
    ("chapter5_5_2_6_烹饪与加热辅助", "茶饮瓶", 12, "从冰箱取出并放入餐盒"),
    ("chapter5_5_2_6_烹饪与加热辅助", "苹果", 12, "装入餐盒"),
    ("chapter5_5_2_6_烹饪与加热辅助", "番茄", 14, "放入冰箱"),
    ("chapter5_5_2_11_家电按钮与旋钮操作", "收音机", 0, "按键开启"),
    ("chapter5_5_2_11_家电按钮与旋钮操作", "烤面包机", 8, "放入橱柜归位"),
    ("chapter5_5_2_11_家电按钮与旋钮操作", "食品加工机", 8, "放入橱柜归位"),
    ("chapter5_5_2_7_设备维护与工具使用", "电钻", 19, "放入工具箱"),
    ("chapter5_5_2_7_设备维护与工具使用", "钳子", 19, "放入工具箱"),
    ("chapter5_5_2_7_设备维护与工具使用", "手电筒", 19, "放入工具箱"),
    ("chapter5_5_2_7_设备维护与工具使用", "内六角扳手", 19, "放入工具箱"),
    ("chapter5_5_2_7_设备维护与工具使用", "螺丝刀", 19, "放入工具箱"),
    ("chapter5_5_2_7_设备维护与工具使用", "工具箱", 19, "收纳并合盖"),
    ("chapter5_5_2_8_物品递送", "书籍", 18, "从床边取用并放至床头柜"),
    ("chapter5_5_2_8_物品递送", "饮料瓶", 17, "从冰箱取出并递送到客厅"),
    ("chapter5_5_2_9_整理收纳", "收纳箱", 23, "装入书籍收纳"),
    ("chapter5_5_2_13_衣物鞋类与玄关归位", "鞋子", 22, "摆放至鞋架"),
    ("chapter5_5_2_13_衣物鞋类与玄关归位", "棒球帽", 32, "放入洗衣机清洗"),
    ("chapter5_5_2_7_设备维护与工具使用", "海报", 34, "悬挂至墙面"),
    ("chapter5_5_2_7_设备维护与工具使用", "墙钉", 34, "作为挂置固定点"),
    ("chapter5_5_2_7_设备维护与工具使用", "数码相机", 35, "安装到三脚架"),
    ("chapter5_5_2_7_设备维护与工具使用", "相机三脚架", 35, "安装相机"),
    ("chapter5_5_2_14_工作学习区服务", "电脑", 28, "归位至书桌下方"),
    ("chapter5_5_2_14_工作学习区服务", "显示器", 28, "摆放至书桌"),
    ("chapter5_5_2_14_工作学习区服务", "键盘", 28, "摆放至书桌"),
    ("chapter5_5_2_14_工作学习区服务", "鼠标", 28, "摆放至键盘旁"),
    ("chapter5_5_2_14_工作学习区服务", "文件夹", 28, "摆放至书桌"),
    ("chapter5_5_2_14_工作学习区服务", "笔记本", 28, "叠放于文件夹上"),
    ("chapter5_5_2_14_工作学习区服务", "办公椅", 28, "移动并归位至书桌旁"),
    ("chapter5_5_2_14_工作学习区服务", "订书机", 29, "摆放至书桌"),
    ("chapter5_5_2_14_工作学习区服务", "铅笔", 29, "放入笔盒"),
    ("chapter5_5_2_14_工作学习区服务", "笔盒", 29, "收纳文具"),
    ("chapter5_5_2_14_工作学习区服务", "笔记本电脑", 29, "摆放至书桌并合盖"),
    ("chapter5_5_2_15_卫浴用品与个人卫生服务", "洗涤剂", 27, "放至水槽下方收纳"),
    ("chapter5_5_2_15_卫浴用品与个人卫生服务", "卫生巾盒", 27, "摆放至卫浴置物架"),
    ("chapter5_5_2_15_卫浴用品与个人卫生服务", "牙膏", 27, "放入漱口杯"),
    ("chapter5_5_2_15_卫浴用品与个人卫生服务", "牙刷", 27, "放入漱口杯"),
]


def build_selection(manifest_path: Path) -> list[dict]:
    by_task = defaultdict(list)
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        by_task[int(row["task_index"])].append(row)
    offsets = defaultdict(int)
    result = []
    for task_dir, object_id, task_index, operation in SPECS:
        records = by_task[task_index]
        row = records[offsets[task_index]]
        offsets[task_index] += 1
        result.append({"task_dir": task_dir, "object_id": object_id, "episode_index": int(row["episode_index"]), "instruction": row["instruction"], "operation": operation})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = build_selection(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"objects": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
