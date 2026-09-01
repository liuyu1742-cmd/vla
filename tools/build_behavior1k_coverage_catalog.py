"""Build an evidence-backed 15-task / 120-object catalog from BEHAVIOR-1K."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


CATEGORIES = {
    "living_room_appliance_management": ("客厅设备管理", [0]),
    "waste_and_general_storage": ("垃圾处理与通用收纳", [1, 16, 23]),
    "seasonal_decor_organization": ("节庆装饰整理", [2, 9, 26]),
    "dining_cleanup_and_dish_storage": ("餐后清理与餐具收纳", [3, 11, 25, 47]),
    "bathroom_personal_care_organization": ("浴室与个人用品整理", [5, 27]),
    "garden_outdoor_maintenance": ("花园与户外维护", [6, 15, 36, 38, 39, 44]),
    "children_room_toy_desk_organization": ("儿童房玩具与书桌整理", [7, 21, 29]),
    "kitchen_appliance_and_beverage_setup": ("厨房电器与饮品布置", [8, 10, 40]),
    "garage_transport_and_grocery_transfer": ("车库搬运与采购转运", [13, 14]),
    "bedroom_entryway_tidy": ("卧室与玄关整理", [18, 22]),
    "tool_and_home_repair": ("工具收纳与居家维修", [19, 34, 35, 37]),
    "workstation_organization": ("居家工作区整理", [28]),
    "laundry_and_washable_item_cleaning": ("洗衣与可洗物品清洁", [31, 32, 33]),
    "food_storage_and_preparation": ("食品收纳与烹饪准备", [4, 12, 17, 20, 24, 41, 42, 43, 45, 46, 48, 49]),
    "home_safety_and_fireplace_care": ("家庭安全与壁炉维护", [30]),
}

# object_id|Chinese name|source task index|verbatim phrase in the task instruction
OBJECT_LINES = """radio_receiver|收音机接收器|0|radio receiver
table|桌子|0|table
can_of_soda|汽水罐|1|can of soda
trash_can|垃圾桶|1|tash can
coffee_table|茶几|17|coffee table
book|书籍|18|book
newspaper|报纸|30|newspaper
wood_fireplace|燃木壁炉|30|wood fireplace
pumpkin|南瓜|2|pumpkins
candle|蜡烛|2|candles
cabinet|柜子|2|cabinet
cauldron|大锅|2|cauldron
wreath|花环|9|wreath
candy_cane|拐杖糖|9|candy canes
pillar_candle|柱状蜡烛|9|pillar candles
sofa|沙发|9|sofa
dining_table|餐桌|9|dining-room table
gift_box|礼品盒|9|gift boxes
pizza|披萨|3|pizzas
plate|盘子|3|plate
refrigerator|冰箱|3|refrigerator
bowl|碗|3|bowls
sink|水槽|3|sink
hinged_jar|带铰链玻璃罐|4|hinged jars
chopping_board|砧板|4|chopping board
countertop|操作台|4|countertop
toaster|烤面包机|8|toaster
coffee_maker|咖啡机|10|coffee maker
paper_coffee_filter|滤纸|10|paper coffee filter
saucer|茶托|10|saucer
coffee_cup|咖啡杯|10|coffee cup
electric_kettle|电热水壶|10|electric kettle
mousetrap|捕鼠器|5|mousetraps
bathroom_floor|浴室地面|5|bathroom floor
detergent_bottle|洗涤剂瓶|27|detergent bottles
sanitary_napkin_box|卫生巾盒|27|box of sanitary napkins
bathroom_shelf|浴室置物架|27|bathroom shelf
soap_dispenser|皂液器|27|soap dispenser
cup|杯子|27|cup
toothpaste_tube|牙膏管|27|toothpaste tube
toothbrush|牙刷|27|toothbrush
wicker_basket|柳条篮|6|wicker basket
easter_egg|复活节彩蛋|6|Easter eggs
lawn|草坪|6|lawn
tree|树木|6|tree
plywood_sheet|胶合板|15|plywood sheets
broom|扫帚|36|broom
garden_patio|庭院地面|36|patio floor
pesticide_atomizer|农药喷雾器|38|pesticide atomizer
potted_plant|盆栽植物|38|potted plants
axe|斧头|44|axe
log|原木|44|logs
chopping_block|劈柴墩|44|chopping block
driveway|车道|44|driveway
firewood|木柴|30|firewood
cigar_lighter|点火器|30|cigar lighter
board_game|桌游|7|board games
bed|床|7|bed
jigsaw_puzzle|拼图|7|jigsaw puzzles
tennis_ball|网球|7|tennis ball
toy_box|玩具箱|7|toy box
dice|骰子|21|dice
teddy_bear|泰迪熊玩偶|21|teddy bears
bookcase|书柜|21|bookcase
digital_camera|数码相机|13|digital camera
container|收纳容器|13|container
tennis_racket|网球拍|13|tennis racket
car_trunk|汽车后备箱|13|car trunk
sack_of_groceries|杂货袋|14|sack of groceries
tomato|番茄|14|tomato
carton_of_milk|牛奶盒|14|carton of milk
storage_container|储物箱|16|storage containers
nightstand|床头柜|18|nightstand
sandal|凉鞋|18|sandals
gym_shoe|运动鞋|22|gym shoes
hallstand|鞋架|22|hallstand
computer|电脑|28|computer
desk|书桌|28|desk
monitor|显示器|28|monitor
keyboard|键盘|28|keyboard
mouse|鼠标|28|mouse
folder|文件夹|28|folder
swivel_chair|转椅|28|swivel chair
notebook|笔记本|28|notebook
pen|钢笔|28|pen
pencil|铅笔|29|pencil
pencil_case|笔袋|29|pencil case
drill|电钻|19|drill
pliers|钳子|19|pliers
flashlight|手电筒|19|flashlight
allen_wrench|内六角扳手|19|Allen wrench
screwdriver|螺丝刀|19|screwdriver
toolbox|工具箱|19|toolbox
poster|海报|34|poster
wall_nail|墙钉|34|wall nails
camera_tripod|相机三脚架|35|camera tripod
scrub_brush|刷子|37|scrub brush
boxing_glove|拳击手套|31|boxing gloves
washer|洗衣机|31|washer
baseball_cap|棒球帽|32|baseball caps
teddy_toy|泰迪玩具|33|teddy toys
softball|垒球|33|softball
bok_choy|小白菜|20|bok choy
vidalia_onion|维达利亚洋葱|20|Vidalia onions
mixing_bowl|搅拌碗|20|mixing bowls
leek|韭葱|20|leeks
broccoli|西兰花|20|broccoli
sweet_corn|甜玉米|20|sweet corn
microwave|微波炉|40|microwave
cabbage|卷心菜|41|cabbage
chili|辣椒|41|chili
knife|刀|41|knife
frying_pan|煎锅|41|frying pan
stove|炉灶|41|stove
bell_pepper|甜椒|43|bell peppers
beet|甜菜根|43|beets
zucchini|西葫芦|43|zucchini
parer|削皮刀|43|parer
hot_dog|热狗|45|hot dogs
bacon|培根|46|bacon"""


def parse_objects() -> list[dict[str, object]]:
    objects = []
    for line in OBJECT_LINES.splitlines():
        object_id, object_zh, task_index, phrase = line.split("|", 3)
        objects.append({"object_id": object_id, "object_zh": object_zh, "source_task_index": int(task_index), "source_phrase": phrase})
    if len(objects) != 120 or len({row["object_id"] for row in objects}) != 120:
        raise ValueError("catalog must contain exactly 120 unique objects")
    return objects


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--subset-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    tasks = [json.loads(line) for line in (args.metadata_dir / "tasks.jsonl").read_text(encoding="utf-8").splitlines()]
    by_index = {task["task_index"]: task for task in tasks}
    task_to_category = {index: (category_id, category_zh) for category_id, (category_zh, indices) in CATEGORIES.items() for index in indices}
    if set(task_to_category) != set(by_index):
        raise ValueError("the 15 categories must cover every official source task exactly once")
    manifest = [json.loads(line) for line in args.subset_manifest.read_text(encoding="utf-8").splitlines()]
    demo_count = Counter(row["task_index"] for row in manifest)

    task_rows = []
    for category_id, (category_zh, indices) in CATEGORIES.items():
        task_rows.append({
            "task_id": category_id,
            "task_name_zh": category_zh,
            "source_task_count": len(indices),
            "source_task_indices": ",".join(str(index) for index in indices),
            "source_task_names": "; ".join(by_index[index]["task_name"] for index in indices),
            "demonstrations": sum(demo_count[index] for index in indices),
            "trainability": "ready_for_conversion",
            "evidence": "BEHAVIOR-1K 2025 paired Parquet action/state + RGB-head video subset",
        })

    object_rows = []
    for item in parse_objects():
        source_index = item["source_task_index"]
        category_id, category_zh = task_to_category[source_index]
        instruction = by_index[source_index]["task"]
        if str(item["source_phrase"]).casefold() not in instruction.casefold():
            raise ValueError(f"source phrase not found for {item['object_id']}: {item['source_phrase']}")
        object_rows.append({
            "task_id": category_id, "task_name_zh": category_zh,
            "object_id": item["object_id"], "object_zh": item["object_zh"],
            "source_task_index": source_index, "source_task_name": by_index[source_index]["task_name"],
            "source_phrase": item["source_phrase"], "demonstrations": demo_count[source_index],
            "paired_rgb_videos": demo_count[source_index], "trainability": "ready_for_conversion",
            "evidence": f"BEHAVIOR-1K task-{source_index:04d}; phrase={item['source_phrase']}",
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "task_catalog.json").write_text(json.dumps(task_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "object_catalog.json").write_text(json.dumps(object_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    for filename, rows in (("task_catalog.csv", task_rows), ("object_catalog.csv", object_rows)):
        with (args.output_dir / filename).open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({"tasks": len(task_rows), "objects": len(object_rows), "demos": len(manifest)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

