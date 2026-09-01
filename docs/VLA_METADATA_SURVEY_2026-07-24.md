# VLA 数据集元数据普查

日期：2026-07-24

## 结论

- 本地 BridgeData V2：60064 条 episode，1152 个 TFRecord 分片。
- 本地 RoboCasa 1.0.1 源码：提取 378 个任务类、198 个正式对象类别。
- 数据驱动任务候选：取原生复合任务目录中任务数最多的 15 类。
- 对象候选：按任务实际引用频次优先、正式对象注册表补齐，当前输出 120 个。
- 分类不使用任务二、Excel 或参考文档第五章中的类别名称。

## 15 类任务候选（数据集原生类别）

| 排名 | 原生任务类别 | 任务数 | 示例任务 |
|---:|---|---:|---|
| 1 | `washing_dishes` | 19 | washing_dishes::adjust_water_temperature、washing_dishes::clear_sink、washing_dishes::collect_washing_supplies、washing_dishes::divide_basins、washing_dishes::dry_dishes |
| 2 | `setting_the_table` | 13 | setting_the_table::align_silverware、setting_the_table::arrange_bread_basket、setting_the_table::arrange_bread_bowl、setting_the_table::arrange_drinkware、setting_the_table::beverage_organization |
| 3 | `washing_fruits_and_vegetables` | 11 | washing_fruits_and_vegetables::afterwash_sorting、washing_fruits_and_vegetables::air_dry_fruit、washing_fruits_and_vegetables::clear_clutter、washing_fruits_and_vegetables::clear_sink_space、washing_fruits_and_vegetables::drain_veggies |
| 4 | `boiling` | 8 | boiling::boil_corn、boiling::boil_eggs、boiling::boil_pot、boiling::cool_kettle、boiling::fill_kettle |
| 5 | `clearing_table` | 8 | clearing_table::bowl_and_cup、clearing_table::candle_cleanup、clearing_table::clear_receptacles_for_cleaning、clearing_table::cluster_items_for_clearing、clearing_table::condiment_collection |
| 6 | `frying` | 8 | frying::assemble_cooking_array、frying::distribute_steak_on_pans、frying::frying_pan_adjustment、frying::meal_prep_staging、frying::press_chicken |
| 7 | `loading_fridge` | 8 | loading_fridge::create_child_friendly_fridge、loading_fridge::load_condiments_in_fridge、loading_fridge::load_fridge_by_type、loading_fridge::load_fridge_fifo、loading_fridge::load_prepared_food |
| 8 | `managing_freezer_space` | 8 | managing_freezer_space::clear_freezer、managing_freezer_space::freeze_bottled_waters、managing_freezer_space::freeze_ice_tray、managing_freezer_space::maximize_freezer_space、managing_freezer_space::move_fridge_to_freezer |
| 9 | `restocking_supplies` | 8 | restocking_supplies::beverage_sorting、restocking_supplies::fresh_produce_organization、restocking_supplies::refill_condiment_station、restocking_supplies::restock_bowls、restocking_supplies::restock_canned_food |
| 10 | `baking` | 7 | baking::cookie_dough_prep、baking::cool_baked_cake、baking::cool_baked_cookies、baking::cupcake_cleanup、baking::mix_cake_frosting |
| 11 | `portioning_meals` | 7 | portioning_meals::distribute_chicken、portioning_meals::portion_fruit_bowl、portioning_meals::portion_hot_dogs、portioning_meals::portion_in_tupperware、portioning_meals::portion_on_size |
| 12 | `sauteing_vegetables` | 7 | sauteing_vegetables::adjust_heat、sauteing_vegetables::butter_on_pan、sauteing_vegetables::place_vegetables_evenly、sauteing_vegetables::preheat_pot、sauteing_vegetables::shake_pan |
| 13 | `toasting_bread` | 7 | toasting_bread::get_toasted_bread、toasting_bread::p_j_sandwich_prep、toasting_bread::serve_warm_croissant、toasting_bread::toast_bagel、toasting_bread::toast_baguette |
| 14 | `brewing` | 6 | brewing::arrange_tea、brewing::deliver_brewed_coffee、brewing::kettle_boiling、brewing::organize_coffee_condiments、brewing::prepare_coffee |
| 15 | `chopping_food` | 6 | chopping_food::arrange_cutting_fruits、chopping_food::arrange_vegetables、chopping_food::bread_setup_slicing、chopping_food::clear_cutting_board、chopping_food::meat_transfer |

## 120 物体候选

| 排名 | 对象类别 | 精确任务引用 | 对象组可选 | 合计覆盖 | 证据 |
|---:|---|---:|---:|---:|---|
| 1 | `bowl` | 63 | 1 | 64 | `official_registry` |
| 2 | `plate` | 58 | 0 | 58 | `official_registry` |
| 3 | `mug` | 24 | 1 | 25 | `official_registry` |
| 4 | `cutting_board` | 22 | 0 | 22 | `official_registry` |
| 5 | `pan` | 20 | 0 | 20 | `official_registry` |
| 6 | `cup` | 17 | 1 | 18 | `official_registry` |
| 7 | `glass_cup` | 14 | 1 | 15 | `official_registry` |
| 8 | `tupperware` | 13 | 0 | 13 | `official_registry` |
| 9 | `knife` | 12 | 7 | 19 | `official_registry` |
| 10 | `cheese` | 10 | 1 | 11 | `official_registry` |
| 11 | `pot` | 10 | 0 | 10 | `official_registry` |
| 12 | `tray` | 10 | 0 | 10 | `official_registry` |
| 13 | `sponge` | 9 | 5 | 14 | `official_registry` |
| 14 | `bread` | 8 | 7 | 15 | `official_registry` |
| 15 | `fork` | 8 | 7 | 15 | `official_registry` |
| 16 | `spoon` | 8 | 7 | 15 | `official_registry` |
| 17 | `sugar_cube` | 8 | 4 | 12 | `official_registry` |
| 18 | `saucepan` | 8 | 0 | 8 | `official_registry` |
| 19 | `onion` | 7 | 63 | 70 | `official_registry` |
| 20 | `fish` | 7 | 40 | 47 | `official_registry` |
| 21 | `tomato` | 6 | 63 | 69 | `official_registry` |
| 22 | `steak` | 6 | 40 | 46 | `official_registry` |
| 23 | `baguette` | 6 | 7 | 13 | `official_registry` |
| 24 | `sandwich_bread` | 6 | 7 | 13 | `official_registry` |
| 25 | `shaker` | 6 | 6 | 12 | `official_registry` |
| 26 | `milk` | 6 | 4 | 10 | `official_registry` |
| 27 | `ice_cube` | 6 | 0 | 6 | `official_registry` |
| 28 | `kettle_non_electric` | 6 | 0 | 6 | `official_registry` |
| 29 | `saucepan_with_lid` | 6 | 0 | 6 | `official_registry` |
| 30 | `chicken_drumstick` | 5 | 40 | 45 | `official_registry` |
| 31 | `cake` | 5 | 4 | 9 | `official_registry` |
| 32 | `jam` | 5 | 2 | 7 | `official_registry` |
| 33 | `butter_stick` | 5 | 1 | 6 | `official_registry` |
| 34 | `oven_tray` | 5 | 0 | 5 | `official_registry` |
| 35 | `pitcher` | 5 | 0 | 5 | `official_registry` |
| 36 | `garlic` | 4 | 63 | 67 | `official_registry` |
| 37 | `lemon_wedge` | 4 | 63 | 67 | `official_registry` |
| 38 | `sausage` | 4 | 40 | 44 | `official_registry` |
| 39 | `condiment_bottle` | 4 | 8 | 12 | `official_registry` |
| 40 | `ketchup` | 4 | 8 | 12 | `official_registry` |
| 41 | `tongs` | 4 | 6 | 10 | `official_registry` |
| 42 | `cupcake` | 4 | 4 | 8 | `official_registry` |
| 43 | `croissant` | 4 | 1 | 5 | `official_registry` |
| 44 | `salt_and_pepper_shaker` | 4 | 0 | 4 | `official_registry` |
| 45 | `bell_pepper` | 3 | 63 | 66 | `official_registry` |
| 46 | `carrot` | 3 | 63 | 66 | `official_registry` |
| 47 | `cucumber` | 3 | 63 | 66 | `official_registry` |
| 48 | `spatula` | 3 | 7 | 10 | `official_registry` |
| 49 | `wooden_spoon` | 3 | 7 | 10 | `official_registry` |
| 50 | `bottled_water` | 3 | 3 | 6 | `official_registry` |
| 51 | `yogurt` | 3 | 3 | 6 | `official_registry` |
| 52 | `boxed_food` | 3 | 2 | 5 | `official_registry` |
| 53 | `canned_food` | 3 | 2 | 5 | `official_registry` |
| 54 | `egg` | 3 | 1 | 4 | `official_registry` |
| 55 | `colander` | 3 | 0 | 3 | `official_registry` |
| 56 | `avocado` | 2 | 63 | 65 | `official_registry` |
| 57 | `broccoli` | 2 | 63 | 65 | `official_registry` |
| 58 | `lemon` | 2 | 63 | 65 | `official_registry` |
| 59 | `lettuce` | 2 | 63 | 65 | `official_registry` |
| 60 | `lime` | 2 | 63 | 65 | `official_registry` |
| 61 | `strawberry` | 2 | 40 | 42 | `official_registry` |
| 62 | `kebab_skewer` | 2 | 13 | 15 | `official_registry` |
| 63 | `dish_brush` | 2 | 11 | 13 | `official_registry` |
| 64 | `mayonnaise` | 2 | 8 | 10 | `official_registry` |
| 65 | `syrup_bottle` | 2 | 8 | 10 | `official_registry` |
| 66 | `hotdog_bun` | 2 | 7 | 9 | `official_registry` |
| 67 | `digital_scale` | 2 | 6 | 8 | `official_registry` |
| 68 | `peeler` | 2 | 6 | 8 | `official_registry` |
| 69 | `spray` | 2 | 5 | 7 | `official_registry` |
| 70 | `chocolate` | 2 | 4 | 6 | `official_registry` |
| 71 | `pancake` | 2 | 4 | 6 | `official_registry` |
| 72 | `waffle` | 2 | 4 | 6 | `official_registry` |
| 73 | `bottled_drink` | 2 | 3 | 5 | `official_registry` |
| 74 | `can` | 2 | 3 | 5 | `official_registry` |
| 75 | `juice` | 2 | 3 | 5 | `official_registry` |
| 76 | `honey_bottle` | 2 | 2 | 4 | `official_registry` |
| 77 | `cream_cheese_stick` | 2 | 1 | 3 | `official_registry` |
| 78 | `blender_jug` | 2 | 0 | 2 | `official_registry` |
| 79 | `cinnamon` | 2 | 0 | 2 | `official_registry` |
| 80 | `ice_cube_tray` | 2 | 0 | 2 | `official_registry` |
| 81 | `straw` | 2 | 0 | 2 | `official_registry` |
| 82 | `wine_glass` | 2 | 0 | 2 | `official_registry` |
| 83 | `chili_pepper` | 1 | 63 | 64 | `official_registry` |
| 84 | `corn` | 1 | 63 | 64 | `official_registry` |
| 85 | `pickle_slice` | 1 | 63 | 64 | `official_registry` |
| 86 | `potato` | 1 | 63 | 64 | `official_registry` |
| 87 | `tomato_slice` | 1 | 63 | 64 | `official_registry` |
| 88 | `asparagus` | 1 | 62 | 63 | `official_registry` |
| 89 | `dumpling` | 1 | 48 | 49 | `official_registry` |
| 90 | `banana` | 1 | 40 | 41 | `official_registry` |
| 91 | `cherry` | 1 | 40 | 41 | `official_registry` |
| 92 | `shrimp` | 1 | 40 | 41 | `official_registry` |
| 93 | `turkey_slice` | 1 | 40 | 41 | `official_registry` |
| 94 | `pizza` | 1 | 13 | 14 | `official_registry` |
| 95 | `vinegar` | 1 | 10 | 11 | `official_registry` |
| 96 | `liquor` | 1 | 8 | 9 | `official_registry` |
| 97 | `mustard` | 1 | 8 | 9 | `official_registry` |
| 98 | `oil_and_vinegar_bottle` | 1 | 8 | 9 | `official_registry` |
| 99 | `wine` | 1 | 8 | 9 | `official_registry` |
| 100 | `bagel` | 1 | 7 | 8 | `official_registry` |
| 101 | `bread_flat` | 1 | 7 | 8 | `official_registry` |
| 102 | `ladle` | 1 | 7 | 8 | `official_registry` |
| 103 | `aluminum_foil` | 1 | 6 | 7 | `official_registry` |
| 104 | `cheese_grater` | 1 | 6 | 7 | `official_registry` |
| 105 | `pizza_cutter` | 1 | 6 | 7 | `official_registry` |
| 106 | `reamer` | 1 | 6 | 7 | `official_registry` |
| 107 | `rolling_pin` | 1 | 6 | 7 | `official_registry` |
| 108 | `strainer` | 1 | 6 | 7 | `official_registry` |
| 109 | `donut` | 1 | 5 | 6 | `official_registry` |
| 110 | `soap_dispenser` | 1 | 5 | 6 | `official_registry` |
| 111 | `cookie_dough_ball` | 1 | 4 | 5 | `official_registry` |
| 112 | `marshmallow` | 1 | 4 | 5 | `official_registry` |
| 113 | `bar` | 1 | 2 | 3 | `official_registry` |
| 114 | `candle` | 1 | 2 | 3 | `official_registry` |
| 115 | `cereal` | 1 | 2 | 3 | `official_registry` |
| 116 | `chips` | 1 | 2 | 3 | `official_registry` |
| 117 | `peanut_butter` | 1 | 2 | 3 | `official_registry` |
| 118 | `basket` | 1 | 0 | 1 | `official_registry` |
| 119 | `flour_bag` | 1 | 0 | 1 | `official_registry` |
| 120 | `paprika` | 1 | 0 | 1 | `official_registry` |

## BridgeData V2 本地结构

- 路径：`C:\OpenVLA-Simulator\datasets\oxe\bridge_orig\1.0.0`
- 语言字段：`featuresDict.features.episode_metadata.featuresDict.features.has_language`, `featuresDict.features.episode_metadata.featuresDict.features.has_language.description`, `featuresDict.features.episode_metadata.featuresDict.features.has_language.pythonClassName`, `featuresDict.features.episode_metadata.featuresDict.features.has_language.tensor`, `featuresDict.features.episode_metadata.featuresDict.features.has_language.tensor.dtype`, `featuresDict.features.episode_metadata.featuresDict.features.has_language.tensor.encoding`, `featuresDict.features.episode_metadata.featuresDict.features.has_language.tensor.shape`, `featuresDict.features.steps.sequence.feature.featuresDict.features.language_embedding`, `featuresDict.features.steps.sequence.feature.featuresDict.features.language_embedding.description`, `featuresDict.features.steps.sequence.feature.featuresDict.features.language_embedding.pythonClassName`, `featuresDict.features.steps.sequence.feature.featuresDict.features.language_embedding.tensor`, `featuresDict.features.steps.sequence.feature.featuresDict.features.language_embedding.tensor.dtype`, `featuresDict.features.steps.sequence.feature.featuresDict.features.language_embedding.tensor.encoding`, `featuresDict.features.steps.sequence.feature.featuresDict.features.language_embedding.tensor.shape`, `featuresDict.features.steps.sequence.feature.featuresDict.features.language_embedding.tensor.shape.dimensions`, `featuresDict.features.steps.sequence.feature.featuresDict.features.language_instruction`, `featuresDict.features.steps.sequence.feature.featuresDict.features.language_instruction.description`, `featuresDict.features.steps.sequence.feature.featuresDict.features.language_instruction.pythonClassName`, `featuresDict.features.steps.sequence.feature.featuresDict.features.language_instruction.text`
- `train`：53192 episodes，1024 shards，117.08 GB
- `val`：6872 episodes，128 shards，15.40 GB

## 其他官方数据源登记

| 数据集 | 官方任务/轨迹规模 | 本轮使用方式 | 任务级元数据状态 |
|---|---|---|---|
| robocasa | 365 tasks; RoboCasa365 includes 65 atomic and 300 composite task definitions | primary simulated household task and object taxonomy | local official source registry parsed |
| bridge_v2 | large real-robot kitchen and household manipulation corpus | local real-robot trajectory and language validation | local TFDS schema parsed; episode language sampling pending |
| libero | 130 tasks in four suites | candidate metadata-only supplement for household goals and objects | official task manifest retrieval pending |
| droid | 76k trajectories, 350 hours, 564 scenes | candidate real-world language and scene diversity supplement | aggregate registered; language metadata retrieval pending |
| robomind | 107k real trajectories, 479 tasks, 96 object classes | candidate real-world task and object supplement | aggregate registered; official metadata retrieval pending |
| rh20t | 110k+ contact-rich sequences across 147 task definitions | candidate contact-rich and human-video/robot-action supplement | aggregate registered; task descriptions retrieval pending |
| behavior_1k | 1,000 household activities and 10k+ objects | candidate high-level household goal and object-scope supplement | aggregate registered; BDDL task metadata retrieval pending |

## 限制与下一步

- RoboCasa 对象注册项是官方可生成对象类别；精确引用与对象组可选均来自源码静态配置，不等同于完整轨迹帧频。
- BridgeData 已由隔离的轻量 TFRecord 读取器抽样真实语言指令，未安装完整 TensorFlow；详见 `bridge_language_sample_summary.json`。
- 其他数据集先拉任务表/语言元数据，只有确认能补足当前覆盖缺口后才下载轨迹子集。
- 训练验收必须继续按 held-out seed 和仿真成功谓词执行，不能用元数据覆盖数代替动作成功率。
