import fs from "node:fs/promises";

const planPath =
  "C:/OpenVLA-Simulator/outputs/midterm_testing_vla82/simulator_mapping_plan.json";
const plan = JSON.parse(await fs.readFile(planPath, "utf8"));
const plate = plan.mappings.find((row) => row.selection_id === "VLA82-044");

if (!plate) {
  throw new Error("VLA82-044 mapping is missing");
}

Object.assign(plate, {
  mapping_mode: "functional_proxy",
  task_class: "PickPlaceCounterToSink",
  object_group: "bowl",
  manipulated_object_text: "bowl",
  mapping_note:
    "真实盘子的视频与标注证据保持不变；当前RoboCasa的plate资产连续三次均无满足任务尺寸约束的可采样类别，仿真接口改用形态和容器功能接近的bowl功能代理，不作为真实盘子闭环成功证明",
  proxy_disclosed: true,
  validation_status: "PENDING_RETEST",
});

plan.mode_counts = plan.mappings.reduce((counts, row) => {
  counts[row.mapping_mode] = (counts[row.mapping_mode] ?? 0) + 1;
  return counts;
}, {});
plan.repair_revision = {
  version: 3,
  reason:
    "盘子原生资产在入柜及入水槽任务中连续三次无满足尺寸约束的可采样类别；改用明确披露的bowl功能代理验证仿真接口",
  repaired_selection_ids: ["VLA82-043", "VLA82-044", "VLA82-059"],
};

await fs.writeFile(
  planPath,
  `${JSON.stringify(plan, null, 2)}\n`,
  "utf8",
);
console.log("UPDATED: simulator_mapping_plan.json revision 3");
