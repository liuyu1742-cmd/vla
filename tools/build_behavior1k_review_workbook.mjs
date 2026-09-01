import fs from "node:fs/promises";
import path from "node:path";
import { Workbook, SpreadsheetFile } from "@oai/artifact-tool";

const root = "C:/OpenVLA-Simulator";
const metadataDir = path.join(root, "datasets/behavior_1k/2025_challenge_metadata");
const outputDir = path.join(root, "outputs/whole_home_vla_coverage");
const outputFile = path.join(outputDir, "BEHAVIOR-1K_15任务_122物体_审阅表.xlsx");
const C = [
  ["household_organization","家庭物品整理","将散放物品分类、归位到明确容器或位置。",[7,18,21,22,23,29]],
  ["item_delivery_placement","物品递送与放置","将指定物品在房间之间取送、放置；包含打开并使用收纳空间。",[13,14,16,17]],
  ["waste_disposal","垃圾处理","识别垃圾并投放到垃圾桶。",[1]],
  ["meal_cleanup","餐具与餐后清理","收集餐具、剩余食物并放入水槽、柜体或冰箱。",[3,11,25,47]],
  ["food_storage","食品储存与食材管理","将食品、饮料和干货放入容器、冰箱或柜体。",[12,24]],
  ["cooking_service","烹饪、加热与饮品服务","完成食材分拣、切配、加热、烹饪及饮品相关操作。",[4,20,40,41,42,43,45,46,48,49]],
  ["appliance_operation","家用电器操作","对小型家电进行取放、摆位或开关前后的操作。",[8]],
  ["surface_cleaning","表面清洁与物件养护","使用清洁工具去除物件表面污渍或灰尘。",[37]],
  ["laundry_care","洗衣与织物护理","把可洗物品放入洗衣机并完成清洗。",[31,32,33]],
  ["repair_tool_use","居家维修与工具使用","取用、收纳工具，完成悬挂或器件安装。",[19,34,35]],
  ["interior_layout","室内陈设与功能区布置","在固定台面上布置可用的家居功能区。",[10]],
  ["bathroom_service","卫浴用品服务","将卫生用品放到浴室的水槽、置物架或指定容器。",[27]],
  ["workspace_service","工作与学习设备服务","布置、归位和连接书桌上的办公学习设备与文具。",[28]],
  ["entertainment_service","家庭娱乐设备服务","识别并操作客厅中的娱乐设备。",[0]],
  ["pest_control","虫害防治与卫生维护","取放和布置害虫防治物品。",[5]],
];
const G = {
household_organization:[
["board_game","桌游",7,"拾取后放入玩具箱或书柜"],["jigsaw_puzzle","拼图",7,"拾取后放入玩具箱"],["tennis_ball","网球",7,"拾取后放入玩具箱"],["toy_box","玩具箱",7,"打开或定位后作为玩具收纳目标"],["dice","骰子",21,"拾取后放入书柜"],["teddy_bear","泰迪熊玩偶",21,"从地面或床面拾取并放入书柜"],["bookcase","书柜",21,"打开/定位格层后放入书籍或玩具"],["gym_shoe","运动鞋",22,"拾取并成对放到鞋架"],["hallstand","鞋架",22,"作为鞋类放置目标"],["book","书籍",23,"从书柜取下或放入收纳箱"],["box","纸箱",23,"作为书籍收纳容器"]],
item_delivery_placement:[
["digital_camera","数码相机",13,"拾取并放入容器或带到指定位置"],["container","收纳容器",13,"装入物品后搬运并放置"],["tennis_racket","网球拍",13,"拾取并放入汽车后备箱"],["car_trunk","汽车后备箱",13,"打开后放入物品并关闭"],["sack_of_groceries","杂货袋",14,"从后备箱取出并搬运至厨房"],["tomato","番茄",14,"从杂货袋取出并放入冰箱"],["carton_of_milk","牛奶盒",14,"从杂货袋取出并放入冰箱"],["water_bottle","饮用水瓶",17,"从冰箱取出并放到茶几"],["coffee_table","茶几",17,"作为饮用水放置目标"],["storage_container","储物箱",16,"搬运到指定空间并堆叠放置"]],
waste_disposal:[["can_of_soda","汽水罐",1,"识别为垃圾后拾取并投放"],["trash_can","垃圾桶",1,"打开或定位后接收投放物"]],
meal_cleanup:[
["pizza","披萨",3,"连同盘子拾取并放入冰箱"],["plate","盘子",3,"拾取后放入水槽或柜体"],["refrigerator","冰箱",3,"打开、放入食品后关闭"],["bowl","碗",3,"拾取后放入水槽"],["sink","水槽",3,"作为餐具放置/清洗目标"],["half_chicken","半只鸡",25,"装入保鲜盒后冷藏"],["half_apple_pie","半个苹果派",25,"装入保鲜盒后冷藏"],["tupperware_container","保鲜盒",25,"打开、装入食物并放入冰箱"],["kitchen_cabinet","厨房柜体",11,"打开后收纳餐盘并关闭"]],
food_storage:[
["apple_half","苹果半块",12,"拾取并装入餐盒"],["club_sandwich","俱乐部三明治",12,"拾取并装入餐盒"],["chocolate_chip_cookie","巧克力曲奇",12,"拾取并装入餐盒"],["packing_box","餐盒",12,"作为食品和饮料收纳容器"],["bottle_of_tea","茶饮瓶",12,"从冰箱取出并放入餐盒"],["oatmeal_box","燕麦盒",24,"从台面拾取并放入柜体"],["chips_bag","薯片袋",24,"从台面拾取并放入柜体"],["olive_oil_bottle","橄榄油瓶",24,"从台面拾取并放入柜体"],["sugar_jar","糖罐",24,"从台面拾取并放入柜体"]],
cooking_service:[
["hinged_jar","带铰链玻璃罐",4,"打开、装入食物、关闭并放回柜体"],["cooked_bratwurst","熟香肠",4,"从砧板拾取并放入玻璃罐"],["chopping_board","砧板",4,"作为食材切配承载面"],["countertop","操作台",4,"作为取放、切配和布置区域"],["bok_choy","小白菜",20,"拾取后按类别放入搅拌碗"],["vidalia_onion","维达利亚洋葱",20,"分类、切配并放入容器"],["mixing_bowl","搅拌碗",20,"承接分类后的蔬菜"],["leek","韭葱",20,"按类别拾取并放入搅拌碗"],["broccoli","西兰花",20,"按类别拾取并放入搅拌碗"],["sweet_corn","甜玉米",20,"按类别拾取并放入搅拌碗"],["microwave","微波炉",40,"打开、放入食品、加热后取出/确认"],["popcorn_bag","爆米花袋",40,"放入微波炉并加热"],["cabbage","卷心菜",41,"从冰箱取出、切配并下锅"],["chili","辣椒",41,"从冰箱取出、切配并下锅"],["knife","刀具",41,"抓取后沿砧板执行切配动作"],["frying_pan","煎锅",41,"放置食材并在炉灶上加热"],["stove","炉灶",41,"承载煎锅并执行烹饪加热"],["bell_pepper","甜椒",43,"从冰箱取出后切丁"],["beet","甜菜根",43,"从冰箱取出后切丁"],["zucchini","西葫芦",43,"从冰箱取出后切丁"],["parer","削皮刀",43,"抓取后对蔬菜执行切配"],["hot_dog","热狗",45,"从冰箱取出并放入微波炉加热"],["bacon","培根",46,"从冰箱取出并在煎锅烹饪"],["tray","托盘",46,"承载培根并用于取放"],["steak","牛排",48,"切配后分装进碗"],["pineapple","菠萝",48,"切配后分装进碗"],["carving_knife","切肉刀",48,"抓取后对食材执行切配"],["grated_cheese","碎奶酪",49,"从保鲜盒取出后铺放到披萨面团"],["pepperoni","意式香肠片",49,"从保鲜盒取出后铺放到披萨"],["mushroom","蘑菇",49,"切半后铺放到披萨"],["pizza_dough","披萨面团",49,"作为配料铺放与烘烤基底"],["cookie_sheet","烤盘",49,"承载披萨并送入烤箱"],["oven","烤箱",49,"放入烤盘并执行烘烤"]],
appliance_operation:[["toaster","烤面包机",8,"拾取后放入柜体或摆放到指定台面"],["food_processor","食品料理机",8,"拾取后放入柜体或摆放到指定台面"],["french_press","法压壶",8,"拾取后放入柜体或摆放到指定台面"]],
surface_cleaning:[["scrub_brush","刷子",37,"抓取后沿物体表面往复刷洗"],["cornet","短号",37,"固定/扶持并对刷洗区域进行清洁"]],
laundry_care:[["boxing_glove","拳击手套",31,"拾取后放入洗衣机清洗"],["washer","洗衣机",31,"打开、放入可洗物并启动/确认清洗"],["baseball_cap","棒球帽",32,"拾取后放入洗衣机清洗"],["teddy_toy","泰迪玩具",33,"从柜体取出并放入洗衣机清洗"],["softball","垒球",33,"从柜体取出并放入洗衣机清洗"]],
repair_tool_use:[["drill","电钻",19,"从工具箱取用或放回，供维修使用"],["pliers","钳子",19,"从工具箱取用或放回"],["flashlight","手电筒",19,"从工具箱取用或放回"],["allen_wrench","内六角扳手",19,"从工具箱取用或放回"],["screwdriver","螺丝刀",19,"从工具箱取用或放回"],["toolbox","工具箱",19,"打开、收纳工具并关闭"],["poster","海报",34,"拾取后对齐墙面并悬挂"],["wall_nail","墙钉",34,"作为海报悬挂定位点"],["camera_tripod","相机三脚架",35,"与相机对准并完成安装连接"]],
interior_layout:[["coffee_maker","咖啡机",10,"定位在台面并作为功能区中心摆放"],["bottle_of_coffee","咖啡瓶",10,"从架子取下并放到咖啡机旁"],["paper_coffee_filter","咖啡滤纸",10,"放置到咖啡机顶部/滤杯位置"],["saucer","茶托",10,"放到咖啡机旁并承接咖啡杯"],["coffee_cup","咖啡杯",10,"放到茶托上并调整位置"],["electric_kettle","电热水壶",10,"放到咖啡机旁完成服务区布置"],["kitchen_shelf","厨房置物架",10,"作为咖啡瓶取放来源位置"]],
bathroom_service:[["detergent_bottle","洗涤剂瓶",27,"放到浴室水槽下方并并列摆放"],["sanitary_napkin_box","卫生巾盒",27,"放到浴室置物架"],["bathroom_shelf","浴室置物架",27,"作为卫生用品放置目标"],["soap_dispenser","皂液器",27,"放到浴室水槽上"],["bathroom_cup","浴室杯子",27,"保持在水槽上并收纳牙刷牙膏"],["toothpaste_tube","牙膏管",27,"放入浴室杯子"],["toothbrush","牙刷",27,"放入浴室杯子"]],
workspace_service:[["computer","电脑",28,"放到书桌下方或指定位置"],["desk","书桌",28,"作为设备与文具布置目标"],["monitor","显示器",28,"放到书桌上并对齐"],["keyboard","键盘",28,"放到显示器旁"],["mouse","鼠标",28,"放到键盘旁"],["folder","文件夹",28,"从椅子取下并放到书桌"],["swivel_chair","转椅",28,"移动到书桌旁"],["notebook","笔记本",28,"叠放到文件夹上"],["pen","钢笔",28,"放到笔记本上"],["pencil","铅笔",29,"放入笔袋"],["pencil_case","笔袋",29,"收纳铅笔与钢笔"],["stapler","订书机",29,"从书柜取出并放到书桌"],["laptop","笔记本电脑",29,"从床上移到书桌并合上"]],
entertainment_service:[["radio_receiver","收音机接收器",0,"定位后执行开机/关机操作"]],
pest_control:[["mousetrap","捕鼠器",5,"从柜体取出后放到浴室地面指定位置"]],
};

const tasks = (await fs.readFile(path.join(metadataDir, "tasks.jsonl"), "utf8")).trim().split(/\r?\n/).map(JSON.parse);
const taskMap = new Map(tasks.map(t => [t.task_index,t]));
const rows = [];
for (const [taskId, objects] of Object.entries(G)) {
  const cat = C.find(c => c[0] === taskId);
  for (const [objectId, objectZh, taskIndex, operation] of objects) {
    const source = taskMap.get(taskIndex);
    if (!source) throw new Error("unknown task "+taskIndex);
    rows.push([taskId,cat[1],objectId,objectZh,operation,taskIndex,source.task_name,source.task,"BEHAVIOR-1K：23维动作/状态Parquet + 头部RGB视频","下载中；完成后核验"]);
  }
}
if(rows.length < 120) throw new Error("need 120 objects, got "+rows.length);
if(new Set(rows.map(r => r[2])).size !== rows.length) throw new Error("duplicate object id");

const wb = Workbook.create();
const summary = wb.worksheets.add("总览");
const taskSheet = wb.worksheets.add("15类任务");
const objectSheet = wb.worksheets.add("任务物体操作表");
const sourceSheet = wb.worksheets.add("数据证据");
for (const sh of [summary,taskSheet,objectSheet,sourceSheet]) sh.showGridLines=false;

summary.getRange("A1:J1").merge();
summary.getRange("A1").values=[["BEHAVIOR-1K 家庭 VLA：15 类任务与 122 种物体（审阅版）"]];
summary.getRange("A1:J1").format={fill:"#0B5E8E",font:{bold:true,color:"#FFFFFF",size:16},horizontalAlignment:"center"};
summary.getRange("A3:B7").values=[["指标","当前值"],["任务类别数",C.length],["物体种类数",rows.length],["数据源","BEHAVIOR-1K 2025 Challenge Demos"],["训练输入","RGB头部视频 + 23维动作/状态 + 英文任务指令"]];
summary.getRange("D3:J7").values=[["审阅说明","","","","","",""],["每个物体均附真实源任务、英文原始指令和机器人操作方式。","","","","","",""],["任务按家政服务结果划分，避免按儿童、车库或节庆重复拆分。","","","","","",""],["下载完成后，数据状态会更新为已核验并进入 OpenVLA 转换。","","","","","",""],["此表为审阅版；不会把尚未下载的对象伪装成已训练数据。","","","","","",""]];
summary.getRange("A3:B3").format={fill:"#D9EAF7",font:{bold:true,color:"#0B3D5C"}};
summary.getRange("D3:J3").format={fill:"#D9EAF7",font:{bold:true,color:"#0B3D5C"}};
summary.getRange("A3:B7").format.borders={preset:"all",style:"thin",color:"#D9E2F3"};
summary.getRange("D3:J7").format.wrapText=true;
summary.getRange("A:A").format.columnWidth=20;summary.getRange("B:B").format.columnWidth=28;summary.getRange("D:D").format.columnWidth=32;summary.getRange("E:J").format.columnWidth=15;

const taskRows=C.map(c=>[c[0],c[1],c[2],c[3].join(","),c[3].map(i=>taskMap.get(i).task_name).join("; "),c[3].length*20,"下载中；每原始任务计划20条","BEHAVIOR-1K 2025"]);
taskSheet.getRange("A1:H"+(taskRows.length+1)).values=[["task_id","任务名称","机器人可交付结果","源任务索引","真实源任务名称","计划示教数","数据状态","来源"],...taskRows];
taskSheet.tables.add("A1:H"+(taskRows.length+1),true,"TasksTable");
taskSheet.freezePanes.freezeRows(1);

objectSheet.getRange("A1:J"+(rows.length+1)).values=[["task_id","任务名称","object","物体中文名","机器人如何操作","源任务索引","真实源任务","原始英文任务指令","RGB/动作证据","数据状态"],...rows];
objectSheet.tables.add("A1:J"+(rows.length+1),true,"ObjectsTable");
objectSheet.freezePanes.freezeRows(1);objectSheet.freezePanes.freezeColumns(2);
objectSheet.getRange("A2:J"+(rows.length+1)).format.wrapText=true;
objectSheet.getRange("J2:J"+(rows.length+1)).conditionalFormats.add("containsText",{text:"下载中",format:{fill:"#FFF2CC",font:{color:"#7F6000"}}});

sourceSheet.getRange("A1:H6").values=[
["数据源","本地路径/公开地址","示教与模态","动作","任务证据","当前用途","转换说明","状态"],
["BEHAVIOR-1K 2025","datasets/behavior_1k/2025_challenge_household_subset","1,000条计划下载；每条Parquet+头部RGB","23维float32","tasks.jsonl；50项真实英文任务","本表15类/122物体的主证据","LeRobot Parquet+MP4 → OpenVLA训练样本","下载中"],
["BEHAVIOR-1K元数据","datasets/behavior_1k/2025_challenge_metadata","10,000 episode元数据；50任务","模式说明见info.json","tasks.jsonl + episodes.jsonl","任务、物体、指令可追溯","不需要转换","已下载"],
["BridgeData V2","datasets/oxe/bridge_orig/1.0.0","真实机器人操作数据","OXE/RLDS","本地元数据普查","OpenVLA基础数据","可按OXE/RLDS读取","已下载"],
["CMU Stretch","datasets/oxe/cmu_stretch_lerobot","室内机器人示教","8维动作","135 episodes","补充门柜等交互","LeRobot → 训练样本","已下载"],
["UTokyo PR2 fridge","datasets/oxe/utokyo_pr2_opening_fridge_lerobot","冰箱开合示教","8维动作","80 episodes","补充冰箱交互","LeRobot → 训练样本","已下载"]];
sourceSheet.tables.add("A1:H6",true,"SourcesTable");
sourceSheet.getRange("A2:H6").format.wrapText=true;
sourceSheet.getRange("H2").conditionalFormats.add("containsText",{text:"下载中",format:{fill:"#FFF2CC",font:{color:"#7F6000"}}});

for (const sh of [taskSheet,objectSheet,sourceSheet]) {sh.getUsedRange().format.autofitRows();}
taskSheet.getRange("A:A").format.columnWidth=29;taskSheet.getRange("B:B").format.columnWidth=22;taskSheet.getRange("C:C").format.columnWidth=34;taskSheet.getRange("E:E").format.columnWidth=52;taskSheet.getRange("G:H").format.columnWidth=25;
objectSheet.getRange("A:A").format.columnWidth=27;objectSheet.getRange("B:B").format.columnWidth=20;objectSheet.getRange("C:C").format.columnWidth=22;objectSheet.getRange("D:D").format.columnWidth=18;objectSheet.getRange("E:E").format.columnWidth=34;objectSheet.getRange("F:F").format.columnWidth=11;objectSheet.getRange("G:G").format.columnWidth=34;objectSheet.getRange("H:H").format.columnWidth=70;objectSheet.getRange("I:I").format.columnWidth=38;objectSheet.getRange("J:J").format.columnWidth=20;
sourceSheet.getRange("A:A").format.columnWidth=25;sourceSheet.getRange("B:B").format.columnWidth=52;sourceSheet.getRange("C:C").format.columnWidth=37;sourceSheet.getRange("D:D").format.columnWidth=16;sourceSheet.getRange("E:G").format.columnWidth=30;sourceSheet.getRange("H:H").format.columnWidth=18;

console.log((await wb.inspect({kind:"table",range:"任务物体操作表!A1:J6",include:"values",tableMaxRows:6,tableMaxCols:10})).ndjson);
console.log((await wb.inspect({kind:"match",searchTerm:"#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",options:{useRegex:true,maxResults:50},summary:"formula errors"})).ndjson);
await fs.mkdir(outputDir,{recursive:true});
for(const entry of [["总览","A1:J8","preview_summary.png"],["15类任务","A1:H16","preview_tasks.png"],["任务物体操作表","A1:J18","preview_objects.png"],["数据证据","A1:H6","preview_sources.png"]]){
 const image=await wb.render({sheetName:entry[0],range:entry[1],scale:1.2,format:"png"});
 await fs.writeFile(path.join(outputDir,entry[2]),new Uint8Array(await image.arrayBuffer()));
}
const output=await SpreadsheetFile.exportXlsx(wb);
await output.save(outputFile);
console.log(JSON.stringify({outputFile,objects:rows.length,categories:C.length}));

