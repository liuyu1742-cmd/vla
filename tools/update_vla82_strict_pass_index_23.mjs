import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const [input, output, previewPath] = process.argv.slice(2);
const wb = await SpreadsheetFile.importXlsx(await FileBlob.load(input));
const sheet = wb.worksheets.getItemAt(0);

sheet.getRange("A27:K27").copyTo(sheet.getRange("A28:K28"), "all");
sheet.getRange("A28:K28").values = [[
  23,
  "VLA82-058",
  "卫浴用品与个人卫生服务",
  "牙刷",
  "从台面夹取牙刷并放入漱口杯",
  "双夹持垫抓取 → 抬升越过杯沿 → 杯口正上方对中 → 竖直插入杯内 → 受控释放 → 杯内稳定停留",
  "PASS",
  "通过（1280×720，正向无遮挡；连续夹持、目标接触、受控释放、稳定停留）",
  "C:\\OpenVLA-Simulator\\outputs\\strict_pass_videos_720p_20260828\\VLA82-058_牙刷放入漱口杯_严格PASS_720p正向无遮挡.mp4",
  "C:\\OpenVLA-Simulator\\outputs\\midterm_testing_vla82\\vla058_toothbrush_cup_256p_20260828_attempt3\\expert_demos\\VLA82-058\\episode-2005.json",
  "C:\\OpenVLA-Simulator\\outputs\\midterm_testing_vla82\\vla058_toothbrush_cup_256p_20260828_attempt3\\expert_demos\\VLA82-058\\episode-2005.npz",
]];
sheet.getRange("B2").formulas = [["=COUNTIF(G6:G28,\"PASS\")"]];
sheet.getRange("B3").formulas = [["=COUNTA(H6:H28)"]];
sheet.getRange("A28:K28").format.wrapText = true;
sheet.getRange("A28:H28").format.verticalAlignment = "center";

const tableCheck = await wb.inspect({
  kind: "table",
  range: `${sheet.name}!A1:K28`,
  include: "values,formulas",
  tableMaxRows: 30,
  tableMaxCols: 11,
  maxChars: 24000,
});
console.log(tableCheck.ndjson);
const errors = await wb.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "final formula error scan",
});
console.log(errors.ndjson);
const preview = await wb.render({
  sheetName: sheet.name,
  range: "A1:K28",
  scale: 1,
  format: "png",
});
await fs.writeFile(previewPath, new Uint8Array(await preview.arrayBuffer()));
const out = await SpreadsheetFile.exportXlsx(wb);
await out.save(output);
