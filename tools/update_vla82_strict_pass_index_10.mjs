import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const [input, output, previewPath] = process.argv.slice(2);
const wb = await SpreadsheetFile.importXlsx(await FileBlob.load(input));
const sheet = wb.worksheets.getItemAt(0);
const root = "C:\\OpenVLA-Simulator\\outputs\\midterm_testing_vla82";
const demo = `${root}\\full_simulation_objectwise_8x60\\expert_demos`;

sheet.getRange("A15:K15").values = [[
  10,
  "VLA82-027",
  "整理收纳",
  "冰箱抽屉",
  "关闭冰箱下层抽屉",
  "接近 → 接触抽屉前沿 → 持续推合 → 达到关闭阈值",
  "PASS",
  "通过",
  `${root}\\clean_motion_regeneration_20260813\\strict_pass_clear_view\\VLA82-027\\episode-2000.strict-pass-drawer-close-clear-final-720p.mp4`,
  `${demo}\\VLA82-027\\episode-2000.json`,
  `${demo}\\VLA82-027\\episode-2000.npz`,
]];
sheet.getRange("B2").formulas = [["=COUNTIF(G6:G15,\"PASS\")"]];
sheet.getRange("B3").formulas = [["=COUNTIF(H6:H15,\"通过\")"]];
sheet.getRange("A15:K15").format.wrapText = true;
sheet.getRange("A15:H15").format.verticalAlignment = "center";

const tableCheck = await wb.inspect({
  kind: "table", range: `${sheet.name}!A1:K15`, include: "values,formulas",
  tableMaxRows: 20, tableMaxCols: 12, maxChars: 16000,
});
console.log(tableCheck.ndjson);
const errors = await wb.inspect({
  kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: {useRegex: true, maxResults: 100}, summary: "formula errors",
});
console.log(errors.ndjson);
const preview = await wb.render({sheetName: sheet.name, range: "A1:K15", scale: 1, format: "png"});
await fs.writeFile(previewPath, new Uint8Array(await preview.arrayBuffer()));
const out = await SpreadsheetFile.exportXlsx(wb);
await out.save(output);
