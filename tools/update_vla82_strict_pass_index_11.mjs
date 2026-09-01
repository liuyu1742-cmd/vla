import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const [input, output, previewPath] = process.argv.slice(2);
const wb = await SpreadsheetFile.importXlsx(await FileBlob.load(input));
const sheet = wb.worksheets.getItemAt(0);
const root = "C:\\OpenVLA-Simulator\\outputs\\midterm_testing_vla82";
const demo = `${root}\\full_simulation_objectwise_8x60\\expert_demos`;

// Preserve the established table styling and then append the newly verified item.
sheet.getRange("A15:K15").copyTo(sheet.getRange("A16:K16"), "all");
sheet.getRange("A16:K16").values = [[
  11,
  "VLA82-014",
  "烹饪与加热辅助",
  "量杯",
  "从台面抓取量杯并放入抽屉",
  "抓取 → 姿态调整 → 搬运 → 抽屉内受控释放",
  "PASS",
  "通过",
  `${root}\\clean_motion_regeneration_20260814\\strict_pass_clear_view\\VLA82-014\\episode-2000.strict-pass-controlled-drawer-placement-720p.mp4`,
  `${demo}\\VLA82-014\\episode-2000.json`,
  `${demo}\\VLA82-014\\episode-2000.npz`,
]];
sheet.getRange("B2").formulas = [["=COUNTIF(G6:G16,\"PASS\")"]];
sheet.getRange("B3").formulas = [["=COUNTIF(H6:H16,\"通过\")"]];
sheet.getRange("A16:K16").format.wrapText = true;
sheet.getRange("A16:H16").format.verticalAlignment = "center";

const tableCheck = await wb.inspect({
  kind: "table", range: `${sheet.name}!A1:K16`, include: "values,formulas",
  tableMaxRows: 20, tableMaxCols: 12, maxChars: 18000,
});
console.log(tableCheck.ndjson);
const errors = await wb.inspect({
  kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 }, summary: "formula errors",
});
console.log(errors.ndjson);
const preview = await wb.render({ sheetName: sheet.name, range: "A1:K16", scale: 1, format: "png" });
await fs.writeFile(previewPath, new Uint8Array(await preview.arrayBuffer()));
const out = await SpreadsheetFile.exportXlsx(wb);
await out.save(output);
