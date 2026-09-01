import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const root = "C:/OpenVLA-Simulator";
const outputDir = `${root}/outputs/vla_88_docx_table_reorganized`;
const inputPath = `${outputDir}/VLA_88物体标签_按表5-8-1至5-8-11整理.xlsx`;
const dataPath = `${outputDir}/dataset_video_labels.json`;
const outputPath = `${outputDir}/VLA_88物体标签_实际数据路径与真实视频标注.xlsx`;

const source = JSON.parse(await fs.readFile(dataPath, "utf8"));
if (source.missing.length !== 0 || source.records.length !== 88) {
  throw new Error(`Dataset mapping incomplete: ${source.records.length} records, ${source.missing.length} missing`);
}
const byKey = new Map(source.records.map((record) => [`${record.table_no}:${record.source_row}`, record]));

const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(inputPath));
const summary = workbook.worksheets.getItem("汇总");

summary.getRange("A1:G1").unmerge();
summary.getRange("A1:I1").merge();
summary.getRange("A2:G2").unmerge();
summary.getRange("A2:I2").merge();
summary.getRange("A1").values = [["VLA 88 个物体操作标签（实际数据路径与真实视频标注）"]];
summary.getRange("A2").values = [[
  "数据依据：C:\\OpenVLA-Simulator\\datasets\\chapter5_5_2_5 到 chapter5_5_2_15 的实际物体目录；表号仅用于分组，来源列改为实际路径。",
]];
summary.getRange("A18:I18").values = [[
  "表号", "源表标题", "源表序号", "任务/技能名称", "操作对象", "表内操作标签", "实际数据路径", "真实的视频标注", "视频文件与标注来源",
]];
summary.getRange("G20:I107").values = source.records.map((record) => [
  record.actual_path,
  record.real_video_label,
  `${record.video_files.join("、")}；标注来源：${record.real_label_source}；状态：${record.conversion_status}`,
]);
summary.getRange("A1:I1").format = {
  fill: "#1F4E78",
  font: { bold: true, color: "#FFFFFF", size: 16 },
  horizontalAlignment: "center",
  verticalAlignment: "center",
};
summary.getRange("A2:I2").format = {
  fill: "#EAF2F8",
  font: { color: "#1F2937", italic: true, size: 10 },
  wrapText: true,
  verticalAlignment: "center",
};
summary.getRange("A18:I18").format = {
  fill: "#5B9BD5",
  font: { bold: true, color: "#FFFFFF" },
  horizontalAlignment: "center",
  verticalAlignment: "center",
  wrapText: true,
};
summary.getRange("A18:I107").format.borders = { preset: "all", style: "thin", color: "#D9E2F3" };
summary.getRange("A18:I18").format.rowHeight = 34;
summary.getRange("A20:I107").format.rowHeight = 52;
summary.getRange("G20:I107").format.wrapText = true;
summary.getRange("G:G").format.columnWidth = 58;
summary.getRange("H:H").format.columnWidth = 42;
summary.getRange("I:I").format.columnWidth = 42;
summary.tables.items[0].delete();
summary.tables.add("A18:I107", true, "VLA88SummaryDatasetTable").style = "TableStyleMedium2";

for (const tableNo of Array.from({ length: 11 }, (_, i) => i + 1)) {
  const sheet = workbook.worksheets.getItem(`表5-8-${tableNo}`);
  const usedEndRow = sheet.getUsedRange().rowCount + 1;
  const endRow = usedEndRow - 1;
  sheet.getRange("A1:E1").unmerge();
  sheet.getRange("A1:H1").merge();
  sheet.getRange("A2:E2").unmerge();
  sheet.getRange("A2:H2").merge();
  sheet.getRange("A1").values = [[`表5-8-${tableNo}（实际数据路径与真实视频标注）`]];
  sheet.getRange("A2").values = [[
    "操作类型为源表标签；实际数据路径、真实视频标注和视频文件均来自 datasets 下对应物体目录。",
  ]];
  sheet.getRange("A4:H4").values = [["序号", "技能名称", "操作对象", "操作类型", "来源表", "实际数据路径", "真实的视频标注", "视频文件与标注来源"]];
  const data = [];
  for (let row = 5; row <= endRow; row += 1) {
    const sourceRow = sheet.getRange(`A${row}`).values[0][0];
    const objectName = sheet.getRange(`C${row}`).values[0][0];
    const record = byKey.get(`${tableNo}:${sourceRow}`);
    if (!record || record.object_name !== objectName) {
      throw new Error(`Dataset row mismatch on 表5-8-${tableNo}, row ${row}, object ${objectName}`);
    }
    data.push([
      sourceRow,
      sheet.getRange(`B${row}`).values[0][0],
      objectName,
      sheet.getRange(`D${row}`).values[0][0],
      `表5-8-${tableNo}`,
      record.actual_path,
      record.real_video_label,
      `${record.video_files.join("、")}；标注来源：${record.real_label_source}；状态：${record.conversion_status}`,
    ]);
  }
  sheet.getRange(`A5:H${endRow}`).values = data;
  sheet.getRange("A1:H1").format = {
    fill: "#1F4E78",
    font: { bold: true, color: "#FFFFFF", size: 14 },
    horizontalAlignment: "center",
    verticalAlignment: "center",
  };
  sheet.getRange("A2:H2").format = {
    fill: "#EAF2F8",
    font: { color: "#1F2937", italic: true, size: 10 },
    wrapText: true,
    verticalAlignment: "center",
  };
  sheet.getRange("A4:H4").format = {
    fill: "#5B9BD5",
    font: { bold: true, color: "#FFFFFF" },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
  };
  sheet.getRange(`A4:H${endRow}`).format.borders = { preset: "all", style: "thin", color: "#D9E2F3" };
  sheet.getRange("A4:H4").format.rowHeight = 34;
  sheet.getRange(`A5:H${endRow}`).format.rowHeight = 52;
  sheet.getRange(`F5:H${endRow}`).format.wrapText = true;
  sheet.getRange("F:F").format.columnWidth = 58;
  sheet.getRange("G:G").format.columnWidth = 42;
  sheet.getRange("H:H").format.columnWidth = 42;
  sheet.tables.items[0].delete();
  sheet.tables.add(`A4:H${endRow}`, true, `VLA88Dataset_${tableNo}`).style = "TableStyleMedium2";
}

const check = await workbook.inspect({
  kind: "table",
  range: "汇总!A1:I25",
  include: "values,formulas",
  tableMaxRows: 25,
  tableMaxCols: 9,
  tableMaxCellChars: 120,
});
console.log("DATASET_SUMMARY_CHECK");
console.log(check.ndjson);
const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "final formula error scan",
});
console.log("FORMULA_ERRORS");
console.log(errors.ndjson);

const summaryPreview = await workbook.render({ sheetName: "汇总", range: "A1:I25", scale: 1, format: "png" });
await fs.writeFile(`${outputDir}/preview_dataset_summary.png`, new Uint8Array(await summaryPreview.arrayBuffer()));
for (const tableNo of Array.from({ length: 11 }, (_, i) => i + 1)) {
  const preview = await workbook.render({ sheetName: `表5-8-${tableNo}`, autoCrop: "all", scale: 1, format: "png" });
  await fs.writeFile(`${outputDir}/preview_dataset_${tableNo}.png`, new Uint8Array(await preview.arrayBuffer()));
}

const xlsx = await SpreadsheetFile.exportXlsx(workbook);
await xlsx.save(outputPath);
console.log(`EXPORTED ${outputPath}`);
