import fs from "node:fs/promises";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const root = "C:/OpenVLA-Simulator";
const outputDir = `${root}/outputs/vla_88_docx_table_reorganized`;
const dataPath = `${outputDir}/table_5_8_1_to_5_8_11.json`;
const outputPath = `${outputDir}/VLA_88物体标签_按表5-8-1至5-8-11整理.xlsx`;

const source = JSON.parse(await fs.readFile(dataPath, "utf8"));
const { tables, records } = source;
if (tables.length !== 11 || records.length !== 88) {
  throw new Error(`Expected 11 source tables and 88 records, got ${tables.length} tables and ${records.length} records`);
}

const workbook = Workbook.create();
const summary = workbook.worksheets.add("汇总");
summary.showGridLines = false;

summary.getRange("A1:G1").merge();
summary.getRange("A1").values = [["VLA 88 个物体操作标签（按 Word 表 5-8-1 至 5-8-11 重整）"]];
summary.getRange("A2:G2").merge();
summary.getRange("A2").values = [[
  "数据来源：第五章（中期验收版_修订_分类与公式版）.docx；仅整理表 5-8-1 至 5-8-11，保留原表中的任务、物体和具体操作标签。",
]];
summary.getRange("A3:F3").values = [["总条目数", null, "源表数量", null, "整理口径", "Word 表 5-8-1 至 5-8-11"]];
summary.getRange("B3").formulas = [["=COUNTA(E20:E107)"]];
summary.getRange("D3").formulas = [["=COUNTA(A6:A16)"]];

summary.getRange("A5:C5").values = [["表号", "源表标题", "条目数"]];
summary.getRange("A6:B16").values = tables.map((table) => [
  `5-8-${table.table_no}`,
  table.title,
]);
summary.getRange("C6").formulas = [["=COUNTIF($A$20:$A$107,A6)"]];
summary.getRange("C6:C16").fillDown();

summary.getRange("A18:G18").values = [[
  "表号", "源表标题", "源表序号", "任务/技能名称", "操作对象", "具体操作/视频标签", "来源",
]];
summary.getRange("A20:G107").values = records.map((record) => [
  `5-8-${record.table_no}`,
  record.table_title,
  record.source_row,
  record.skill_name,
  record.object_name,
  record.operation_type,
  `第五章（中期验收版_修订_分类与公式版）.docx · 表5-8-${record.table_no}`,
]);

summary.getRange("A1:G1").format = {
  fill: "#1F4E78",
  font: { bold: true, color: "#FFFFFF", size: 16 },
  horizontalAlignment: "center",
  verticalAlignment: "center",
};
summary.getRange("A2:G2").format = {
  fill: "#EAF2F8",
  font: { color: "#1F2937", italic: true, size: 10 },
  wrapText: true,
  verticalAlignment: "center",
};
summary.getRange("A3:F3").format = {
  fill: "#F3F6FA",
  font: { bold: true, color: "#1F2937" },
  horizontalAlignment: "center",
  verticalAlignment: "center",
  wrapText: true,
};
summary.getRange("A5:C5").format = {
  fill: "#A9D18E",
  font: { bold: true, color: "#1F2937" },
  horizontalAlignment: "center",
};
summary.getRange("A18:G18").format = {
  fill: "#5B9BD5",
  font: { bold: true, color: "#FFFFFF" },
  horizontalAlignment: "center",
  verticalAlignment: "center",
  wrapText: true,
};
summary.getRange("A6:C16").format.borders = { preset: "all", style: "thin", color: "#D9E2F3" };
summary.getRange("A18:G107").format.borders = { preset: "all", style: "thin", color: "#D9E2F3" };
summary.getRange("A1:G1").format.rowHeight = 30;
summary.getRange("A2:G2").format.rowHeight = 32;
summary.getRange("A3:F3").format.rowHeight = 28;
summary.getRange("A5:C5").format.rowHeight = 24;
summary.getRange("A18:G18").format.rowHeight = 30;
summary.getRange("A6:C16").format.rowHeight = 24;
summary.getRange("A20:G107").format.rowHeight = 34;
summary.getRange("A20:C107").format.horizontalAlignment = "center";
summary.getRange("E20:F107").format.wrapText = true;
summary.getRange("A:A").format.columnWidth = 11;
summary.getRange("B:B").format.columnWidth = 34;
summary.getRange("C:C").format.columnWidth = 11;
summary.getRange("D:D").format.columnWidth = 22;
summary.getRange("E:E").format.columnWidth = 16;
summary.getRange("F:F").format.columnWidth = 34;
summary.getRange("G:G").format.columnWidth = 45;
summary.tables.add("A18:G107", true, "VLA88SummaryTable").style = "TableStyleMedium2";
summary.freezePanes.freezeRows(18);
summary.freezePanes.freezeColumns(5);

for (const table of tables) {
  const sheet = workbook.worksheets.add(`表5-8-${table.table_no}`);
  sheet.showGridLines = false;
  sheet.getRange("A1:E1").merge();
  sheet.getRange("A1").values = [[table.title]];
  sheet.getRange("A2:E2").merge();
  sheet.getRange("A2").values = [["字段顺序与 Word 源表一致；“操作类型”即物体在该表中的具体操作/视频标签。"]];
  sheet.getRange("A4:E4").values = [["序号", "技能名称", "操作对象", "操作类型", "来源表"]];
  sheet.getRange(`A5:E${4 + table.rows.length}`).values = table.rows.map((row) => [
    row.source_row,
    row.skill_name,
    row.object_name,
    row.operation_type,
    `表5-8-${table.table_no}`,
  ]);
  const endRow = 4 + table.rows.length;
  sheet.getRange("A1:E1").format = {
    fill: "#1F4E78",
    font: { bold: true, color: "#FFFFFF", size: 14 },
    horizontalAlignment: "center",
    verticalAlignment: "center",
  };
  sheet.getRange("A2:E2").format = {
    fill: "#EAF2F8",
    font: { color: "#1F2937", italic: true, size: 10 },
    wrapText: true,
    verticalAlignment: "center",
  };
  sheet.getRange("A4:E4").format = {
    fill: "#5B9BD5",
    font: { bold: true, color: "#FFFFFF" },
    horizontalAlignment: "center",
    verticalAlignment: "center",
  };
  sheet.getRange(`A4:E${endRow}`).format.borders = { preset: "all", style: "thin", color: "#D9E2F3" };
  sheet.getRange("A1:E1").format.rowHeight = 28;
  sheet.getRange("A2:E2").format.rowHeight = 26;
  sheet.getRange("A4:E4").format.rowHeight = 26;
  sheet.getRange(`A5:E${endRow}`).format.rowHeight = 30;
  sheet.getRange(`A5:A${endRow}`).format.horizontalAlignment = "center";
  sheet.getRange(`D5:D${endRow}`).format.wrapText = true;
  sheet.getRange(`A5:E${endRow}`).format.verticalAlignment = "center";
  sheet.getRange("A:A").format.columnWidth = 8;
  sheet.getRange("B:B").format.columnWidth = 24;
  sheet.getRange("C:C").format.columnWidth = 18;
  sheet.getRange("D:D").format.columnWidth = 38;
  sheet.getRange("E:E").format.columnWidth = 12;
  sheet.tables.add(`A4:E${endRow}`, true, `VLA5811_${table.table_no}`).style = "TableStyleMedium2";
  sheet.freezePanes.freezeRows(4);
}

const check = await workbook.inspect({
  kind: "table",
  range: "汇总!A1:G25",
  include: "values,formulas",
  tableMaxRows: 25,
  tableMaxCols: 7,
  tableMaxCellChars: 120,
});
console.log("SUMMARY_CHECK");
console.log(check.ndjson);
const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "final formula error scan",
});
console.log("FORMULA_ERRORS");
console.log(errors.ndjson);

const summaryPreview = await workbook.render({ sheetName: "汇总", range: "A1:G25", scale: 1, format: "png" });
await fs.writeFile(`${outputDir}/preview_summary.png`, new Uint8Array(await summaryPreview.arrayBuffer()));
for (const table of tables) {
  const sheetName = `表5-8-${table.table_no}`;
  const preview = await workbook.render({ sheetName, autoCrop: "all", scale: 1, format: "png" });
  await fs.writeFile(`${outputDir}/preview_${table.table_no}.png`, new Uint8Array(await preview.arrayBuffer()));
}

const xlsx = await SpreadsheetFile.exportXlsx(workbook);
await xlsx.save(outputPath);
console.log(`EXPORTED ${outputPath}`);
