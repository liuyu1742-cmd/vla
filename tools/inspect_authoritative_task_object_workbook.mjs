import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const sourcePath = "C:/RobotProject/RobotProject/docs/家庭服务任务物体数据表.xlsx";
const input = await FileBlob.load(sourcePath);
const workbook = await SpreadsheetFile.importXlsx(input);
const sheets = await workbook.inspect({
  kind: "sheet,table",
  include: "id,name,values",
  maxChars: 20000,
  tableMaxRows: 145,
  tableMaxCols: 9,
  tableMaxCellChars: 80,
});
console.log(sheets.ndjson);
