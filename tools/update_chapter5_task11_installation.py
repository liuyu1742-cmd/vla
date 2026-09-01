"""Update the existing Chapter 5 document in place for the verified Task 5.2.11.

The change preserves the 15-category taxonomy: appliance button/knob actions
remain a support sample set of Task 5.2.1, while 5.2.11 becomes indoor
installation and layout using only four verified VLA training triples.
"""

from pathlib import Path

from docx import Document


DOC_DIR = Path(r"C:\OpenVLA-Simulator\outputs\chapter5_midterm_revision")

TASK11_OBJECT_ROWS = [
    ("室内安装与布置", "海报", "悬挂至墙面"),
    ("室内安装与布置", "墙钉", "作为挂置固定点"),
    ("室内安装与布置", "数码相机", "安装到三脚架"),
    ("室内安装与布置", "相机三脚架", "固定相机"),
]


def _set_paragraph_text(paragraph, text: str) -> None:
    """Replace visible text while retaining the first run's formatting."""
    if paragraph.runs:
        paragraph.runs[0].text = text
        for run in paragraph.runs[1:]:
            run.text = ""
    else:
        paragraph.add_run(text)


def _set_cell_text(cell, text: str) -> None:
    if cell.paragraphs:
        _set_paragraph_text(cell.paragraphs[0], text)
        for paragraph in cell.paragraphs[1:]:
            _set_paragraph_text(paragraph, "")
    else:
        cell.text = text


def _find_docx() -> Path:
    matches = list(DOC_DIR.glob("*.docx"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one docx in {DOC_DIR}, found {matches}")
    return matches[0]


def _replace_task11_section(document: Document) -> None:
    paragraphs = document.paragraphs
    for index, paragraph in enumerate(paragraphs):
        if paragraph.text.strip().startswith("5.2.11"):
            _set_paragraph_text(paragraph, "5.2.11 室内安装与布置")
            body = paragraphs[index + 1]
            _set_paragraph_text(
                body,
                "本节为替换原网络控制项后的VLA优先任务，面向室内轻量固定与设备布置。"
                "当前只纳入已完成逐项训练三元组核验的海报、墙钉、数码相机和相机三脚架："
                "每个对象均绑定真实任务指令、RGB视频和含机器人动作字段的Parquet标签。"
                "海报与墙钉对应墙面挂置场景；数码相机与相机三脚架对应相机安装场景。"
                "中期验收将动作收敛为单件取放、指定位置挂置或相机与三脚架连接，"
                "以目标位置、挂置状态或连接关系作为成功判据，不扩展为多部件装修流程。"
                "原“家电面板与按钮操作”的实体样本仍保留为5.2.1家电综合管理的现场执行支撑数据，"
                "但不再作为独立任务计数。",
            )
            return
    raise RuntimeError("Task 5.2.11 heading not found")


def _update_summary_table(document: Document) -> None:
    table = document.tables[0]
    for row in table.rows[1:]:
        if row.cells[0].text.strip() == "5.2.1":
            _set_cell_text(row.cells[3], "照明、空调、电视、冰箱、微波炉、烤面包机、电热水壶、搅拌机；另含按钮/旋钮类现场操作支撑样本")
        if row.cells[0].text.strip() == "5.2.11":
            values = (
                "5.2.11",
                "室内安装与布置",
                "VLA优先",
                "海报、墙钉、数码相机、相机三脚架",
                "单件取放、挂置或相机与三脚架连接；以目标位置/连接状态验收",
                "★★★☆☆",
            )
            for cell, value in zip(row.cells, values):
                _set_cell_text(cell, value)
            return
    raise RuntimeError("Task 5.2.11 summary row not found")


def _update_object_table(document: Document) -> None:
    table = document.tables[1]
    replacement_indices = [
        index
        for index, row in enumerate(table.rows)
        if row.cells[0].text.strip() == "家电面板与按钮操作"
    ]
    if len(replacement_indices) != 8:
        raise RuntimeError(f"Expected 8 legacy button/knob rows, found {len(replacement_indices)}")

    for row_index, values in zip(replacement_indices[:4], TASK11_OBJECT_ROWS):
        for cell, value in zip(table.rows[row_index].cells, values):
            _set_cell_text(cell, value)

    for row_index in reversed(replacement_indices[4:]):
        table._tbl.remove(table.rows[row_index]._tr)


def _update_appliance_narrative(document: Document) -> None:
    for paragraph in document.paragraphs:
        if paragraph.text.strip().startswith("本任务沿用网络控制路线，对象包括照明、空调、电视、冰箱"):
            _set_paragraph_text(
                paragraph,
                "本任务沿用网络控制路线，对象包括照明、空调、电视、冰箱、微波炉、"
                "烤面包机、电热水壶、搅拌机。中期以命令下发、状态回读和日志完整性为验收证据，"
                "不要求机械臂执行，也不与VLA闭环成功率混合统计。"
                "微波炉启动/停止键、烤面包机压杆、电热水壶开关、搅拌机电源键、炉灶旋钮、"
                "定时器和水龙头手柄等实体控制部件，任务归属均纳入本节的现场执行子项；"
                "对应视觉训练样本为便于版本管理仍独立存放，不改变其家电综合管理归属。"
                "保留：命令下发、状态回读、日志验收及实体控制支撑样本。",
            )
            return
    raise RuntimeError("Task 5.2.1 narrative not found")


def update_document(path: Path | None = None) -> Path:
    path = path or _find_docx()
    document = Document(path)
    _update_appliance_narrative(document)
    _replace_task11_section(document)
    _update_summary_table(document)
    _update_object_table(document)
    document.save(path)
    return path


if __name__ == "__main__":
    print(update_document())
