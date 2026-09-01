import importlib.util
from pathlib import Path


MODULE = Path(__file__).parents[1] / "update_chapter5_from_audit.py"
SPEC = importlib.util.spec_from_file_location("update_chapter5_from_audit", MODULE)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def test_build_rows_preserves_exact_source_instruction_and_marks_pair_complete():
    records = [
        {
            "task_code": "5.2.8",
            "object": "书籍",
            "instruction": "In the bedroom, move the book from the bed onto either nightstand.",
            "triple_complete": True,
        }
    ]

    rows = MOD.build_rows(records)

    assert rows == [
        {
            "task_code": "5.2.8",
            "task_name": "物品递送",
            "object": "书籍",
            "operation": "卧室：将书籍从床上移至床头柜。",
            "source_instruction": "In the bedroom, move the book from the bed onto either nightstand.",
            "status": "指令、RGB 视频、动作标签已配对",
        }
    ]


def test_add_column_updates_python_docx_table_column_count():
    from docx import Document

    doc = Document()
    table = doc.add_table(rows=2, cols=3)

    MOD.add_column(table)

    assert len(table.columns) == 4
    assert all(len(row.cells) == 4 for row in table.rows)