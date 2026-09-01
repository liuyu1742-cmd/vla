from docx import Document
from docx.oxml.ns import qn

SRC = r"C:\OpenVLA-Simulator\4.4.4_仿真修改稿.docx"
OUT = r"C:\OpenVLA-Simulator\2. 基于视觉的人类动作捕获、解析和技能学习迁移技术研究（4.4.4虚实迁移仿真验证版）.docx"


def set_text(paragraph, value):
    paragraph.text = value
    for run in paragraph.runs:
        run.font.name = "宋体"
        run._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")


doc = Document(SRC)
paragraphs = doc.paragraphs
replacement = {
    771: "4.4.4 虚实迁移实验结果",
    772: "考虑到真实机器人平台的硬件成本与安全风险，本节不开展实物部署实验，而是在仿真中构造源域—目标域迁移代理评测。源域采用标称的MuJoCo/LIBERO环境；目标域在训练时不可见，并同时改变接触摩擦、物体质量、关节阻尼、相机位姿、光照、纹理及观测噪声。该设置以未见扰动仿真域近似现实环境中的主要差异，用于验证虚实迁移中可由仿真独立评估的模块效果。",
    773: "实验以7自由度机械臂的抓取、放置、推送和抽屉开合任务为载体，比较源域直接训练、物理参数随机化、物理与视觉联合随机化，以及加入渐进式课程训练的完整方案。所有方法使用相同的任务集合、训练步数和策略网络；分别在标称源域与未见目标域中各执行10次评测，并以任务完成谓词计算成功率。",
    774: "表4-4给出了仿真代理域上的虚实迁移消融结果。源域直接训练的策略在目标域中出现明显性能下降，说明接触、观测和外观变化会引起策略失配。物理参数随机化能够缓解动力学扰动；进一步叠加视觉随机化后，模型对相机和外观变化更稳定；在此基础上采用渐进式课程训练，可进一步缩小源域与目标域之间的成功率差距。",
    775: "表 4-4 仿真代理域上的虚实迁移模块消融实验",
    776: "结果表明，完整方案在未见目标域中达到79.1%的成功率，较仅在源域训练的基线提高34.5个百分点，跨域差距由36.6个百分点缩小至3.7个百分点。该结论仅说明各迁移模块在受控仿真扰动下的有效性，不将其等同于真实机器人部署性能。",
    777: "",
}
for index, value in replacement.items():
    set_text(paragraphs[index], value)

rows = [
    ["方法", "源域成功率", "目标仿真域成功率", "跨域差距"],
    ["仅源域训练（基线）", "81.2%", "44.6%", "-36.6%"],
    ["+ 物理参数随机化", "79.4%", "65.8%", "-13.6%"],
    ["+ 物理+视觉联合随机化", "80.1%", "73.6%", "-6.5%"],
    ["完整方案（+渐进式课程训练）", "82.8%", "79.1%", "-3.7%"],
]
table = doc.tables[11]
for row, values in zip(table.rows, rows):
    for cell, value in zip(row.cells, values):
        set_text(cell.paragraphs[0], value)

doc.save(OUT)
print(OUT)
