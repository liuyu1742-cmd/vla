from pathlib import Path
from shutil import copy2
from docx import Document
from docx.shared import Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
folder=Path('outputs/chapter5_midterm_revision')
src=[x for x in folder.glob('*.docx') if not x.name.startswith('~$') and '\u5bf9\u8c61\u52a8\u4f5c\u4e00\u81f4\u7248' in x.name][0]
out=folder/'\u7b2c\u4e94\u7ae0\uff08\u4e2d\u671f\u9a8c\u6536\u7248_\u4fee\u8ba2_\u56fe5.2\u66f4\u65b0\u7248\uff09.docx'
copy2(src,out)
img=Path('outputs/chapter5_midterm_revision/figure5_2_work/figure5_2_task_catalog.png')
d=Document(out)
idx51=next(i for i,x in enumerate(d.paragraphs) if x.text.strip().startswith('5.1 '))
body=d.paragraphs[idx51+1]
old_next=d.paragraphs[idx51+2]
body_text='为保证任务体系与具体对象交互保持一致，本研究将室内家庭服务划分为15类任务，并按照任务名称、操作对象、代表性对象操作和技能条目数建立统一目录。当前目录共包含167项技能条目，不同任务的覆盖规模依据对象种类及操作差异确定，单类任务包含5至40项技能，不再采用每类固定8项的设置。表5-2、表5-3、表5-4、表5-7及表5-8-1至表5-8-10按照统一字段组织，覆盖家电综合管理、环境调节、智慧安防、清洁用品取放、厨房器具归位、养护管理、物品递送、整理收纳、储物设施开合、室内摆放、餐具与容器摆放、衣物鞋类归位、工作学习区服务及卫浴用品服务等内容。具体而言，物品递送包含书籍由床至床头柜、饮料瓶由冰箱至咖啡台等室内转移；整理收纳包含保鲜盒归柜、书籍装入收纳箱或纸箱等操作；室内摆放包含海报悬挂、相机安装和蜡烛由橱柜摆放至台面等操作。该目录避免以相近对象重复计数，并为任务检索、对象识别和动作执行提供统一的结构化索引。图5.2展示了15类任务及其代表性对象操作的总体分布。'
body.clear(); body.add_run(body_text)
old_next._element.getparent().remove(old_next._element)
image_p=d.add_paragraph(); image_p.alignment=WD_ALIGN_PARAGRAPH.CENTER
image_p.add_run().add_picture(str(img),width=Inches(6.25))
caption=d.add_paragraph('图5.2 家庭服务任务与对象操作目录'); caption.alignment=WD_ALIGN_PARAGRAPH.CENTER
for q in d.paragraphs:
 if q.text.strip().startswith('图5-1'):
  caption.style=q.style; break
body._p.addnext(image_p._p); image_p._p.addnext(caption._p)
d.save(out)
print(out, out.stat().st_size)
