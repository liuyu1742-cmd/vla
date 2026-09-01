from pathlib import Path
from shutil import copy2
from copy import deepcopy
from docx import Document
from openpyxl import load_workbook

ROOT=Path(r'C:\OpenVLA-Simulator')
SRC=Path(r'C:\Users\sjtu101\Desktop\2. 基于视觉的人类动作捕获、解析和技能学习迁移技术研究.docx')
OUT=ROOT/'outputs'/'evidence_revised_report'
OUT.mkdir(parents=True,exist_ok=True)
DST=OUT/'基于视觉的人类动作捕获_解析和技能学习迁移技术研究_规范修订版.docx'
LEDGER=ROOT/'outputs'/'habit_evidence_workbook'/'家居VLA_15任务_120物体_真实示教证据.xlsx'

def repl(p, text):
    rpr=deepcopy(p.runs[0]._element.rPr) if p.runs and p.runs[0]._element.rPr is not None else None
    p.clear(); r=p.add_run(text)
    if rpr is not None:r._element.get_or_add_rPr().append(rpr)
def cell(c,text):
    p=c.paragraphs[0]; rpr=deepcopy(p.runs[0]._element.rPr) if p.runs and p.runs[0]._element.rPr is not None else None
    p.clear();r=p.add_run(str(text))
    if rpr is not None:r._element.get_or_add_rPr().append(rpr)
    for x in c.paragraphs[1:]:x.clear()
def table(t, header, rows):
    for i,row in enumerate(t.rows):
        vals=header if i==0 else (rows[i-1] if i-1<len(rows) else ['']*len(header))
        for j,c in enumerate(row.cells):cell(c,vals[j] if j<len(vals) else '')

copy2(SRC,DST); d=Document(DST)

# 4.2.5: retain the original experiment and its table, but remove unperformed numbers.
repl(d.paragraphs[699],'本项目尚未完成本节所述5个典型家政任务的视频模仿学习对比实验，因而不报告任务成功率、新物体泛化率或推理延迟。当前已完成的工作仅为公开真实示教数据的本地完整性核验和OpenVLA推理环境部署；这些工程核验不能替代本节的实验结果。')
repl(d.paragraphs[700],'后续实验将沿用本节原有的5个典型任务设置，在统一机械臂、相机、训练预算和成功判据下采集数据。待每项任务完成重复试验后，再补充均值、标准差、成功率和推理延迟；在此之前，表4-1只保留实验实施状态。')
repl(d.paragraphs[701],'表 4-1 视频模仿学习实验实施状态')
table(d.tables[8],['任务','演示次数','技能学习成功率','新物体泛化成功率','推理延迟'],[
['物品抓取','未开展','未报告','未报告','未报告'],['抽屉开关','未开展','未报告','未报告','未报告'],['精细旋转操作','未开展','未报告','未报告','未报告'],['桌面擦拭','未开展','未报告','未报告','未报告'],['衣物折叠','未开展','未报告','未报告','未报告'],['说明','待统一实验协议','不填充推测数值','不填充推测数值','不填充推测数值']])

# 4.4.4: same experiment, transparently not carried out on real hardware.
repl(d.paragraphs[768],'本项目尚未接入实体机械臂、实体相机或真实接触传感器，因此未开展本节域随机化、系统辨识与渐进迁移的对照实验。当前RoboCasa仅用于仿真环境内的接口联调，不能据此计算真实成功率或Sim-to-Real Gap。')
repl(d.paragraphs[769],'后续将按本节原有4种方法设置，在相同任务、相同对象分布和相同成功判据下分别完成仿真与实体评测；每种方法至少使用20个随机种子，再报告均值、置信区间与差距。在实体试验完成前，表4-4不填入任何迁移性能数值。')
repl(d.paragraphs[770],'现阶段只能确认仿真闭环接口可运行；领域随机化增益、系统辨识增益及实体部署成功率均仍待实验验证。')
repl(d.paragraphs[771],'表 4-4 域随机化与渐进迁移实验实施状态')
table(d.tables[11],['方法','仿真成功率','真实成功率','Sim-to-Real Gap'],[
['无域随机化（基线）','未开展','未开展','未报告'],['物理+视觉域随机化','未开展','未开展','未报告'],['域随机化+系统辨识','未开展','未开展','未报告'],['渐进式四阶段迁移','未开展','未开展','未报告']])

# 4.6: preserve the three original experiment subsections; do not substitute another experiment.
repl(d.paragraphs[808],'本项目已完成OpenVLA模型加载、本地CUDA推理和RoboCasa仿真接口联调，但尚未按本节所述协议完成OpenVLA微调方法对比、OpenVLA-OFT基准评估和ALOHA双臂评估。因此，本节不将工程联调结果写作上述三项实验的性能结果。')
repl(d.paragraphs[810],'当前仅实施了LoRA适配器训练流水线和训练清单隔离检查，未在相同数据、相同训练预算下完成Full Fine-tuning、LoRA、Last Layer Only和Frozen Vision的受控对比。表4-7保留原方法组，但所有性能字段均标为未开展。')
repl(d.paragraphs[811],'待完成四种微调方法的重复训练后，将按照原表报告可训练参数量、显存占用和任务成功率；本次修订不填入未经本项目复现的数值。')
repl(d.paragraphs[812],'表 4-7 OpenVLA微调方法对比实验实施状态')
table(d.tables[14],['微调方法','可训练参数量','显存(batch=16)','成功率'],[['Full Fine-tuning','未开展','未测量','未报告'],['LoRA (rank=32)','未完成受控对比','未测量','未报告'],['Last Layer Only','未开展','未测量','未报告'],['Frozen Vision','未开展','未测量','未报告']])
repl(d.paragraphs[815],'本项目未部署OpenVLA-OFT，也未在LIBERO四个任务套件上执行原文所述的基准评估。RoboCasa中的单一整理入柜联调任务不能替代LIBERO基准，因此不使用其结果填充表4-8。')
repl(d.paragraphs[816],'表4-8仅记录基准实验尚未实施。后续如采用OpenVLA-OFT，应固定模型版本、训练数据、任务套件和评价脚本，并分别报告各套件的重复试验结果。')
repl(d.paragraphs[817],'本次未实施并行解码、连续动作表示和回归损失的消融实验，故不报告其性能贡献。')
repl(d.paragraphs[818],'表 4-8 OpenVLA-OFT基准实验实施状态')
table(d.tables[15],['任务套件','OpenVLA-OFT','OpenVLA baseline','Diffusion Policy'],[['LIBERO-Spatial','未开展','未开展','未开展'],['LIBERO-Object','未开展','未开展','未开展'],['LIBERO-Goal','未开展','未开展','未开展'],['LIBERO-Navigation','未开展','未开展','未开展'],['平均','未报告','未报告','未报告']])
repl(d.paragraphs[821],'本项目未部署ALOHA双臂机器人，亦未采集折叠衣物、拾取放置、搅拌和精细装配的ALOHA实验数据。因此本节不报告双臂任务成功率、语言跟随准确率、吞吐量或推理延迟。')
repl(d.paragraphs[822],'当前RoboCasa联调仅证明仿真观察—动作接口可运行，不能替代ALOHA实体双臂评估。待具备ALOHA或等效双臂平台后，将按原任务集、相同成功判据和重复次数补充表4-9。')
repl(d.paragraphs[823],'表 4-9 ALOHA双臂机器人评估实施状态')
table(d.tables[16],['方法','类型','平均成功率','语言跟随准确率'],[['OpenVLA-OFT+','未部署','未报告','未报告'],['RDT-1B','未部署','未报告','未报告'],['π0','未部署','未报告','未报告'],['Diffusion Policy','未部署','未报告','未报告'],['说明','无ALOHA平台','不填充推测数值','不填充推测数值']])
repl(d.paragraphs[824],'本项目后续应先完成本节三类实验的可重复实施：固定数据切分和随机种子，记录训练曲线与显存；在公开基准上运行统一评价脚本；最后在实体双臂平台上完成安全约束下的重复试验。')
repl(d.paragraphs[826],'本章已完成的工作是OpenVLA本地推理环境、公开真实示教数据的本地核验和RoboCasa接口联调；视频模仿学习对比、虚实迁移、OpenVLA-OFT和ALOHA双臂性能实验均未按原协议完成。为保证结论可追溯，本章不再保留未复现的成功率、迁移差距和延迟数据。后续将以统一数据切分、任务判据和重复试验补充各表结果。')

# Fifth chapter: use the approved real-evidence taxonomy while retaining headings/table count/style.
names=['家庭物品整理与收纳','室内物品取用、递送与摆放','室内垃圾分类与投放','餐具整理与餐后清洁','食品存放与食材管理','烹饪、加热与饮品服务','家电与水电设施控制','室内卫生清洁与物件维护','衣物清洗与织物护理','家庭维修与工具使用','室内布置与家具陈设','浴室用品与个人卫生服务','工作学习区域服务','居家休闲与娱乐用品服务','室内出入与随身物品服务']
desc=['对书籍、文件夹、文具、柜子和收纳箱等进行分类、放入、叠放与归位。','从室内储物位取用水瓶、托盘、容器等物体，递送到指定位置并安全摆放。','识别纸巾、废纸、饮料罐等废弃物，完成分类与投放。','对盘子、碗、叉子、玻璃杯、锅和洗碗机执行收集、清洗、装载与归位。','对冰箱、保鲜盒、厨房柜、密封罐等执行开关、存放、密封与分类。','围绕平底锅、炉灶、烤箱、微波炉和烤盘执行放置、加热、取出与端送。','围绕灯开关、水龙头、电水壶、烤面包机和插线板执行按压、旋转与开关控制。','围绕掸子、刷子、海绵、清洁布、桌面、白板和置物架执行除尘、擦拭、刷洗与复位。','围绕洗衣机、枕套、帽子、毛巾、T恤、牛仔裤和洗衣篮执行装载、清洗、折叠与收纳。','围绕工具箱、电钻、钳子、扳手、螺丝刀、锤子和万用表执行取放、对准、旋转与检测。','围绕海报、墙钉、椅子、沙发、花瓶、相框和手机支架执行固定、推拉、摆正与陈设。','围绕洗涤剂、皂液器、牙膏、牙刷、洗发水和卫生纸执行取用、补充与归位。','围绕显示器、键盘、鼠标、订书机、笔记本电脑、打孔器和摄像头执行摆正、按键与调整。','围绕数码相机、三脚架、棋子、游戏手柄、运动器材、桌游、拼图和耳机执行取放与收纳。','围绕鞋子、外套、雨伞、随身包、拉链头、钥匙串、门把手和鞋架执行挂放、开合与归位。']
repl(d.paragraphs[829],'本项目技能库以“任务类别—物体—对象操作—示教来源—训练状态”为基本记录单元。只有同时具有任务文本、视觉观测和动作标签，或具有明确公开转换路线的真实示教数据，才进入候选技能清单。')
repl(d.paragraphs[830],'技能执行采用任务层、对象层和动作原语层的分层表示。任务层描述用户目标；对象层标注可操作物体与目标区域；动作原语层记录接近、抓取、移动、放置、开关、旋转和擦拭等可复用动作。')
repl(d.paragraphs[831],'现阶段清单包括15类室内日常任务和120种具体物体。HABIT、BEHAVIOR-1K和DROID-100的本地配对数据具备转换条件；其他公开示教仅标记为待下载与动作映射，不表述为已完成训练。')
repl(d.paragraphs[833],'图 5.1 技能库总体框架图')
repl(d.paragraphs[835],'本分类体系以现有真实机器人示教为约束，覆盖室内日常操作，避免将户外、采购、儿童专属和重复整理类需求纳入。15类任务按主要操作目的划分，每类包含8种物体，共120种；对象操作和数据状态见表5-1至表5-9。')
for i,(n,x) in enumerate(zip(names,desc)):
    h=836+3*i;repl(d.paragraphs[h],f'5.2.{i+1} {n}');repl(d.paragraphs[h+1],x);repl(d.paragraphs[h+2],'')
repl(d.paragraphs[880],'技能分类、物体及对象操作详见表5-1至表5-9。表内“本地配对”表示已核验真实示教数据，“待下载映射”表示公开真实示教尚需补齐本地数据与动作转换。')
repl(d.paragraphs[881],'表 5-1 15类室内任务与示教证据总表')

wb=load_workbook(LEDGER,read_only=True,data_only=True);ws=wb['任务物体清单']; rows=[list(r[:8]) for r in ws.iter_rows(min_row=4,values_only=True) if r[0]]; g={}
for r in rows:g.setdefault(str(r[0]),[]).append(r)
summary=[]
for i,n in enumerate(names):
    rr=g[f'{i+1:02d}']; status='本地配对/待下载映射';summary.append([str(i+1),n,str(len(rr)),f'{rr[0][3]}、{rr[1][3]}等',status])
table(d.tables[17],['序号','任务类型','技能数量','典型技能','实现难度'],summary)
repl(d.paragraphs[884],'本节按任务类别列出120种对象操作。表中“技能名称”采用“物体+操作”的形式；控制对象为具体物体；操作类型为该物体对应的可执行动作。')
caps=['表 5-2 家庭物品整理与室内递送','表 5-3 垃圾分类与餐后清洁','表 5-4 食品管理与烹饪加热','表 5-5 设施控制与室内卫生清洁','表 5-6 衣物护理与家庭维修','表 5-7 室内陈设与浴室用品服务','表 5-8 工作学习与休闲用品服务','表 5-9 室内出入与随身物品服务']
for p,c in zip([885,887,889,891,893,895,897,899],caps):repl(d.paragraphs[p],c)
for ti,pairs in zip(range(18,26),[(0,1),(2,3),(4,5),(6,7),(8,9),(10,11),(12,13),(14,)]):
    data=[]
    for ci in pairs:
        for r in g[f'{ci+1:02d}']:data.append([str(len(data)+1),f'{r[3]}{r[4]}',r[3],r[4]])
    table(d.tables[ti],['序号','技能名称','控制对象','操作类型'],data)
repl(d.paragraphs[923],'本章以真实示教证据为边界，形成15类室内日常任务、120种物体及对应对象操作的技能清单。清单将本地配对数据与待下载映射数据明确区分，后续需逐项完成动作空间转换、微调和统一成功判据评测，方可形成可部署技能。')
d.save(DST);print(DST)
