# VLA 元数据普查实施计划

1. 建立统一任务记录、对象记录和证据级别校验。
2. 用 AST 静态解析本地 RoboCasa，避免创建仿真环境：
   - 任务类和 docstring；
   - `get_ep_meta()` 中的语言指令；
   - `_get_obj_cfgs()` 中的对象组和角色；
   - `OBJ_CATEGORIES` 中的对象属性。
3. 解析本地 BridgeData V2 的 `dataset_info.json` 和 `features.json`，记录分割、字段和本地完整性。
4. 建立官方数据源清单，按任务级元数据可获得性分级。
5. 通过规则化归类生成 15 个高层家庭任务候选，并保留未匹配项。
6. 从显式对象注册和任务对象配置中生成对象排名；输出真实数量，最多取 120。
7. 生成 JSON、JSONL、CSV 和 Markdown 报告。
8. 运行单元测试、CLI 重建和结果一致性检查。
