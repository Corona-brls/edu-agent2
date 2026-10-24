# V6-FINAL-8最小修复计划

> 命令：8B——基于8A证据制定最小修复与验收口径计划  
> 状态：`READY_FOR_REVERSE_REVIEW`  
> 编制日期：2026-10-10  
> 边界：本文件只制定计划；未修改产品代码、正式测试、配置、`.env`、数据库、公共API、公共Schema、认证实现或超时，也未执行命令8C。

## 1. 前置条件复核

命令8A证据位置：

- `diagnostics/v6final8-diagnostic-20261010T081112Z-5f6946f3/persistent_evidence/diagnostic_report.json`
- `diagnostics/v6final8-diagnostic-20261010T081112Z-5f6946f3/persistent_evidence/frozen_acceptance_inputs.json`
- `diagnostics/v6final8-diagnostic-20261010T081112Z-5f6946f3/persistent_evidence/source_chain_analysis.json`
- `diagnostics/v6final8-diagnostic-20261010T081112Z-5f6946f3/persistent_evidence/practice_positive/result.json`
- `diagnostics/v6final8-diagnostic-20261010T081112Z-5f6946f3/persistent_evidence/practice_insufficient/result.json`
- `diagnostics/v6final8-diagnostic-20261010T081112Z-5f6946f3/persistent_evidence/state_machine/result.json`
- `diagnostics/v6final8-diagnostic-20261010T081112Z-5f6946f3/persistent_evidence/deepseek_ask/result.json`
- `diagnostics/v6final8-diagnostic-20261010T081112Z-5f6946f3/persistent_evidence/upload_exact/result.json`
- `diagnostics/v6final8-diagnostic-20261010T081112Z-5f6946f3/persistent_evidence/flow_12/result.json`

复核结果：

- 8A状态为`DIAGNOSIS_COMPLETE`。
- `missing_decisive_evidence`为空，`UNVERIFIED=0`。
- 唯一确认产品缺陷为flow_04的`PRODUCT_GROUNDING_CONTRACT_DEFECT`。
- flow_06为`STATE_MACHINE_PASS / ACCEPTANCE_INPUT_GAP`。
- flow_08为`QUESTION_EVIDENCE_MISMATCH / FIXTURE_EVIDENCE_INSUFFICIENT`。
- flow_12为`TEST_ORACLE_CONFLICT_RESOLVED`。
- 固定输入已保存；正式数据库、140个受管产品文件、隔离runtime和端口不变量均通过。

因此8B允许制定一个flow_04产品修复，以及flow_06、08、12的验收口径修订；不存在`NEED_MORE_EVIDENCE`项。

## 2. 裁决总表

| 项目 | 8B裁决 | 是否改产品 | 后续动作 |
|---|---|---:|---|
| flow_04 实践指导 | `FIX_PRODUCT` | 是 | 修复实践证据传递、逐字段span绑定、服务器渲染、Guard全字段覆盖和单次生成重试 |
| flow_06 ask_user/continue | `NO_CHANGE / AMEND_ORACLE` | 否 | 确定性状态机作为验收；DeepSeek自由决策只作行为观察 |
| flow_08 上传资料问答 | `AMEND_ORACLE` | 否 | 只换成8A冻结的完整资料/问题组合 |
| flow_12 刷新与深链 | `AMEND_ORACLE` | 否 | 沿用历史刷新口径，不开发持久登录 |

flow_04和flow_08没有被8A确认成共同产品缺陷。因此本批不把单课程/单上传普通问答全面迁移到新契约；flow_08只作为共享检索/Guard代码变化后的兼容性smoke和修订后的验收项。

## 3. flow_04：`FIX_PRODUCT`

### 3.1 根因

完整链路中的三个缺口共同构成已确认缺陷：

1. `backend/rag/hybrid_retriever.py::_snippet`把检索展示摘要限制为240字符；`backend/agent/orchestrator.py::_evidence_from_result`又把该展示摘要直接作为`EvidenceItem.text`交给模型。正常上传并索引的完整chunk仍在数据库中，但实践生成只看见前240字符。
2. `backend/agent/orchestrator.py::_answer`对`practice_guide`使用自由字段`PracticeGuide`，而不是服务器控制的source/span契约；`_format_practice_guide`与`_practice_evidence_basis`独立拼接正文、证据摘要和引用，没有逐字段绑定。
3. `backend/guards/citation_guard.py::enforce`对单来源非grounded回答只检查引用附近的claim窗口，不检查研究目标、方法、工具、每一步、结果分析、创新点和风险等全部字段。8A证据不足对照因此出现模板生成资料外内容但Guard仍判定通过。

现有测试只证明结构化输出能解析、practice intent能路由以及正文含“研究目标”等标题，不证明每个字段和每一步均由选定span支持。

### 3.2 受影响与不受影响路径

受影响：

- 已登录学生通过`/api/practice/guide`进入Agent的实践指导。
- `/api/chat`被Planner识别为`practice_guide`的路径。
- 课程本地检索或`course_retriever`工具提供证据后，实践结构化生成、服务器渲染、Citation Guard和终态判定。
- DeepSeek失败后当前Template fallback生成`PracticeGuide`的路径。

本批明确不改：

- Planner动作选择、`ask_user/continue`状态机、run所有权和403权限语义。
- 单课程/单上传普通`course_qa`的产品契约；flow_08没有证明该路径存在产品缺陷。
- 多外部来源现有`GroundedSpanAnswer`契约及其一次生成重试。
- MCP调用、真实工具调用、工具超时和`TOOL_TIMEOUT`语义。
- `HybridRetriever`的排序、阈值、240字符展示摘要及检索观察格式。
- 公共API请求/响应字段、数据库表结构、认证、前端会话和flow_12行为。

### 3.3 最小设计

#### A. 请求内重新水合完整、已授权的实践证据

在`AgentOrchestrator._answer`进入practice分支前新增请求内辅助函数，例如：

```text
_hydrate_practice_evidence(state, answer_evidence, storage)
```

规则：

1. 对含`document_id/chunk_id`的course证据，只从`LearningLogger.document_chunks(course_id=state.course_id)`中查找同一`document_id + chunk_id`、`INDEXED`的服务器记录。
2. 找到后，用数据库中的`chunk_text`构造只供本次模型请求、span构造和Guard使用的`EvidenceItem`副本；不得相信检索结果、模型或工具返回的全文。
3. 文档/课程/chunk/index状态任一不匹配时fail closed，终态为`ABSTAINED`，内部错误语义为`PRACTICE_EVIDENCE_SCOPE_MISMATCH`。
4. `document_id=0`的受控本地fallback不能冒充数据库chunk；它只能以已有服务器读取到的文本建立span，无法满足完整实践字段时按证据不足拒答。
5. 完整`chunk_text`不得写入retrieval observation、agent trace、API citation snippet或诊断日志；外部可见摘要继续使用现有240字符snippet。模型调用日志继续只保存已有prompt/candidate哈希。
6. 不修改`RetrievalResult`和`HybridRetriever._snippet`，避免把完整chunk加入`to_dict()`后意外暴露给UI、日志或公共接口。

#### B. 实践专用证据契约

在内部LLM结构化Schema中增加`GroundedPracticeGuide`。核心字段全部使用现有`GroundedSpanClaim`元素，而不是自由字符串：

```text
research_goal: list[GroundedSpanClaim]
method_design: list[GroundedSpanClaim]
tools: list[GroundedSpanClaim]
experiment_steps: list[GroundedSpanClaim]
result_analysis: list[GroundedSpanClaim]
innovation_points: list[GroundedSpanClaim]
risks: list[GroundedSpanClaim]
proposed_extensions: list[str]
```

语义：

- 前七个核心字段都是“证据中的事实或资料明确给出的方案”，每个元素必须带服务器已知的`source_id`和`evidence_span_id`。
- `proposed_extensions`是可选的“拟议实验方案”，没有引用，不得进入核心事实字段；服务器必须以“拟议方案（非资料事实、待验证）”单独标识。
- 证据不足的核心字段不得用proposal填充。冻结的“仅根据资料”输入下，核心字段不完整应拒答，不得用模板通用语句补齐。
- 把七个字段按固定顺序扁平化后，复用`_validate_grounded_span_answer`的unknown source、unknown span、source/span mismatch、quote、identifier、duplicate和支持性校验，再恢复字段/序号信息。
- 请求明确要求步骤数量时，只采用通用、有限范围的数量解析（阿拉伯数字或常用中文数字，限定合理上限）写入内部`required_step_count`元数据；不得为`F8-DIAG-PRACTICE-A`或某一句验收文本硬编码。冻结正例要求6个已绑定步骤。
- 没有任何合格事实，或冻结正例要求的核心字段/步骤数不完整时，终态`ABSTAINED`，错误语义`PRACTICE_EVIDENCE_INCOMPLETE`。

该Schema仅用于模型内部结构化输出，不改变FastAPI公共请求/响应Schema。

#### C. 服务器渲染与Guard全字段覆盖

新增实践专用服务器render unit，至少包含：

```text
section
ordinal
kind = evidence_fact | proposal
claim_text
source_id
evidence_span_id
evidence_quote
```

渲染要求：

- `evidence_fact`由服务器添加相邻`[Sx]`及精确证据quote，模型不能输出引用标记或quote。
- `proposal`由服务器添加固定的“拟议方案（非资料事实、待验证）”标签，不带引用或quote。
- 删除practice旧的“统一Evidence basis + 正文 + 文末引用集合”拼接方式；不能靠独立引用行给整篇正文背书。

Citation Guard增加内部practice renderer模式和`verified_practice_bindings`参数。它必须遍历最终回答全部非空正文单元：

1. 每个核心字段事实的section、ordinal、claim、source、span和quote必须与请求内verified binding逐项同序匹配。
2. 任一缺失、额外、换序、重复、伪造source、错span、跨source、quote篡改、未绑定正文或字段标题下游离内容均拒答。
3. proposal必须具有服务器固定标签、不得含引用/quote，且文本和顺序必须与服务器已验证的proposal render unit一致；proposal不能算作核心字段完整性或证据支持。
4. 继续调用现有`_validate_grounded_citation`检查问题相关性和selected-span支持，不降低词项/具体事实检查，不增加“宽松模式”。
5. Guard只接受服务器传入的请求内span/binding；模型提供的source/span字符串不能直接成为可信绑定。

建议新增内部错误码：

- `PRACTICE_GROUNDING_RENDER_FORMAT`
- `PRACTICE_GROUNDING_BINDING_MISMATCH`
- `PRACTICE_PROPOSAL_FORMAT`
- `PRACTICE_EVIDENCE_INCOMPLETE`
- `PRACTICE_EVIDENCE_SCOPE_MISMATCH`

用户可见信息继续使用统一安全拒答文本，不回显内部证据或归一化内容。已有`GROUNDING_UNKNOWN_SOURCE`、`GROUNDING_UNKNOWN_SPAN`、`GROUNDING_SPAN_SOURCE_MISMATCH`、`GROUNDING_QUOTE_MISMATCH`、`GROUNDING_DUPLICATE_*`和`UNSUPPORTED_CITATION`继续复用。

#### D. 最多一次、仅重做生成

practice第一次候选发生可恢复的结构化/绑定/Guard失败时，允许调用专用`_retry_grounded_practice_once`一次：

- 复用同一`LLMRequest`、同一完整证据副本、同一span集合和同一run/step。
- 只再次调用最终结构化生成；不得再次运行Planner、retriever、LocalRetriever、MCP或真实工具。
- retry metadata记录`grounding_retry_attempt=1`、前一错误码及候选哈希，不记录候选全文。
- 第二次仍失败立即`ABSTAINED`，保持原错误码或`PRACTICE_GROUNDING_FAILED`，不得进行第三次生成。
- transport/DNS/TLS/429/5xx重试仍只由现有provider层限次机制负责，不与grounding生成重试叠加为重新跑Agent。

Template行为：

- 显式`LLM_PROVIDER=template`时，TemplateProvider必须生成同一个`GroundedPracticeGuide`契约，只能从服务器span提取内容。
- DeepSeek失败后若保留Template fallback，fallback候选也必须通过完全相同的契约、渲染和Guard，并在响应中保持`provider=template`、`model=local-template`、`fallback_used=true`；不得计为DeepSeek成功。
- 若Template不能满足核心字段/步骤要求，则拒答；禁止继续使用当前不带span的通用`PracticeGuide`作为成功答案。

#### E. 脱敏的首个Guard失败点

扩展内部`CitationCheck`，增加可选的`failure_detail`，只允许包含：

```text
error_code
section
ordinal
source_id
span_id
claim_sha256
normalized_claim_sha256
claim_length
normalization_flags
```

要求：

- 只保留首个失败单元。
- 禁止保存claim、quote、完整证据、提示词、URL查询参数、token、密钥或跨课程文本。
- `AgentOrchestrator._trace`使用现有`agent_traces.metadata_json`写入`citation_guard_failed`事件；无需新增数据库表或迁移。
- API响应不新增该字段，普通学生不可读取内部诊断。

### 3.4 拟修改文件和函数

| 文件 | 计划修改 |
|---|---|
| `backend/agent/orchestrator.py` | practice证据重新水合；构建完整span；生成/验证/渲染`GroundedPracticeGuide`；一次practice生成重试；完整性与步骤数校验；脱敏Guard trace；移除旧practice统一引用拼接 |
| `backend/llm/schemas.py` | 增加内部`GroundedPracticeGuide`，复用`GroundedSpanClaim`；不改公共API Schema |
| `backend/llm/template_provider.py` | 显式template及fallback均只生成新的extractive实践契约，不再生成资料外通用`PracticeGuide`成功答案 |
| `backend/guards/citation_guard.py` | 增加practice renderer全字段校验、proposal格式校验和内部`failure_detail`；保留现有fail-closed语义 |
| `tests/test_practice_grounding_contract.py`（新建） | flow_04根因、正例、负例、重试次数、无重复检索/工具调用、日志脱敏的集中单元/集成测试 |
| `tests/test_llm_provider.py` | Template/DeepSeek对`GroundedPracticeGuide`的结构化解析和provider/fallback标识测试 |
| `tests/test_r3_grounded_answer_contract.py` | 共享Guard的practice renderer覆盖及现有grounded renderer不回归 |
| `tests/test_api_endpoints.py` | `/api/practice/guide`与chat practice路由的正例、证据不足拒答和公共响应兼容性 |

本批不修改：

- `backend/rag/hybrid_retriever.py`：240字符继续作为展示/观察摘要；完整证据只在practice请求内从受权SQLite chunk重新水合。
- `backend/storage/logger.py`及数据库迁移：使用现有`agent_traces.metadata_json`。
- `backend/main.py`和公共请求/响应模型：接口字段不变。
- `frontend/app.py`、认证、continue状态机、MCP实现和全局配置。

### 3.5 安全不变量和禁止事项

- Citation Guard继续fail closed；禁止删除、绕过、跳过或降低支持性校验。
- source/span必须由当前请求内、当前课程、已INDEXED的服务器证据生成并验证。
- 不得接受模型自报source、span、quote或document/chunk归属。
- 跨课程、跨班级、未知document/chunk、非INDEXED、伪造source、错span、重复claim、额外未绑定正文仍必须拒绝。
- proposal必须显式标为非资料事实，不能带引用，不能满足核心证据字段或步骤数。
- 禁止把完整chunk放进retrieval observation、API citation摘要或诊断日志。
- 禁止记录API key、JWT、完整prompt、完整claim/quote或敏感全文。
- 一次grounding重试不得重复Planner、检索、MCP、工具调用、问题写入或产生迟到写入。
- Template fallback不得冒充DeepSeek；provider/model/fallback标志必须真实。
- 不改变Agent总步数、工具调用上限、业务超时和`TOOL_TIMEOUT`语义。

### 3.6 测试计划

#### 单元测试

1. 已授权、已INDEXED的`document_id/chunk_id`能重新水合完整`chunk_text`；错误课程、错误document/chunk、PENDING/FAILED均拒绝。
2. 对外observation和citation snippet仍为短摘要，不含完整后半段或敏感全文。
3. `GroundedPracticeGuide`七个核心字段逐项解析并保留section/ordinal；冻结正例的六个实验步骤分别绑定到合法span。
4. evidence事实缺source/span、未知source、未知span、span/source错配、quote篡改、duplicate、额外正文、字段换序均fail closed。
5. proposal有固定标签且无引用；proposal混入引用、quote或冒充核心事实时拒绝。
6. 只有题目/占位文本的证据不足对照不能由Template通用内容补齐，终态必须`ABSTAINED`。
7. 首候选不合规时仅生成重试一次；Planner、retriever、MCP、真实工具和tool_call计数不增加；第二次失败拒答。
8. `failure_detail`只含白名单字段和哈希；断言不含原claim、quote、完整chunk、密钥或token。
9. 显式Template作为primary时输出真实provider标志；DeepSeek fallback时不得被标成DeepSeek成功。

#### API/集成测试

1. 使用8A冻结的`F8-DIAG-PRACTICE-A`完整资料，经正常上传和INDEXED后，固定问题返回六个步骤；研究目标、方法、工具、每一步、结果分析、创新点和风险均有正确document/chunk/source/span绑定，provider元数据真实。
2. 使用冻结的`5G NAS authentication.`极简资料和原问题，必须安全拒答；不能出现当前模板通用五步内容。
3. `/api/practice/guide`和chat路由返回字段保持兼容；只增加内部trace，不改变公共响应结构。
4. 使用错误课程的同名marker不能被检索、绑定或引用。

#### 共享Guard负例回归

- 运行现有`tests/test_r3_grounded_answer_contract.py`中selected-span overreach、missing/extra/swapped binding、unknown/wrong span、quote tampering、forged source、duplicate和无支持claim用例。
- 运行一个已通过的普通课程引用兼容性smoke：`tests/test_api_endpoints.py::test_chat_endpoint_records_question`。
- 运行上传/索引兼容性smoke：`tests/test_api_endpoints.py::test_http_upload_pending_then_reindex_makes_document_retrievable`。
- 不重跑全部历史V6，不重跑H5/F01-F15。

### 3.7 数据库、API和Schema兼容性

- 数据库：不新增表/列/索引，不迁移；只复用`agent_traces.metadata_json`。
- 公共API：请求和响应字段、HTTP状态及认证方式不变。
- 公共Schema：不变。`GroundedPracticeGuide`只属于LLM内部结构化契约。
- 已保存run：无需回填或迁移；新契约只作用于修复后创建的practice run。
- 引用对象继续返回现有document/chunk/page/section/source字段。

### 3.8 回滚方式

实施时应保持提交/补丁仅包含上述允许文件。若需回滚：

1. 整体回退practice专用Schema、hydration、renderer、Guard practice mode和相关测试；不得只回退Guard而保留可输出未绑定正文的新renderer。
2. 不需要数据库回滚或数据迁移。
3. 回滚后恢复原安全行为可能重新出现flow_04拒答/Guard盲区，因此回滚仅用于紧急恢复，不能作为最终验收通过依据。

### 3.9 修复后的最小浏览器补验

只补验以下范围：

1. flow_04正例：使用冻结完整资料和固定问题，提交一次；确认六步及全部字段均受证据支持、引用指向正确document/chunk、DeepSeek无fallback时如实显示。
2. flow_04证据不足对照：使用冻结极简资料和原问题，确认安全拒答；不得把拒答算缺陷。
3. flow_08修订后的上传问答：使用冻结完整句/固定问题，确认PENDING→INDEXED→精确复述和正确引用。这是验收口径修订兼共享路径smoke，不是产品缺陷复修。
4. 一个既有PASS的课程引用smoke：复用flow_03的课程A精确句和引用检查，只用于排除共享Agent/Guard回归，不重开flow_03全部验收。
5. 一个引用负例：伪造或错span回答必须拒绝。

## 4. flow_06：`NO_CHANGE / AMEND_ORACLE`

不修改`AgentOrchestrator.continue_run`、`_load_or_create_state`、Planner提示、Template触发逻辑或权限代码。

后续验收拆成两项：

1. **确定性状态机（阻断项）**：沿用仓库受支持触发方式；必须验证一次`REQUIRES_INPUT`、固定continue输入、同一run_id到明确终态、其他用户GET/continue均403、终态不可重复continue。
2. **DeepSeek自由语言观察（非阻断项）**：使用冻结问题和continue输入，保存Planner原始/归一化动作。Planner未选`ask_user`或补充后再次追问，只记录行为波动/输入充分性；只有Planner已选`ask_user`而编排层改写动作，或同run/权限/终态语义破坏，才升级产品缺陷。

禁止：

- 不为验收句增加魔法关键词、字符串分支或强制ask_user。
- 不把DeepSeek概率性动作当作continue状态机是否正确的唯一判据。
- 本批不安排flow_06浏览器重跑；仅当8C/8D实际触及continue代码时才运行`tests/test_agent_resume.py`、同run继续、跨用户403和终态不可重复继续测试。

## 5. flow_08：`AMEND_ORACLE`

不修改上传、索引、普通course QA、Citation Guard阈值或前端。

后续验收必须原样使用8A冻结组合：

```text
资料：F8-DIAG-UPLOAD-A：课程A的受控日志样本仅用于比较正常认证与重放检测，不得用于公网目标。

问题：请仅根据刚上传的课程A资料，原样复述包含 F8-DIAG-UPLOAD-A 的完整一句，并显示引用和来源。
```

判定口径：

- 正常上传并进入`INDEXED`。
- 学生只提交一次固定问题。
- 回答原样包含完整句，显示引用和来源，引用必须指向本次上传的正确document/chunk。
- 不出现其他课程marker，不以改问题、删引用、放宽Guard取得PASS。
- 原`F6-UPLOAD-A-ONLY / Teacher upload acceptance sample.`夹具不再用于“复述含marker完整事实句”的断言；其历史安全拒答继续保留。

flow_08若在flow_04共享代码变更后失败，先按“普通course QA兼容性回归”诊断；不得预先把它计为第二个已确认产品缺陷。

## 6. flow_12：`AMEND_ORACLE`

不修改认证、JWT、Streamlit session或前端存储代码，不开发刷新持久登录。

后续只按以下口径补齐尾段，学生和教师各一次：

- 同一Streamlit会话正常rerun保持当前账号及适用筛选。
- 浏览器硬刷新允许回到登录门。
- 同账号重新登录后筛选可重置，但只能看到授权课程/班级。
- 退出后的学生/教师深链不显示已认证内容并要求重新登录。
- 任一阶段不得出现跨课程、跨班级或前一账号数据。
- 不把JWT写入URL、普通localStorage或页面可读明文。

若未来用户明确新增“硬刷新后无缝保持登录/筛选”，必须另立`NEW_FEATURE_REQUEST`，采用服务端会话或`HttpOnly + Secure（适用时）+ SameSite` cookie并重新做授权校验；该功能不阻断本轮收尾。

## 7. 实施清单（等待8C批准后才可执行）

### 7.1 允许的产品文件

- `backend/agent/orchestrator.py`
- `backend/llm/schemas.py`
- `backend/llm/template_provider.py`
- `backend/guards/citation_guard.py`

除非8C给出明确替换条款，8D不得增加其他产品文件。

### 7.2 允许的测试文件

- 新建`tests/test_practice_grounding_contract.py`
- 修改`tests/test_llm_provider.py`
- 修改`tests/test_r3_grounded_answer_contract.py`
- 修改`tests/test_api_endpoints.py`

`tests/test_agent_resume.py`默认只运行、不修改；只有8C发现计划意外影响continue并明确批准时才可调整。

### 7.3 明确禁止修改

- `backend/rag/hybrid_retriever.py`
- `backend/storage/logger.py`和所有数据库迁移
- `backend/main.py`公共端点契约
- `frontend/`、认证/JWT实现、`.env`、全局配置和业务超时
- MCP/真实工具实现、Planner策略和continue状态机
- H5、F01-F15及既有`PASS_SAVED/PASS_LOCKED_REUSED`证据

## 8. 回归与证据复用边界

- H5、F01-F15继续`PASS_LOCKED_REUSED`。
- browser precheck、flow_01、02、03、05、07、09、10、11和权限矩阵继续`PASS_SAVED`。
- 只运行本计划3.6列出的相关正式测试；不执行完整V6。
- 浏览器只补flow_04正/负、修订后的flow_08、一个flow_03引用smoke和一个引用负例。
- flow_06不因DeepSeek波动重开；flow_12只按修订口径补尾段。
- 只有实施实际触及数据库Schema、认证核心、配置加载或公共API（本计划均禁止）时，才允许8C重新评估是否扩大回归。

## 9. 8C必须反向审查的争议点

1. **证据重新水合位置**：在orchestrator内按授权chunk重新水合、而不修改`RetrievalResult`，是否充分修复240字符截断并避免全文泄露。
2. **核心事实与proposal边界**：`proposed_extensions`是否仍可能夹带事实；是否应在“仅根据资料”请求中完全禁止proposal，而不是仅显式标记。
3. **步骤数约束**：通用、有限的`required_step_count`解析是否必要且不会成为面向验收句的魔法字符串；若不采用，如何稳定证明冻结输入的六步逐项绑定。
4. **Template策略**：DeepSeek失败后是否允许Template在同一严格契约下作为显式fallback，还是practice应直接拒答；无论哪种，均不得把Template计为DeepSeek成功。
5. **Guard复用风险**：practice renderer模式是否与现有多外部来源grounded renderer完全隔离，避免破坏flow_03、混合来源引用、引用负例和`TOOL_TIMEOUT`语义。
6. **失败诊断隐私**：`failure_detail`白名单是否足以定位第一失败点，同时不泄露完整claim、quote、课程全文、URL敏感参数或密钥。
7. **完整性语义**：核心字段缺失时应整次ABSTAIN，还是允许字段级“证据不足”；冻结flow_04要求不得用proposal或模板补齐，8C需给出唯一可实施口径。
8. **允许文件是否最小**：特别审查是否真的无需修改`hybrid_retriever.py`、`logger.py`、公共API Schema或前端。

## 10. 8B停止点

裁决：`READY_FOR_REVERSE_REVIEW`

下一步只能把本计划交给用户和命令8C做反向技术/安全评审。未获得8C的`APPROVED_FOR_IMPLEMENTATION`前，不得修改产品或正式测试；本命令不自动执行8C。
