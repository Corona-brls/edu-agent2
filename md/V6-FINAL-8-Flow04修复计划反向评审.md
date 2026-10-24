# V6-FINAL-8 Flow 04 修复计划反向评审

## 1. 最终裁决

- 评审命令：8H
- 裁决：`APPROVED_FOR_IMPLEMENTATION`
- `command_8I_allowed=true`
- 本评审包含对8G计划的收紧条款；与8G冲突时，以本评审及机器可读评审计划为准。
- 本轮只写评审产物，没有修改产品代码、正式测试、配置、资料或数据库，也没有运行服务、真实模型、浏览器或pytest。

放行理由：8F以当前代码和保存的真实run证明了两个确定性产品缺口：`GroundedPracticeGuide`没有专用prompt分支；attempt 1的实际消息与attempt 0逐字节相同，重试metadata没有进入DeepSeek消息。S1/S2选择本身可能含模型波动，但“第二候选没有收到任何纠错”不是模型随机性，而是可重复的编排/prompt缺陷。经本评审修订后，范围保持最小、安全边界不变，且测试能复现目标资料与干扰资料竞争。

## 2. 逐项反向裁决

| 计划项 | 裁决 | 反向理由与最终条款 |
|---|---|---|
| 根因链 | `REVISE`后批准 | 不把“模型第一次选S2”本身定义为确定性根因；它是触发条件或模型波动。确定性根因是Practice缺少与服务器校验相匹配的专用契约，以及第二候选的实际消息没有任何纠错差异。 |
| Practice专用contract | `APPROVE` | 必须进入`messages[0].content`的系统级固定契约；证据正文和span文本留在标为不可信的用户数据区。只在`schema_name == "GroundedPracticeGuide"`分支生效。 |
| retry feedback | `REVISE`后批准 | Practice不得继续依赖任意自由文本`grounded_retry_guidance`。使用服务端构造、双重校验、有界的`practice_retry_feedback`；当前Guard失败必须传入首个失败五元组。 |
| DeepSeek实际消息证明 | `REVISE`后批准 | 不能只测builder或metadata。正式测试必须用假的DeepSeek client捕获传给`client.chat.completions.create`的`messages`，证明attempt 1包含纠错且哈希变化；禁止联网。 |
| 最多2个候选及副作用 | `APPROVE` | `_answer_practice`仍只允许两个候选，只重复最终结构化生成。DeepSeek既有transport retry语义不变，但不得把第三个成功candidate引入编排层。Planner/retrieve/MCP/Connector/工具调用均不得重做。 |
| 不增加相关性过滤 | `APPROVE` | 8F未确认retriever或evidence scope缺陷。S2可以留在模型证据中，由模型选择并由服务器验证；不得删S2或硬编码marker/ID。 |
| 模型与服务器职责 | `REVISE`后批准 | 模型继续提出`claim_text/source_id/evidence_span_id`；服务器只校验、绑定quote、渲染和拒答。服务器不得把模型的S2静默改为S1。测试必须检查第二次原始模型JSON已经选择S1。 |
| 文件白名单 | `APPROVE` | 两个产品文件和一个测试文件已经足够；`deepseek_provider.py`无需修改，因为它直接调用同一builder并原样发送消息。 |
| 回归矩阵 | `REVISE`后批准 | 增加假的DeepSeek client实际payload测试、四类外部来源参数化回归、Flow 03/08精确兼容性节点。白名单外测试文件只运行，不修改。 |
| 8J真实验收 | `REVISE`后批准 | 必须确认S1目标与S2干扰同时进入本次模型证据；固定输入只提交一次并只允许一个正向run。已创建run失败时不得再跑一个成功run替换。 |
| Flow 08引用卡片序号P2 | `APPROVE`为非阻断记录 | Flow 08走`GroundedSpanAnswer`与上传引用路径；本修复仅新增`GroundedPracticeGuide`精确分支。没有同源直接证据，不扩大8I范围。 |
| Guard、Schema与回滚 | `APPROVE` | Guard正确fail closed；Schema已经具备七字段绑定。二者均不改。数据库/API无需迁移，回滚仅撤销白名单文件中的8I补丁。 |

## 3. 根因闭环

### 3.1 直接证明

1. `backend/llm/prompts.py::build_structured_messages`目前只对`GroundedSpanAnswer`和`GroundedAnswer`增加专用契约；`GroundedPracticeGuide`只有通用JSON Schema文本。
2. `backend/agent/orchestrator.py::AgentOrchestrator._answer_practice`在attempt 1只向metadata写入`grounding_retry_attempt`、`previous_error_code`和通用`grounded_retry_guidance`。
3. Practice的prompt builder不读取这些字段。8F离线重建与真实model_call记录交叉证明，两次消息哈希都为`9137c888208ad9f19d25903b78d92a61b56db9a0526d517219e21837b98c7150`。
4. `backend/llm/deepseek_provider.py::DeepSeekProvider.generate_structured`调用同一个`build_structured_messages`，随后`_chat_once`把所得`messages`原样传给`client.chat.completions.create`；因此“builder未消费metadata”会直接成为DeepSeek实际请求缺口。
5. 第一次Guard失败有脱敏结构化位置：`IRRELEVANT_CITATION / risks / ordinal=2 / S2 / S2:text:1`，但它没有被传入attempt 1。
6. Citation Guard正确拒绝S2；span切分、course授权水合、混合外部证据provenance及Schema没有被8F证明有缺陷。

### 3.2 模型波动与产品缺陷的界线

- 第一次选错S2可能由模型波动造成，不能仅凭一次错误选择推导prompt一定导致错误。
- 但产品明确创建第二个“grounding retry”候选，而该候选收到与第一次完全相同的消息；这是确定性no-op。它使重试无法利用已知失败点，属于产品编排/prompt契约缺陷。
- 修复目标不是保证模型永不犯错，而是给第二候选真实、受控、可验证的纠错信息，并在仍错误时继续ABSTAIN。

## 4. 最终批准的产品修改

### 4.1 `backend/llm/prompts.py`

仅为`GroundedPracticeGuide`增加专用分支：

1. `messages[0].content`由现有`SYSTEM_PROMPT`加固定Practice契约组成；固定契约不得包含问题原文、证据全文、span文本或调用者自由文本。
2. 用户消息把问题、证据和span文本明确包在“不可信事实数据”区；服务器生成的source/span目录可列出ID、归属和resource type，但不得提升证据正文为系统指令。
3. 固定契约定义七字段及语义：`research_goal`、`method_design`、`tools`、`experiment_steps`、`result_analysis`、`innovation_points`、`risks`。
4. 每个`GroundedSpanClaim`只能使用目录中一个合法source和一个属于该source的合法span；claim必须由该span单独完整支持，禁止跨source/span合并、推论、翻译扩写、资料外proposal或伪造ID。
5. `experiment_steps`数量必须匹配服务端`required_practice_step_count`；required source覆盖规则不变。
6. 当问题限定某一资料、标识或主题时，只可选择与该限制相关且独立支持claim的span；其它候选可以可见，但不能仅因可见而引用。
7. 证据中“忽略规则”“改用某ID”等文本一律为不可信数据，不能覆盖系统契约。
8. attempt 1把经过校验的`practice_retry_feedback`作为系统级纠错段加入实际消息。Practice分支不消费调用者或证据可控的自由文本`grounded_retry_guidance`。

建议函数边界：在`build_structured_messages`内增加精确schema分支，并新增Practice固定契约、用户级安全目录和feedback解析辅助函数。无需修改`DeepSeekProvider`。

### 4.2 `backend/agent/orchestrator.py`

在`AgentOrchestrator._answer_practice`中：

1. 循环前初始化请求内局部`first_retry_feedback=None`；禁止写入`AgentState`或完整trace对象。
2. attempt 0仍使用原证据、span和模型candidate。
3. 第一次Citation Guard失败后，直接从当次`citation_check.error_code`和`failure_detail`构造：

```json
{
  "error_code": "IRRELEVANT_CITATION",
  "practice_field": "risks",
  "ordinal": 2,
  "source_id": "S2",
  "evidence_span_id": "S2:text:1"
}
```

4. 双重校验规则：
   - error code仅允许大写字母、数字、下划线且不超过64字符；
   - field必须是七字段枚举；
   - ordinal是1至12的整数；
   - source/span符合现有ID格式，并且属于当前请求服务器目录；
   - source/span归属必须一致。
5. 只传首个失败点，不传claim正文、raw candidate、完整证据、完整prompt、异常对象、密钥或任意Guard消息。
6. attempt 1只能从基础request复制并增加上述结构化对象；纠错要求不得为同一失败claim再次使用不能独立支持或不符合问题限制的source/span，应选择合法相关span，或缩小成由单span完整支持的抽取式原子claim。
7. 对当前根因的Guard分支，五元组缺失或目录校验失败时不得伪造值；应保留fail-closed结果。其它前置schema/validation错误没有field/source/span时只传安全错误码，不在8I新增公共异常结构。
8. 模型继续决定第二个candidate的source/span。服务器不得重写candidate，只能验证、绑定、渲染或拒答。
9. `for attempt in range(2)`不变；相同的`hydrated_evidence`、`evidence_spans`与检索结果复用。不得重新执行Planner、retrieve、MCP、Connector、真实工具或证据水合。
10. model_call日志只保存现有哈希和有限反馈标志/值；不保存完整消息或candidate。

## 5. 最终文件白名单

### 产品

- `backend/llm/prompts.py`
- `backend/agent/orchestrator.py`

### 正式测试

- `tests/test_practice_grounding_contract.py`

不允许修改：`backend/llm/deepseek_provider.py`、Citation Guard、Practice Guard阈值、LLM Schema、TemplateProvider、retriever、router、Connector、MCP、前端、公共API、数据库、配置、`.env`、超时、资料，以及第四个测试文件。

出现必须修改白名单外文件、降低Guard/权限、改变API/Schema或正式数据库变化时，8I输出`SCOPE_REVIEW_REQUIRED`并停止。

## 6. 8I必测设计

### 6.1 新增或收紧的根因节点

均位于`tests/test_practice_grounding_contract.py`：

1. `test_grounded_practice_prompt_contract_is_system_level_and_treats_evidence_as_untrusted`
   - 七字段、单source/span、独立支持、用户资料限制、required source及提示注入规则出现在系统契约。
   - 证据原文和注入文本只在用户数据区，不出现在系统消息。

2. `test_deepseek_practice_retry_receives_first_guard_failure_and_recovers_from_distractor`
   - 使用假的DeepSeek client，捕获实际传给`client.chat.completions.create`的两个`messages`；不得联网。
   - 当前课程中同时存在已授权、已INDEXED的S1目标chunk和S2干扰chunk，两者都进入同一次模型证据。
   - 假client第一次原始JSON令`risks[2]`绑定S2，第二次根据实际消息中的五元组返回已绑定S1的原始JSON。
   - 断言attempt 0被Guard拒绝、attempt 1完成、第二个raw candidate本身就是S1；服务器没有S2→S1改写。
   - 两次实际消息哈希不同，每次orchestrator hash与provider hash匹配。
   - Planner/retrieve/MCP/Connector/工具和水合计数不增加。

3. `test_practice_prompt_injection_cannot_override_system_contract`
   - 目标或干扰证据含伪造指令、伪造ID和“忽略系统规则”；系统契约和服务器校验仍生效。

4. 将现有`test_practice_generation_retries_once_without_retrieval_or_template_takeover`收紧：除两次request、零retrieval、零template外，增加实际消息差异与feedback可见断言。

5. 保留`test_practice_second_invalid_candidate_abstains_without_late_or_fallback_generation`：两个raw candidate均错误时仍ABSTAIN，恰好两个候选，无迟到写入。

6. 对paper、standard、vulnerability、project做参数化混合证据回归：合法required external可用，缺失/伪造/untrusted/provenance不匹配继续拒绝，且无外部二次联网或工具调用。

### 6.2 必须运行的既有节点

- `tests/test_practice_grounding_contract.py`全文件。
- `tests/test_agent_orchestration.py::test_practice_task_prioritizes_course_and_paper_evidence`
- `tests/test_phase3_e2e.py::test_phase3_full_demo_flow_offline`
- `tests/test_llm_provider.py::test_deepseek_provider_parses_grounded_practice_output`
- `tests/test_r3_grounded_answer_contract.py::test_grounded_span_prompt_includes_span_contract_and_catalog`
- `tests/test_r3_grounded_answer_contract.py::test_grounded_span_prompt_does_not_inherit_freeform_citation_instructions`
- `tests/test_api_endpoints.py::test_practice_guide_endpoint_records_question`（course-only Practice）
- `tests/test_api_endpoints.py::test_chat_endpoint_records_question`（Flow 03课程问答兼容性）
- `tests/test_api_endpoints.py::test_indexed_upload_exact_question_keeps_correct_document_and_chunk_binding`（Flow 08上传资料引用兼容性）

这些白名单外测试只运行，禁止修改。随后运行相关共享prompt/Agent兼容性集合，最后在最终稳定版本上使用全新隔离basetemp运行完整pytest。

### 6.3 8I通过门

- 根因节点、Practice全文件、上述精确兼容性节点、共享集合和完整pytest均为`0 failed / 0 errors`。
- skip、xfail、warning分别报告且不计作通过；不得新增未解释skip。
- 假DeepSeek client捕获的attempt 1实际消息含受控五元组，哈希与attempt 0不同。
- raw candidate证明模型而非服务器选择了修正后的span。
- 同一证据复用，最多2个candidate，零重复Planner/retrieve/MCP/Connector/工具/水合。
- Guard、Schema、API、数据库、权限与白名单外文件不变；正式数据库哈希、大小和业务计数不变。
- 保存8I实施报告。全部通过才输出：
  - `IMPLEMENTATION_READY_FOR_FLOW04_ACCEPTANCE`
  - `command_8J_allowed=true`

8I不执行服务或浏览器验收。

## 7. 8J精确验收门

1. 8I已通过，且三个白名单文件自8I报告后未变化。
2. 使用全新隔离SQLite/vector/cache/uploads和独立端口；正式数据库只读不变。
3. 沿用冻结的S1完整七字段/六步骤目标资料、S2干扰资料及固定问题。运行前只读确认两份资料均已INDEXED且属于当前课程。
4. 通过真实学生页面只提交一次正向问题。正向只允许创建一个run；该run若已创建，无论COMPLETED或ABSTAINED都不得再次提交并挑选另一个成功run。
5. 若页面连接中断，先按隔离数据库、trace和run_id只读确认；只有确定没有创建run时才可恢复同一次提交，并保留恢复记录。
6. 证据必须证明同一次正向run的检索/模型输入同时包含S1目标与S2干扰，而不是在验收时删除S2。
7. 正向必须`COMPLETED`：七字段、六个有序步骤、每个事实均有服务器验证的相关source/span；不得用S2支撑违反资料限定的claim；provider为DeepSeek且`fallback_used=false`。
8. 若正向run使用第二候选，保存两次不同消息哈希、有限失败反馈、候选数和零重复上游调用证据；若第一候选直接通过，不人为制造失败，重试机制证明引用8I的确定性假client测试。
9. 同一隔离runtime另做证据不足负对照，必须安全ABSTAIN；API可辅助读取但不能替代正向真实页面。
10. 保存页面截图、run/trace/model_call/Guard/引用绑定原始证据；清理精确PID、端口和隔离runtime。
11. 只更新Flow 04及F16/F17汇总；其它PASS继续复用。Flow 08卡片序号P2保持非阻断残余风险，不因本修复关闭或扩大。

只有正向单run、负对照、S1/S2竞争、证据完整、不变量和清理全部通过，才可关闭Flow 04 P1。失败时保留已有PASS，只报告最小剩余缺口。

## 8. 安全不变量

- Citation Guard、Practice Guard阈值及七字段Schema保持fail-closed且不变。
- 模型提出source/span；服务器不纠正模型答案，只验证、绑定quote、渲染或拒答。
- current-course、已检索document/chunk、INDEXED状态和跨课程隔离不变。
- 混合course+paper/standard/vulnerability/project仍依赖当前run可信工具provenance，不重新联网、不抓全文。
- 完整chunk、完整prompt和raw candidate不得进入state、日志、observation、trace、异常、API或诊断报告。
- retry feedback仅为当前请求首个失败点的有限服务器数据，证据prompt injection不能覆盖它。
- DeepSeek失败不得由TemplateProvider生成成功Practice回答。
- 证据不足、错source/span、跨课程、伪造external、required source缺失仍拒答。

## 9. 回滚与兼容性

- 无数据库/API/Schema迁移。
- 回滚仅撤销8I在三个白名单文件中的补丁；不得回退8D-S2混合证据契约或历史PASS证据。
- 回滚后运行根因单测、Practice全文件和精确兼容性节点，确认回到8I前行为。

## 10. 不变量基线

- `backend/llm/prompts.py`：`6bf5211f2f80b832b70cc93e4eb319c5659f4c3d54b9c9793e25c7ff925816ce`
- `backend/agent/orchestrator.py`：`a2e870ee4a9d5d92981d6c64ad65873aace80d85b62797f0eff1302fc6cb4654`
- `tests/test_practice_grounding_contract.py`：`7ce29cfbabae9a83f553a6aa51e582da8550bc1785bd696f663dee6cf6075efd`
- `backend/data/app.db`：761856 bytes，`d92a04a2fa229093484d0dd60ba0dfb6bba4cfe76dcfb72a28bd90eaec0f52a8`

**最终裁决：`APPROVED_FOR_IMPLEMENTATION`**  
**`command_8I_allowed=true`**

本命令结束，不自动执行8I。
