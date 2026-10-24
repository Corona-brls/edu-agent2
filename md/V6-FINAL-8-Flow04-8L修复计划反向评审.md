# V6-FINAL-8 Flow 04 8L修复计划反向技术与安全评审

## 1. 最终裁决

`APPROVED_FOR_IMPLEMENTATION`

`command_8P_allowed=true`

8N方向正确，但本评审对四处条款作了约束性修订：完整`exact_text`是规范默认；连续片段必须逐字符复制；首tuple只说明错误模式而非完整错误列表；每次拒绝的完整脱敏诊断以model-call metadata为唯一规范载体。8P必须以本评审版为准。

## 2. 反向挑战结论

| 评审项 | 裁决 | 理由 |
|---|---|---|
| 根因是否只是模型随机性 | `APPROVE` | 8M直接证明prompt规则与Guard严格token子集接受条件之间缺少可操作的抽取契约。模型波动是触发条件，不是唯一根因。 |
| 逐字抽取契约 | `REVISE`后批准 | 完整`exact_text`为默认；连续片段只能是模型所见`exact_text`中的同语言、连续、逐字符复制片段。自由缩写或语义等价均不允许。 |
| `UNSUPPORTED`与`IRRELEVANT`分流 | `APPROVE` | `UNSUPPORTED`发生前，Guard已确认source/span存在、同源、quote一致且相关，因此保留绑定、纠正claim合理；`IRRELEVANT`与非法绑定必须重选。 |
| 同span纠正是否绕过Guard | `APPROVE` | attempt 1仍经过完整Practice/Citation Guard；保留span不会改变授权或证据，且不能让资料外文本通过严格支持谓词。 |
| 首tuple加全量自审 | `REVISE`后批准 | 首tuple不能枚举其他错误，只能告知失败模式。第二候选必须重建完整JSON，对七字段和六步逐项应用同一抽取规则；仍不合规就拒答。 |
| 服务器生成/替换claim | `REJECT` | span没有可信的服务器字段语义标签。服务器复制span可能把正确证据放入错误字段，并改变事实作者和审计责任。本轮禁止。 |
| 最终失败可观测性 | `REVISE`后批准 | 每个attempt的完整allowlist诊断写入对应model-call metadata；trace只保留attempt、错误码和安全locator摘要，避免两份完整对象漂移。 |
| 真正DeepSeek消息测试 | `APPROVE` | 必须由fake DeepSeek client捕获传给`chat.completions.create`的messages；仅测builder或metadata不成立。 |
| 文件范围 | `APPROVE` | 8M没有Guard、Schema或provider缺陷证据，两个产品文件和一个测试文件足够。 |
| 8Q范围 | `APPROVE` | 只重验Flow 04正向；不得重新消费其余锁定PASS。 |

## 3. 冻结文件白名单

产品：

- `backend/llm/prompts.py`
- `backend/agent/orchestrator.py`

正式测试：

- `tests/test_practice_grounding_contract.py`

禁止修改Citation Guard、LLM Schema、DeepSeek provider、TemplateProvider、retriever、router、Connector、MCP、公共API、前端、数据库、配置、`.env`、超时、权限、课程隔离、验收资料或第四个测试文件。确需越界时必须输出`SCOPE_REVIEW_REQUIRED`。

## 4. 最终允许实施条款

### 4.1 Practice抽取contract

仅修改`GroundedPracticeGuide`专用系统contract：

1. `claim_text`规范默认值为模型所见的完整`EvidenceSpan.exact_text`。
2. 受限备选仅为同一`exact_text`内同语言、连续、逐字符复制的片段。
3. 禁止翻译、同义改写、总结、补主语/结论、重排token、添加步骤前缀、跨span拼接或资料外proposal。
4. 短步骤span使用完整文本；列举型工具span优先形成一个完整claim，避免将一个span自由改写成多项。
5. 每个claim仍由模型明确提交一个合法source和其所属span；服务器不生成或覆盖claim。

证据正文继续处于不可信user data区域，不得复制到system message。prompt injection不能覆盖该contract。

### 4.2 Retry错误分流

- `UNSUPPORTED_CITATION`：该错误来自Guard且binding已经验证时，保留同一source/span，将claim改为规范抽取。如果该span实际上无法满足字段语义，模型可改选另一个合法相关span，但不得保留原不支持文本。
- `IRRELEVANT_CITATION`：为对应字段改选合法相关span。
- unknown source/span或source mismatch：丢弃非法组合，从服务器目录重选。
- 其他结构/绑定错误：按基础contract重建完整对象，不猜测标识符。

attempt 1必须完整重建七字段JSON，并重新审计全部六步、required source coverage、重复claim和每项抽取性。服务端仍只发送一个脱敏首失败tuple；该tuple不是完整错误列表。最多两个候选，只重复最终结构化生成。

### 4.3 每候选失败可观测性

规范记录位置为对应`model_call.metadata.candidate_rejection`，每个被拒候选恰有一个对象：

- `grounding_retry_attempt`
- `error_code`
- `practice_field`
- `ordinal`
- `source_id`
- `span_id`
- `claim_length`
- `claim_sha256`
- `normalized_claim_sha256`
- 布尔`normalization_flags`
- `failure_detail_available`

Guard拒绝时使用其已脱敏failure detail。Practice预校验能定位首claim时填充；不能定位时使用空locator、`ordinal=0`和`failure_detail_available=false`，不得写异常正文。

每个attempt可以另写一条trace摘要，但不得复制完整诊断对象。trace只允许attempt、error code、字段/ordinal及source/span locator。原始`claim_text`、候选JSON、渲染答案、完整chunk、完整system prompt、密钥和Authorization header均禁止落盘。

删除当前只持久化首轮Guard失败的`guard_failure_traced`抑制语义，确保最终attempt tuple可恢复。

### 4.4 明确不采用服务器抽取

服务器继续控制授权、span目录、quote绑定、验证和canonical renderer，但不得从span生成、替换、截断、补全或“规范化”模型的`claim_text`，也不得把S2静默改成S1。

如未来选择服务器生成claim，必须作为独立契约变更，先补充字段语义验证、审计责任和API表现评审；不属于8P。

## 5. 必测矩阵

### 5.1 先红后绿根因节点

在`tests/test_practice_grounding_contract.py`增加：

1. prompt明确要求完整exact或连续逐字片段；现状先红。
2. 正确span加不支持改写在attempt 0被拒；attempt 1实际消息要求保留span并逐字抽取，最终通过。
3. attempt 0同时存在多个改写错误；attempt 1实际消息要求重新审计整个候选。
4. attempt 1仍失败时，其最终failure tuple进入规范metadata。
5. 递归检查日志、trace、state与API序列化内容，不含候选正文和完整chunk。

### 5.2 定向验证

- fake DeepSeek client捕获的两次真实messages哈希不同。
- attempt 1 system message包含分类纠错和七字段/六步全量审计。
- `IRRELEVANT`仍从S2改选合法span；unknown/mismatch不保留非法组合。
- 正确span上的自由改写仍被Guard拒绝，证明阈值未放宽。
- 两次均错误仍ABSTAIN，无Template接管。
- 两次生成期间Planner、retrieve、MCP、Connector、tool次数不增加。
- 每个拒绝attempt恰有一个规范diagnostic对象，最终tuple可恢复。

### 5.3 Practice定向回归

运行`tests/test_practice_grounding_contract.py`全文件，要求0 failed、0 errors、0 skipped、0 xfail。必须覆盖course-only、混合paper/standard/vulnerability/project、S2干扰、forged/unknown source、wrong/mismatched span、duplicate、证据不足、prompt injection、跨课程、非INDEXED、伪造外部provenance、两候选失败和无Template接管。

### 5.4 精确兼容性节点

- `tests/test_agent_orchestration.py::test_practice_task_prioritizes_course_and_paper_evidence`
- `tests/test_phase3_e2e.py::test_phase3_full_demo_flow_offline`
- `tests/test_llm_provider.py::test_deepseek_provider_parses_grounded_practice_output`
- `tests/test_r3_grounded_answer_contract.py::test_grounded_span_prompt_includes_span_contract_and_catalog`
- `tests/test_r3_grounded_answer_contract.py::test_grounded_span_prompt_does_not_inherit_freeform_citation_instructions`
- `tests/test_api_endpoints.py::test_practice_guide_endpoint_records_question`
- `tests/test_api_endpoints.py::test_chat_endpoint_records_question`
- `tests/test_api_endpoints.py::test_indexed_upload_exact_question_keeps_correct_document_and_chunk_binding`

再运行共享集合：`test_r3_grounded_answer_contract.py`、`test_llm_provider.py`、`test_agent_orchestration.py`、`test_j3_agent_evidence.py`、`test_phase3_e2e.py`和`test_api_endpoints.py`。白名单外测试只运行，禁止修改。

### 5.5 完整回归

最终稳定版本使用全新隔离basetemp运行一次完整pytest，要求0 failed、0 errors。skip、xfail和warnings分别报告，均不计作通过；不得新增无法解释的skip。

## 6. 8P实施通过门

只有以下条件全部满足，8P才能输出`IMPLEMENTATION_READY_FOR_FLOW04_ACCEPTANCE`和`command_8Q_allowed=true`：

- 红测已保存且修复后全部转绿；
- 定向、Practice、精确兼容性、共享集合和完整pytest全部满足门槛；
- 真正fake-client messages与分类retry已证明；
- 最终attempt失败tuple可恢复，隐私递归断言通过；
- Guard、Schema、provider、API和所有未授权文件不变；
- 正式数据库哈希、大小、mtime和业务计数不变；
- 8P未启动服务、真实模型或浏览器。

## 7. 8Q唯一浏览器范围

8Q只重验Flow 04正向。H5、F01至F15、其他11项页面流程、权限矩阵、8P自动化回归和8J证据不足负对照全部复用。

验收必须：

1. 全新隔离runtime中同时INDEX完整S1目标资料和S2干扰资料，并证明两者进入检索/模型证据。
2. 真实学生页面只提交一次固定问题，只允许一个正向run；语义失败后不得另建run挑成功。
3. PASS要求`COMPLETED/ANSWERED`、DeepSeek、`fallback=false`、七字段、恰好六步、全部source/span合法且未修改Guard通过。
4. 若发生retry，第二条真实消息必须含分类纠错和全候选审计，哈希不同且无上游重放；若attempt 0直接通过，不强制制造失败，复用8P的确定性retry测试。
5. 任何候选进入Guard后的语义失败均保存单一run和最终tuple，裁决`PRODUCT_SEMANTIC_FAILURE`，禁止替换run。
6. transport在任何候选/Guard之前失败时，只能按8Q命令明确授权的环境分支处理，不得临时发明业务重试。
7. PASS后只替换F16/F17中的Flow 04当前状态；8L失败保留历史附录，Flow 08 P2风险继续保留。

## 8. 安全不变量与回滚

- Citation/Practice Guard代码与阈值保持不变并继续fail-closed。
- Schema、DeepSeek provider、公共API和数据库Schema不变。
- 模型仍是claim作者和source/span选择者；服务器不静默修复。
- 课程授权、INDEXED校验、跨课程隔离和外部provenance不变。
- 完整chunk、原始候选和完整prompt不进入持久状态、日志、trace、API或诊断。
- evidence prompt injection不能覆盖系统contract和retry纠错。
- Template不能把DeepSeek失败变成成功。

回滚仅撤销8P在三个白名单文件中的增量，恢复8I冻结哈希；不得回退8D-S2混合证据契约或8I消息传播修复。无需数据库迁移。

**最终裁决：`APPROVED_FOR_IMPLEMENTATION`**

**`command_8P_allowed=true`**
