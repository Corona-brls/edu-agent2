# V6-FINAL-8 Flow 04 8L剩余语义缺陷最小修复计划

## 1. 状态与范围

- 8M状态：`DIAGNOSIS_COMPLETE`
- 8M后续门：`command_8N_allowed=true`
- 本计划状态：`READY_FOR_REVERSE_REVIEW`
- 允许进入8O：`command_8O_allowed=true`
- 业务缺陷数：仍为一个Flow 04正向阻断；三个贡献机制不另行计数。

本轮只形成计划。未修改产品、测试、配置、`.env`、Schema或数据库，未运行pytest、模型、服务或浏览器。

## 2. 8I已完成能力与8L新增问题

8I已经完成并继续保留：

- `GroundedPracticeGuide`专用系统级contract真实进入DeepSeek消息；
- 证据正文留在标记为不可信数据的user message；
- 脱敏首失败tuple进入第二次实际请求，两次消息哈希不同；
- 最多两个候选，生成重试不重复Planner、检索、MCP、Connector或真实工具；
- 模型提交`claim_text/source_id/evidence_span_id`，服务器只验证、绑定quote、渲染或拒答；
- Guard、Schema及Template fallback均未放宽。

8L新暴露且由8M确认的剩余问题：

1. `PRACTICE_CLAIM_EXTRACTIVENESS_CONTRACT_INSUFFICIENT`：contract没有把Guard的严格token子集条件转成足够确定的逐字抽取规则。
2. `UNSUPPORTED_RETRY_SPAN_POLICY_CONFLICT`：当前消息错误地把`UNSUPPORTED_CITATION`与`IRRELEVANT_CITATION`都解释为必须换span。
3. `SINGLE_FAILURE_RETRY_FEEDBACK_INCOMPLETE`：只提示首项，未要求重新审计七字段和全部六步。
4. `FINAL_GUARD_FAILURE_OBSERVABILITY_GAP`：只持久化首候选失败，最终候选的failure tuple丢失。

## 3. 最小产品改动

### 3.1 `backend/llm/prompts.py`

收紧`_grounded_practice_guide_contract`，仅影响`GroundedPracticeGuide`：

- `claim_text`默认逐字复制所选span的完整`exact_text`。
- 只有span包含多个可独立事实时，才允许复制同语言、连续且未改写的原文片段。
- 禁止翻译、同义改写、概括、补主语、补结论、重排token、加入步骤前缀或资料外词语。
- 短步骤span应使用完整原文；一个列举型工具span优先形成一个claim，避免把同一span自由改写成多个工具claim。
- 每项仍只能绑定目录中的一个source和属于该source的一个span。

修改`_grounded_practice_guide_contract`中的retry段和`_safe_practice_retry_feedback`消费逻辑，按错误类型给出不同动作：

- `UNSUPPORTED_CITATION`且绑定已由Guard确认合法：保留同一source/span，把claim改成完整`exact_text`或合规连续片段。
- `IRRELEVANT_CITATION`：为对应字段改选合法、相关span。
- unknown source/span或source mismatch：丢弃非法绑定，从服务器目录重新选择。
- 其他结构或绑定错误：按基础contract完整重建，不猜测或伪造ID。

无论首错误是什么，attempt 1都必须重新输出完整七字段对象，并逐项自审：七字段、全部六步、required source coverage、重复claim、单span绑定及逐字抽取性。仍只发送一个脱敏首失败tuple，不发送完整候选或完整失败列表。

### 3.2 `backend/agent/orchestrator.py`

保持`_answer_practice`的两候选循环和同一证据不变，仅补齐分类与可观测性：

- 将每次候选拒绝都关联明确`grounding_retry_attempt`。
- Guard拒绝时，从`CitationCheck.failure_detail`复制固定allowlist：`error_code`、`practice_field`、`ordinal`、`source_id`、`span_id`、`claim_length`、`claim_sha256`、`normalized_claim_sha256`、布尔`normalization_flags`。
- attempt 0和最终attempt 1都持久化；删除当前`guard_failure_traced`仅允许首轮trace的抑制语义。
- Practice结构/绑定预校验失败使用同一诊断结构。能安全定位首个claim时填入字段和哈希；不能定位时使用空ID、`ordinal=0`及`failure_detail_available=false`，不得把异常正文或候选内容写入日志。
- provider调用成功与candidate被Guard接受继续分开表达；最终stop reason取最终候选真实错误码。
- retry仍只重新执行`generate_structured`，不得回到Planner、retrieve、MCP、Connector、证据水合或工具调用。

### 3.3 不采用服务器受控生成claim

本计划明确不让服务器根据模型选择的span生成或覆盖`claim_text`。现有服务器控制的quote绑定与canonical renderer继续保留，但事实claim仍由模型按逐字抽取contract提交，服务器只校验和拒答。

若8O希望服务器直接以span生成claim，这属于显式模型/服务器职责契约变更，必须重新定义审计语义并单独批准；不得以“规范化”为名静默实施。

## 4. 精确文件白名单

产品：

- `backend/llm/prompts.py`
- `backend/agent/orchestrator.py`

正式测试：

- `tests/test_practice_grounding_contract.py`

无需且禁止修改Citation Guard、LLM Schema、DeepSeek provider、TemplateProvider、retriever、router、Connector、MCP、公共API、前端、数据库、配置、`.env`、超时、验收资料或第四个测试文件。若实现证明必须越界，8P应输出`SCOPE_REVIEW_REQUIRED`。

## 5. 先红后绿测试矩阵

所有新增或修改测试均放在`tests/test_practice_grounding_contract.py`：

1. 正确span加不受支持改写claim：attempt 0必须被`UNSUPPORTED_CITATION`拒绝。
2. 同一合法span恢复：attempt 1实际消息明确要求保留该span并改成逐字抽取，原始模型JSON已使用同一span，最终通过。
3. 错误分流：`IRRELEVANT_CITATION`仍要求换span；unknown source/span与source mismatch要求从目录重选。
4. 多错误候选：attempt 0同时有多个改写错误；attempt 1消息必须要求重建并审计全部七字段和六步，而非只修首tuple。
5. 最终失败可观测：attempt 1仍失败时保存第二次的脱敏failure tuple，并保持ABSTAIN。
6. 实际客户端消息：fake DeepSeek client捕获传给`chat.completions.create`的messages；两次哈希不同，第二次包含分类纠错与全量自审指令。
7. 隐私：model-call metadata、trace、state和序列化API中均不存在原始候选JSON、claim正文或完整chunk。
8. 副作用：两次生成期间Planner、retrieve、MCP、Connector和tool调用计数不增加。

必须继续通过的负例与兼容性：

- Guard仍拒绝正确span上的不支持改写，证明阈值未放宽；
- forged/unknown source、错span、source mismatch、S2干扰、duplicate拒绝；
- 证据不足、prompt injection、跨课程、非INDEXED、伪造外部provenance拒绝；
- course-only及course+paper/standard/vulnerability/project通过；
- 普通`GroundedSpanAnswer`、Flow 03课程引用、Flow 08上传引用不继承Practice规则；
- Template不能接管DeepSeek失败。

## 6. 实施验证门

8O批准后，8P按以下顺序执行：

1. 新增根因节点并保存预修复红灯。
2. 在白名单内实施，重跑根因节点。
3. 运行`tests/test_practice_grounding_contract.py`全文件。
4. 运行共享prompt/Agent兼容性节点与Flow 03/08代码级smoke；白名单外测试只运行、不修改。
5. 最终稳定版本使用全新隔离basetemp运行一次完整pytest，要求0 failed、0 errors；skip、xfail和warnings单列，skip不计通过。

只有全部通过、正式数据库不变、白名单外文件不变、隐私断言通过时，8P才能输出：

`IMPLEMENTATION_READY_FOR_FLOW04_ACCEPTANCE`

并允许进入8Q。

## 7. 8Q最小浏览器验收

- 只重验Flow 04正向；其余历史PASS、权限矩阵、代码回归和既有证据不足负对照继续复用。
- 在全新隔离runtime中同时INDEX与8L同等语义的S1完整资料和S2干扰资料。
- 真实学生页面只提交一次固定问题；业务run创建后禁止另建run挑选成功结果。
- PASS要求DeepSeek、`fallback=false`、七字段、恰好六步、全部source/span合法且Guard通过。
- 若发生retry，必须证明：`UNSUPPORTED`保留合法span并纠正为逐字抽取；第二候选全量重审；消息哈希不同；没有重复上游调用。
- 成功后只用本批Flow 04证据更新F16/F17，保留8L失败附录；失败则保存该唯一run并停止。

## 8. 安全不变量与非目标

- Guard继续fail-closed，不改support、relevance、token或Practice阈值。
- 不删除S2、不改检索排序、不硬编码marker/source/document/中文答案。
- 不静默把S2改成S1，不生成或覆盖模型claim。
- 证据和prompt injection不能覆盖系统contract与服务端retry指令。
- 不记录密钥、Authorization header、候选正文或完整私密chunk。
- 最多两个候选；证据不足继续ABSTAIN；Template不得伪装DeepSeek成功。
- Flow 08引用卡片序号P2继续作为独立非阻断风险，本计划不处理。

## 9. 回滚

回滚只撤销8O批准后在三个白名单文件中的8P增量，恢复8I冻结基线：

- `backend/llm/prompts.py`：`010b1ed0285bf32c5ecf8c228ff1024fcc0df0fb39d8882e7e8e114dea625e6b`
- `backend/agent/orchestrator.py`：`0711bf2cd7dc1389e3d3fa189a7c77585d3c9ee1086db8132c349148cfb79063`
- `tests/test_practice_grounding_contract.py`：`c468198e62c0b219a8e546517a63864257df835e996e2d4f75ea8a803aab64e3`

不得回退8D-S2混合证据契约或8I已验证的专用Practice消息链。无需数据库/API/Schema迁移。

## 10. 8O必须反向挑战的争议点

1. “完整exact_text优先、连续原文片段为受限备选”是否与Guard严格token子集确定性一致。
2. `UNSUPPORTED`保留合法span、`IRRELEVANT/unknown/mismatch`重选的分流是否覆盖现有错误码且不会误导模型。
3. 只给首tuple但强制全候选重建/自审，是否足以覆盖多错误候选而无需泄露完整失败列表。
4. 脱敏failure tuple应同时写model-call metadata和trace，还是只保留一个规范载体、另一处保存引用ID以避免重复。
5. 是否确认拒绝服务器生成/覆盖`claim_text`；若批准该路线，必须作为显式契约变更重新评审。

`READY_FOR_REVERSE_REVIEW`

`command_8O_allowed=true`
