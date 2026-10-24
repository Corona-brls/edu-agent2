# V6-FINAL-8修复计划反向技术与安全评审

> 命令：8C——V6-FINAL-8修复计划反向技术与安全评审  
> 被评审计划：`md/V6-FINAL-8最小修复计划.md`  
> 最终裁决：`APPROVED_FOR_IMPLEMENTATION`  
> 日期：2026-10-10  
> 边界：本文件只评审并替换计划条款；未修改产品、正式测试、配置、`.env`、数据库、公共API、公共Schema或验收结果，未执行命令8D。

## 1. 评审输入和独立复核

已独立回读：

- 8A的`diagnostic_report.json`、`frozen_acceptance_inputs.json`、`source_chain_analysis.json`及五组原始诊断结果；
- 8B的`V6-FINAL-8最小修复计划.md`；
- `md/题目最终要求.md`和`md/最终三项缺陷收尾修复计划.md`中的A1、V6-R4-B2、8A至8E口径；
- `backend/agent/orchestrator.py`中的retrieve、answer、grounded retry、span验证、practice renderer和continue路径；
- `backend/rag/hybrid_retriever.py::RetrievalResult/_snippet`；
- `backend/guards/citation_guard.py`中的standard和grounded renderer分支；
- `backend/llm/schemas.py`、`backend/llm/template_provider.py`和DeepSeek structured output路径；
- practice、grounded citation、continue、会话与深链正式测试。

独立归因结论与8A一致：只有flow_04是已确认产品缺陷；flow_06、08、12不得转化为产品修复。

## 2. 对8B各项的裁决

| 8B项目 | 原始裁决 | 8C裁决 | 说明 |
|---|---|---|---|
| flow_04实践指导 | `FIX_PRODUCT` | `REVISE -> APPROVE` | 根因和文件范围正确，但proposal、Template fallback、完整chunk暴露边界及Guard隔离需收紧；替换条款见第4节 |
| flow_06 ask_user/continue | `NO_CHANGE / AMEND_ORACLE` | `REVISE` | 仍不改产品，但8E/F16必须补一条真实浏览器同-run continue证据，不能只保留8A API诊断 |
| flow_08上传资料问答 | `AMEND_ORACLE` | `NOT_A_PRODUCT_DEFECT / APPROVE` | 冻结资料/问题组合正确；只作兼容性回归和修订验收 |
| flow_12刷新与深链 | `AMEND_ORACLE` | `NOT_A_PRODUCT_DEFECT / APPROVE` | 历史口径正确；禁止开发持久登录 |
| 8B定向测试范围 | 定向回归 | `REVISE` | Agent/Citation Guard属于核心共享代码；定向测试通过后还必须在隔离环境运行一次完整pytest |

替换条款已经消除核心安全争议，因此最终状态不是`REVISE`停机，而是对下述`reviewed_repair_plan`给出`APPROVED_FOR_IMPLEMENTATION`。

## 3. 反向检查发现

### 3.1 正确且保留的设计

- 不降低Citation Guard阈值，不把独立文末引用行当作全文支持。
- source/span由服务器当前请求内证据生成，而不是相信模型输出。
- 重新水合仅匹配已检索的document/chunk，不扩大检索范围。
- 生成重试最多一次且只在`_answer`内部发生，不重跑Planner、retriever、MCP或真实工具。
- flow_06不硬编码自由语言触发；flow_12不开发持久登录。
- 不改数据库Schema、公共API或业务超时。

### 3.2 必须替换的设计

#### A. `proposed_extensions`不能进入本次实现

8B允许practice在独立`proposed_extensions`中输出无证据方案。即使加“非资料事实”标签，仍存在三个问题：

1. 冻结问题明确要求“仅根据资料、不要补充资料外事实”；无证据proposal与该约束冲突。
2. Guard可以验证标签和渲染一致性，却不能仅凭标签证明proposal没有夹带数字、实体、工具能力或安全结论。
3. proposal可能被前端或后续消费者误当作实践步骤，从而绕开核心字段的证据完整性。

替换裁决：本次`GroundedPracticeGuide`不包含任何无证据proposal字段。七个实践字段全部只能包含`GroundedSpanClaim`。若未来确需模型拟议方案，另立非阻断功能请求，定义显式API/UX语义和独立安全评审，不搭车修复flow_04。

#### B. DeepSeek失败后禁止Template形成practice成功终态

8B允许DeepSeek失败后Template在同一严格契约下作为fallback。虽然可以显示`fallback_used=true`，仍会把“真实模型未完成”与“本地模板抽取”混成同一个成功的实践生成路径，并可能重现8A观察到的错误归因。

替换裁决：

- 显式配置`LLM_PROVIDER=template`时，TemplateProvider作为primary可以生成严格extractive契约，供离线开发和确定性测试使用。
- primary为DeepSeek时，结构化候选/Guard失败只允许同一DeepSeek provider做一次最终生成重试。
- DeepSeek transport/provider错误经过现有provider层限次恢复后仍失败，practice直接`ABSTAINED / MODEL_ANSWER_FAILED`。
- DeepSeek失败后不得调用TemplateProvider产生practice成功回答；因此不存在“Template冒充DeepSeek成功”。
- 普通course QA和现有多来源grounded路径的fallback策略不在本批改变。

#### C. 完整chunk只能在服务器请求内使用，不得外溢

8B的orchestrator重新水合方向正确，但“完整chunk”必须限定：

- 只可从`document_chunks(course_id=state.course_id)`返回的已INDEXED记录中，按已经出现在本次`answer_evidence`里的`document_id + chunk_id`精确匹配。
- 不得因为同课程而把未检索chunk、同文档其他chunk或同课程其他文档加入模型上下文。
- 用`SafetyGuard.sanitize_untrusted_text`处理后，在服务器内切分成有界span；模型只接收本次选定chunk的有界span内容，不接收整篇文档或其他chunk。
- 完整chunk不得写回`state.evidence`，不得出现在retrieval observation、API citation snippet、trace、异常消息或测试失败快照中。
- Guard使用请求内水合副本；最终API citation仍从原短摘要证据生成。
- 使用chunk hash/请求内hash验证，不改变`RetrievalResult.to_dict()`，不修改`hybrid_retriever.py`。

该设计保持现有课程授权边界，不将水合变成第二次检索，也避免把全文加入公共/持久日志。

#### D. practice Guard必须与现有grounded Guard代码路径隔离

不得修改现有`_validate_grounded_renderer_units`的解析规则以兼容practice标题/编号。应在`CitationGuard.enforce`新增显式、互斥的`verified_practice_bindings`分支：

```text
if verified_practice_bindings is not None:
    validate_practice_renderer(...)
elif verified_spans/bindings is not None:
    existing grounded renderer validation
elif strict mixed-source:
    existing strict validation
else:
    existing standard validation
```

新分支只能由orchestrator服务器传入bindings时启用。不得靠答案文本、标题或marker自动猜测模式。现有grounded、standard、TOOL和refusal逻辑保持原样。

#### E. 失败诊断白名单进一步收窄

8B的`failure_detail.section`必须解释为固定字段枚举，而不是资料section文本。允许字段仅为：

```text
error_code
practice_field  # 固定枚举：research_goal等
ordinal
source_id
span_id
claim_sha256
normalized_claim_sha256
claim_length
normalization_flags  # 固定布尔/枚举集合
```

禁止保存raw claim、quote、chunk、文档标题、资料section、URL、prompt、token、JWT、API key或任意跨课程文本。仅首个失败点写入现有`agent_traces.metadata_json`。

## 4. reviewed_repair_plan

### 4.1 最终批准的产品修复：flow_04

#### 4.1.1 服务器授权证据水合

在`backend/agent/orchestrator.py`增加practice专用请求内水合：

1. 输入只能是本次Planner/retriever已产生的`answer_evidence`。
2. 对每个本地course证据，用当前`state.course_id`查询已INDEXED chunk，并精确匹配原`document_id/chunk_id`。
3. 任一course、document、chunk、index状态不一致，立即`ABSTAINED`，stop reason为`PRACTICE_EVIDENCE_SCOPE_MISMATCH`。
4. 水合副本只用于practice模型请求、span构造和Guard；不写回持久state，不改变外部citation摘要。
5. 本地`document_id=0`fallback只能使用服务器已读取的现有文本，不能伪造成数据库chunk；不足时拒答。

#### 4.1.2 最终`GroundedPracticeGuide`结构

在`backend/llm/schemas.py`增加内部Schema：

```text
research_goal: list[GroundedSpanClaim]
method_design: list[GroundedSpanClaim]
tools: list[GroundedSpanClaim]
experiment_steps: list[GroundedSpanClaim]
result_analysis: list[GroundedSpanClaim]
innovation_points: list[GroundedSpanClaim]
risks: list[GroundedSpanClaim]
```

不包含`proposed_extensions`或任何无source/span的自由正文字段。

验证规则：

- 七个字段至少各有一个合法、非重复、请求内source/span绑定；否则`PRACTICE_EVIDENCE_INCOMPLETE`。
- 所有claim复用现有`_validate_grounded_span_answer`的source、span、quote、identifier和支持性校验。
- 字段名和ordinal由服务器恢复并加入binding，模型不能控制最终renderer元数据。
- 通用`_extract_requested_practice_step_count`只识别明确的阿拉伯/常用中文数字加“步/steps”约束，范围1至12；不识别marker、课程名或特定验收句。冻结输入解析为6，`experiment_steps`必须恰为6。
- 未明确要求步骤数时不强制全局六步，但至少一个已绑定实验步骤；不得改变原始六部分实践框架语义。

#### 4.1.3 服务器renderer和Guard

orchestrator把每个字段渲染为固定section/ordinal下的grounded unit，服务器添加`[Sx]`与精确quote。模型字段中出现citation、quote分隔符或renderer控制文本即拒绝。

`CitationGuard.enforce`新增`verified_practice_bindings`，进入独立practice validator：

- 验证最终回答每个字段、每个步骤和每个正文单元；标题不计事实，但必须来自固定服务器模板。
- 检查unit数量、顺序、field、ordinal、claim、source、span、quote与verified binding完全一致。
- 拒绝额外/缺失/换序/重复正文、伪造source、未知/错span、source/span错配、quote篡改、未绑定事实和跨课程证据。
- 对每个unit继续调用现有selected-span支持性检查；不得降低任何阈值。
- 新practice分支不得改变现有standard和grounded renderer函数行为。

允许新增内部错误码：

- `PRACTICE_EVIDENCE_SCOPE_MISMATCH`
- `PRACTICE_EVIDENCE_INCOMPLETE`
- `PRACTICE_GROUNDING_RENDER_FORMAT`
- `PRACTICE_GROUNDING_BINDING_MISMATCH`
- `PRACTICE_GROUNDING_FAILED`

公共API仍返回现有字段和安全拒答文本。

#### 4.1.4 一次生成重试与终态

- primary候选出现结构化、binding、renderer或`UNSUPPORTED_CITATION`错误时，同一primary provider最多重做一次`GroundedPracticeGuide`生成。
- 重试复用同一run、step、question、hydrated evidence、spans和step-count约束。
- 重试前后Planner decision数、retrieval log数、tool call数、MCP调用数及question写入数不得增加。
- 第二候选失败立即ABSTAIN，不得第三次生成或Template接管。
- DeepSeek provider/transport错误使用现有provider层恢复；耗尽后ABSTAIN，不进入Template practice fallback。
- 显式Template primary必须生成严格extractive契约并通过同一Guard；响应真实标识`provider=template`，不伪装DeepSeek。

#### 4.1.5 脱敏Guard失败点

`CitationCheck`增加内部可选`failure_detail`，只包含第3.2E的白名单。orchestrator通过现有`_trace(..., event_type="citation_guard_failed")`写入一次；不改logger和数据库Schema。

### 4.2 仅修订验收口径

#### flow_06：`NO_CHANGE / AMEND_ORACLE`

- 产品代码、Planner提示、continue状态机均不修改。
- 正式测试继续验证确定性`REQUIRES_INPUT -> same run continue -> terminal`、跨用户GET/continue 403和终态不可重复continue。
- DeepSeek冻结输入仍是行为观察；Planner未选ask_user或补充后再次追问不单独构成产品缺陷。
- **8E/F16必须增加一条真实浏览器证据**：在隔离、受支持的确定性provider配置下，通过真实学生页面提交确定性追问输入，页面显示`REQUIRES_INPUT`，在同一页面提交continue，前后run_id一致并达到明确终态。保存追问前、追问后、continue后页面截图及run详情。该页面证据验证UI/HTTP/Agent串联，不声称DeepSeek自由决策稳定。
- 跨用户403仍由正式API测试证明，无需在浏览器切账号重复。

#### flow_08：`NOT_A_PRODUCT_DEFECT / AMEND_ORACLE`

只使用冻结资料和问题；验证上传、INDEXED、精确复述及正确document/chunk引用。不得修改普通course QA或Guard阈值。它同时作为practice共享orchestrator/Guard修改后的兼容性smoke。

#### flow_12：`NOT_A_PRODUCT_DEFECT / AMEND_ORACLE`

继续使用8A历史口径：同会话rerun保持账号/筛选，硬刷新可回登录门，同账号重登只见授权范围，退出深链不显示认证内容。不开发持久登录，不改JWT或前端存储。

### 4.3 最终允许修改的文件

产品文件，仅限：

- `backend/agent/orchestrator.py`
- `backend/llm/schemas.py`
- `backend/llm/template_provider.py`
- `backend/guards/citation_guard.py`

正式测试，仅限：

- 新建`tests/test_practice_grounding_contract.py`
- `tests/test_llm_provider.py`
- `tests/test_r3_grounded_answer_contract.py`
- `tests/test_api_endpoints.py`

验收/实施报告允许新增独立证据文件；不得为实现修改其他产品或测试文件。若8D发现确需增加文件，必须停止并回到用户/8C重新授权，不能自行扩范围。

明确禁止修改：

- `backend/rag/hybrid_retriever.py`
- `backend/storage/logger.py`和数据库迁移
- `backend/main.py`、公共请求/响应Schema
- `frontend/`、认证/JWT、`.env`、配置加载、业务超时
- Planner策略、continue状态机、MCP和真实工具实现
- 既有H5/F01-F15及`PASS_LOCKED_REUSED/PASS_SAVED`证据

### 4.4 8D测试与回归门

实施顺序：先加复现/负例测试，再写产品修复。

定向测试必须覆盖：

1. 冻结flow_04富证据资料：七个字段及六个步骤逐项绑定正确source/span并通过。
2. 冻结极简资料：ABSTAIN，Template不得补通用内容。
3. 错课程、错document/chunk、非INDEXED水合拒绝。
4. forged source、unknown/wrong span、source/span mismatch、quote tamper、duplicate、missing/extra/swapped binding、额外未绑定正文全部拒绝。
5. 第一次候选失败只重做生成一次，Planner/retrieval/MCP/tool/question计数不增加；第二次失败ABSTAIN。
6. DeepSeek失败不进入Template practice成功；显式Template primary使用严格契约并真实标识。
7. `failure_detail`无raw claim/quote/chunk/title/section/URL/密钥/token。
8. 普通course QA、上传索引、现有mixed-source grounded renderer和`TOOL_TIMEOUT`相关测试不回归。

定向集合通过后，必须在全新隔离SQLite/vector/cache/uploads环境运行一次完整正式pytest：

```text
python -m pytest -q -p no:cacheprovider
```

门禁：

- 0 failed、0 error。
- skipped/xfailed/warnings逐项单列，不能计入passed；不得新增未解释skip。
- 如果完整pytest暴露与本批改动相关的失败，8D必须修复或裁决`IMPLEMENTATION_NOT_READY`。
- 完整pytest不是重跑浏览器V6，不解锁H5/F01-F15。

### 4.5 8E最小浏览器补验

必须执行：

1. flow_04冻结正例：一次提交，七字段/六步骤、正确引用、真实provider/fallback状态。
2. flow_04冻结证据不足对照：安全拒答。
3. flow_06确定性真实页面：`REQUIRES_INPUT -> same run continue -> terminal`，保存页面与run证据。
4. flow_08冻结上传资料/问题：PENDING/INDEXED/精确引用。
5. flow_03一个已通过课程引用smoke，仅排除回归。
6. 一个伪造source或错span引用负例，必须拒答。
7. flow_12按修订口径补齐同会话、硬刷新、重登授权范围和退出深链尾段；不要求刷新持久登录。

不执行12项浏览器全量重跑，不重跑H5/F01-F15。

### 4.6 数据兼容与回滚

- 无数据库迁移、无公共API/Schema变化、无历史run回填。
- 回滚必须原子回退practice Schema、orchestrator水合/renderer/retry、practice Guard分支和对应测试；不得只回退Guard。
- 回滚后正式数据库无需处理，但flow_04缺陷会恢复，因此回滚不能作为最终完成状态。

## 5. 对既有PASS与安全语义的影响判定

- flow_03：practice专用Guard通过显式参数隔离；普通course QA走原standard路径。仍需一个smoke证明未误入practice分支。
- flow_08：普通course QA不改；冻结上传问答补验足以排除orchestrator共享代码回归。
- 权限：水合以当前course_id和本次已检索document/chunk双重约束，不新增查询范围；错课程负例必须加入。
- mixed-source Grounded Guard：旧分支不得修改；运行现有renderer负例。
- TOOL_TIMEOUT：不改MCP、超时或tool路径；重试计数测试必须证明未重复工具调用，完整pytest覆盖共享回归。
- continue：不改状态机；正式测试加真实页面证据后闭环。

## 6. 进入8D的裁决

`APPROVED_FOR_IMPLEMENTATION`

允许执行命令8D，但只能实施第4节的`reviewed_repair_plan`。8B中以下内容已被本评审替换，8D不得采用：

- 任何`proposed_extensions`或无证据proposal输出；
- DeepSeek失败后Template生成practice成功回答；
- 修改现有grounded renderer解析器来兼容practice；
- 把完整chunk写入state、日志、observation、API citation或测试快照；
- 只跑定向测试而跳过隔离完整pytest；
- 在最终F16省略flow_06真实页面continue证据。

本命令到此停止，不修改产品，不执行命令8D。
