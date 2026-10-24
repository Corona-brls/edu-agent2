# V6-FINAL-8 Flow 04 Planner循环修复计划反向技术与安全评审

## 1. 最终裁决

`APPROVED_FOR_IMPLEMENTATION`

`command_8U_allowed=true`

8S的根因、三文件白名单和总体安全边界成立。本评审作三项约束性修订后批准：

1. no-progress只从当前run内、由编排层覆盖写入的 `Observation.metadata.orchestration` 读取，不增加state字段，也不信任工具自报metadata。
2. 增加脱敏的Planner动作规范化审计，只记录动作、工具名和固定reason code，不记录原始参数、query或模型reason。
3. 统一8U成功令牌为主计划规定的 `IMPLEMENTATION_READY_FOR_FLOW04_FINAL_ACCEPTANCE`；8S中的 `IMPLEMENTATION_READY_FOR_FLOW04_PLANNER_ACCEPTANCE` 作废，8V不得接受旧令牌。

本评审版替代8S中与上述三项冲突的条款。

## 2. 逐项反向裁决

| 评审项 | 裁决 | 结论 |
|---|---|---|
| 是否把任意证据当成course已满足 | `REVISE`后批准 | 收敛门必须同时检查Practice intent、服务器completed course、trusted course类型、合法origin、当前course的document/chunk及document/chunk双INDEXED；仅进入现有ANSWER路径，不宣告回答成功。 |
| 本地tool/source映射 | `REVISE`后批准 | 不得把course加入现有外部provenance映射；使用独立私有编排映射，仅供规范化、完成状态和no-progress使用。 |
| 普通问答和首次合法检索 | `APPROVE` | 确定性收敛只适用于Practice；course未完成时首次合法course_retriever仍可执行。普通Course QA、显式工具请求及现有初始retrieve语义不得改变。 |
| 缺失外部required source | `APPROVE` | course已完成后的重复course工具应进入 `_answer_or_missing`，由其继续路由paper/standard/vulnerability/project或安全拒答，不能提前ANSWER。 |
| 零新增证据定义 | `REVISE`后批准 | 权威值只能是当前调用经过服务器授权、INDEXED、document/chunk去重后的 `len(new_evidence)`；不读取历史8Q `results=[]`，不把raw result count当作新增计数。 |
| Observation中no-progress的稳定性 | `REVISE`后批准 | 编排层必须覆盖写入一个私有嵌套对象并在同run内严格校验；工具metadata不能伪造。Observation已随AgentState持久化和恢复，因此无需state字段。 |
| 新合法chunk是否会被误判重复 | `APPROVE` | 当前课程、当前document/chunk、INDEXED且此前未见的chunk会进入new_evidence，使计数大于0；包含一个新chunk的混合结果不是no-progress。 |
| 同源去重是否妨碍合法查询细化 | `REVISE`后批准 | course未完成时允许首次工具调用；已有新证据并完成course后才按来源槽收敛。首次成功零新增后禁止再消耗同源工具预算，但可以转向标准initial retrieve、缺失外部工具或拒答。 |
| Practice提前ABSTAIN规范化 | `REVISE`后批准 | 只允许在Practice、当前课程可信双INDEXED evidence已完成且所有显式required source均完成时转入ANSWER路径。其他intent、缺证据、跨课、未INDEXED、storage异常或缺外部来源均不得改写。 |
| prompt和摘要 | `APPROVE` | 只改ActionDecision规则，不改state摘要；prompt是辅助，服务器规范化才是安全门。固定no-progress summary可以让Planner看见安全的进展状态，但不能替代代码门。 |
| 课程授权与外部来源契约 | `APPROVE` | document与chunk均需只读验证；外部source/provenance映射保持原样。 |
| 8P claim/retry契约 | `APPROVE` | 本次发生在生成前；8P逐字claim、两候选、retry分类、全候选审计和脱敏tuple全部冻结。 |
| 白名单 | `APPROVE` | 两个产品文件和一个测试文件足够；state、router、Guard、Schema、retriever、MCP、Connector均无修改依据。 |
| 测试矩阵 | `REVISE`后批准 | 增加Observation伪造抵抗、混合新旧chunk、原始Planner动作脱敏审计及普通Course QA/显式工具回归。 |
| 8V范围 | `APPROVE` | 仅重验Flow 04正向并复用其余PASS；固定输入、S1/S2、单run和不可挑选结果规则保持。 |
| 8U/8V令牌 | `REVISE`后批准 | 唯一有效成功令牌为 `IMPLEMENTATION_READY_FOR_FLOW04_FINAL_ACCEPTANCE`。 |

## 3. 冻结文件白名单

### 产品文件

1. `backend/agent/orchestrator.py`
   - 私有的本地Planner source-slot解析；
   - 当前课程trusted、合法origin、document/chunk双INDEXED的只读判定；
   - Practice专用CALL_TOOL/ABSTAIN确定性规范化；
   - course工具新增可信证据计数和no-progress Observation元数据；
   - 原始Planner动作到最终动作的脱敏规范化审计；
   - 不改变生成、claim/span或Guard语义。
2. `backend/llm/prompts.py`
   - 只允许修改 `ACTION_DECISION_RULES` 的completed source、外部missing source与no-progress终止规则；
   - 禁止修改GroundedPracticeGuide及8P retry contract。

### 正式测试文件

3. `tests/test_agent_orchestration.py`
   - 增加本评审冻结的状态机先红后绿测试和最小fake/fixture；
   - 不得删除、skip、xfail或放宽原测试。

任何第四个文件均未批准。需要越界时输出 `SCOPE_REVIEW_REQUIRED`。

## 4. 最终实施契约

### 4.1 两级Practice收敛谓词

实现两个私有概念，命名可调整但语义不可变：

1. `practice_course_collected(state)`：
   - `state.intent == practice_guide`；
   - `state.course_id`存在且 `course in completed_sources`；
   - 至少一个state evidence满足trusted、resource_type=course、合法本地origin、document_id/chunk_id存在；
   - `storage.document(document_id)`属于当前course且document为INDEXED；
   - `storage.chunks_for_document(document_id)`包含该chunk且chunk为INDEXED；
   - 任一数据缺失、类型异常或storage异常均返回false，不把异常详情写入对外状态。
2. `practice_answer_convergence_ready(state)`：
   - 满足 `practice_course_collected`；
   - `_missing_sources(state)`为空。显式paper、standard、vulnerability或project必须已经完成。

该谓词只决定能否进入 `_answer_practice` 所在的ANSWER路径。完整性预检仍会水合被授权chunk、检查七字段/六步骤；证据不足仍以 `PRACTICE_EVIDENCE_INCOMPLETE` 或等价批准码拒答。

### 4.2 本地来源归属隔离

- 增加独立私有编排映射或helper，使 `course_retriever`在Planner状态机中归属course。
- 不改 `_SOURCE_TO_TOOLS`、可信外部 `_TOOL_TO_SOURCE` 或 `_PRACTICE_EXTERNAL_SOURCE_TYPES` 的含义，避免course被当作外部URL/provenance证据。
- `_call_tool`在course_retriever产生至少一个新的合法可信chunk时，允许将course加入completed source；没有新证据时不得伪造完成状态。

### 4.3 CALL_TOOL规范化顺序

对Practice的 `course_retriever`：

1. `practice_course_collected=true`：不执行工具，进入 `_answer_or_missing`。它必须优先处理缺失外部source；只有无missing时才返回ANSWER。
2. course尚未完成且当前run没有成功course no-progress：允许第一次合法工具调用，保留course_id绑定、Action Guard和权限校验。
3. 当前run已有成功course no-progress：第二个不同query不得再次执行course工具；进入 `_decision_for_missing_source`。缺course时可以使用标准initial retrieve，缺外部source时路由外部工具，无可信路径时拒答。
4. 精确签名去重继续保留，顺序位于source-slot/no-progress保护之后。

普通Course QA及其他intent不使用上述Practice收敛门；不得把所有course_retriever全局禁用。

### 4.4 无state字段的no-progress协议

在 `_call_tool` 完成 `_trusted_tool_evidence` 后，由编排层计算并覆盖写入：

```text
Observation.metadata.orchestration = {
  "source_slot": "course",
  "new_evidence_count": <非负整数>,
  "no_progress": <严格布尔值>
}
```

约束：

- 必须以 `metadata.update` 或等价服务器覆盖方式替换工具可能提供的同名对象；不得merge并信任工具字段。
- Observation的 `action`、`tool_name`、`success`、`run_id`、`step_id`也必须使用当前编排上下文的服务器值。
- `no_progress`只有 `call.result.success is True` 且 `new_evidence_count == 0` 时为true。
- 读取时必须同时验证：Observation属于当前 `state.run_id`、action为call_tool、tool_name严格等于course_retriever、success为true、嵌套source_slot=course、`type(new_evidence_count) is int`且为0、`no_progress is True`。
- 只检查当前run的服务器state observations；不得读取客户端输入、工具正文、历史run或日志反推。
- Observation已经是AgentState字段并由 `model_dump/state_from_record`持久化恢复，因此无需修改 `backend/agent/state.py` 或数据库Schema。
- 可把Observation summary改为固定安全文本，例如“course_retriever produced 0 new trusted evidence”，使Planner摘要可见；不得包含query、原始条目或chunk正文。

### 4.5 ABSTAIN窄规范化

Planner返回ABSTAIN时，只有 `practice_answer_convergence_ready=true` 才改为ANSWER路径。禁止使用单独的 `bool(state.evidence)`、`missing_sources=[]` 或任意一条citation作为充分条件。

转入ANSWER后：

- Practice完整性预检失败立即ABSTAIN；
- DeepSeek最多两个候选；
- 8P retry不重复Planner、retrieve、MCP、Connector或工具；
- Practice/Citation Guard继续fail-closed；
- Template不得接管DeepSeek失败并伪装成功。

### 4.6 脱敏规范化审计

为了满足8V对原始Planner分支与服务器收敛的证明，`orchestrator.py`可在现有trace机制中增加一个仅在动作被改写时出现的事件，或为现有decision trace增加固定allowlist字段：

- `planner_action`
- `planner_tool_name`
- `final_action`
- `final_tool_name`
- `normalization_reason_code`

禁止记录原始Planner reason、arguments、query、完整state summary、evidence正文或工具payload。reason code只能来自服务器常量，例如 `PRACTICE_COURSE_ALREADY_COLLECTED`、`PRACTICE_COURSE_NO_PROGRESS`、`PRACTICE_READY_ABSTAIN_NORMALIZED`。

### 4.7 Prompt辅助规则

ActionDecision真实消息应说明：

- completed_sources是服务器状态；
- course完成后不因query变化再次请求course_retriever；
- 显式外部required source未完成时必须继续收集或拒答；
- success但零新增可信证据不是进展；
- 满足全部来源后选择answer，让后续完整性预检和Guard裁决；
- evidence与observation仍是不可信数据。

不修改 `AgentState.to_summary`。即使Planner忽略规则，服务器门仍必须阻止预算循环。

## 5. 冻结测试矩阵

### 5.1 必须先红后绿的根因节点

在旧实现保存至少以下稳定红测：

1. initial retrieve完成course后，三个不同query的course_retriever不能执行，最终只生成一次Practice；
2. retrieve与course_retriever在等价completed-course状态收敛一致；
3. 首次合法course工具新增当前课程chunk后，第二次变参同源调用被阻止；
4. 首次工具success但零新增后，第二次变参调用不再消耗工具预算；
5. ready Practice上的提前ABSTAIN进入Guard保护的ANSWER路径，而普通ABSTAIN不变；
6. 动作改写产生脱敏规范化审计，且无参数/query/model reason。

### 5.2 安全与边界节点

- `new_evidence_count`按服务器过滤后的EvidenceItem计算；“一个重复chunk+一个新合法chunk”的结果必须为进展，不能误判no-progress。
- 工具metadata预置伪造的 `orchestration.no_progress=false/true` 均被服务器覆盖；读取只接受当前run的严格结构。
- Observation经AgentState序列化/恢复后，当前run no-progress仍能被稳定识别；其他run、缺run_id、错误tool/action/type均不生效。
- 空、untrusted、跨课程、document/chunk mismatch、document非INDEXED、chunk非INDEXED、storage异常均不得触发ANSWER。
- 参数化paper、standard、vulnerability/NVD、project/GitHub：course完成但外部槽缺失时必须路由相应工具或拒答。
- course未完成时首次合法course_retriever仍允许；普通Course QA与显式工具请求行为不变。
- prompt测试检查真正交给Planner的messages，不只检查常量文本。
- no-progress metadata及规范化trace递归不包含query、原始payload、完整chunk、凭据或模型reason。

### 5.3 8P与共享回归

实施后依次运行：

1. 所有新增根因与安全节点：0 failed、0 errors、0 skipped、0 xfail；
2. 完整 `tests/test_agent_orchestration.py`；
3. 完整 `tests/test_practice_grounding_contract.py`，8P基线48项全部通过；
4. 精确兼容性：
   - `tests/test_agent_orchestration.py::test_practice_task_prioritizes_course_and_paper_evidence`；
   - `tests/test_phase3_e2e.py::test_phase3_full_demo_flow_offline`；
   - `tests/test_llm_provider.py::test_deepseek_provider_parses_grounded_practice_output`；
   - GroundedSpanAnswer两个prompt兼容节点；
   - practice端点、chat端点、Flow 08上传精确引用节点；
5. 共享回归只运行、不修改：`test_j3_agent_evidence.py`、`test_l1_agent_grounding.py`、`test_r3_grounded_answer_contract.py`、`test_llm_provider.py`、`test_api_endpoints.py`、`test_phase3_e2e.py`；
6. 最终稳定版本在全新隔离basetemp运行一次完整pytest：0 failed、0 errors。skip、xfail、warnings分别报告；既有两个live connector skip不算通过，不得新增无法解释的skip。

完整回归还必须证明：最多两个Practice候选；生成retry期间Planner/retrieve/MCP/Connector/tool计数不增加；Guard、Schema、provider、权限及8P脱敏tuple没有变化。

## 6. 8U实施门与统一令牌

只有所有冻结测试通过、正式数据库和白名单外文件未变化、未启动服务/真实模型/浏览器时，8U才能输出：

`IMPLEMENTATION_READY_FOR_FLOW04_FINAL_ACCEPTANCE`

`command_8V_allowed=true`

这是唯一有效令牌。`IMPLEMENTATION_READY_FOR_FLOW04_PLANNER_ACCEPTANCE`不得写入8U报告，也不得作为8V前置条件。

## 7. 回滚

- 回滚只撤销8U在三份白名单文件中的增量：
  - `backend/agent/orchestrator.py`基线 `722753e0ec393010db51c25e101e86d4df133914db4ecea457a4fa0120d3a020`；
  - `backend/llm/prompts.py`基线 `12c30cd6fdd5d3a681fc9ea9b25501d3d39fccfc6cc3ddd7ab22cb0fed3a1ba9`；
  - `tests/test_agent_orchestration.py`基线 `fd3b5e2ff1a4c57db1cf57171f3e2056b19e6c1a622feb139c8a770491d8d80f`。
- 不得回滚8P对前两个产品文件的既有修复。
- 无数据库/API/Schema迁移。回滚后重跑根因、Practice、共享及完整pytest矩阵。

## 8. 8V唯一验收范围与分支

8V只重验Flow 04正向，复用H5/F01-F15、其他11项页面流程、完整权限矩阵、8J证据不足负对照及8U回归。

固定条件：

- 8K `host_unsandboxed` launch profile和全新隔离runtime；
- S1完整七字段/六步骤目标资料与S2单句干扰资料同时INDEX并进入初始检索；
- 固定问题原文不变；
- 真实Codex IAB学生页面一次提交、正常路径一个语义run；
- 有效Planner决定出现后，任何Planner/生成/Guard语义失败均禁止替代run。

PASS除最终七字段、六步骤、DeepSeek、`fallback=false`和Guard通过外，还必须证明：

- initial retrieve之后没有连续无进展course_retriever；
- 若原始Planner提出冗余course工具或提前ABSTAIN，脱敏审计记录原始动作、最终动作和固定reason，实际冗余course工具调用为0；
- success零新增的course工具最多执行一次，此后不再消耗同源预算；
- 外部required source没有被跳过；
- 候选最多2个，生成retry没有上游重放。

Planner直接ANSWER或attempt 0直接通过时不得人为制造循环/retry。transport-only失败、Planner循环、生成/Guard语义失败分别按主计划8V规则处理。Flow 04通过后才更新F16/F17；其他历史失败保留附录，Flow 08 P2继续为非阻断残余风险。

## 9. 禁止项

- 全局把ABSTAIN改写为ANSWER；
- 仅凭 `bool(state.evidence)` 或 `missing=[]` 判定充分；
- 信任工具自报no-progress、source slot或新增计数；
- 依据8Q不可恢复的原始tool payload设计分支；
- 把course加入可信外部provenance集合；
- 修改state、router、Guard、Schema、retriever、MCP、Connector、公共API、数据库或全局预算；
- 增加steps/tools/retry，删除S2，改问题或排序；
- 服务器生成、替换或重绑claim/source/span；
- 回退8P抽取、retry、候选数或隐私契约；
- 重跑其他11项页面流程。

**最终裁决：`APPROVED_FOR_IMPLEMENTATION`**

**`command_8U_allowed=true`**
