# V6-FINAL-8 Practice混合证据契约补充评审

## 1. 裁决

`APPROVED_FOR_8D_S2`

本评审仅补充8C对Practice混合证据路径的范围解释和实施授权，不改变8D已完成的七字段契约、服务器renderer、独立Practice Guard、一次同provider生成重试、DeepSeek失败不由Template接管等既有裁决。

在8D-S2按本文件完成全部代码与测试门之前，状态继续保持：

```text
SCOPE_EXTENSION_REJECTED
IMPLEMENTATION_NOT_READY
command_8E_allowed=false
```

本次未修改产品代码、测试、配置、数据库或验收结果，未执行8D-S2、8E、浏览器验收或产品动作。8D既有137 passed定向证据继续有效，但不能替代8D-S2的新回归门。

## 2. 契约事实复核

### 2.1 原始需求明确要求Practice使用外部来源

`md/题目最终要求.md`的“核心模块：实践指导/智能检索”明确要求学生选择专题后可查找“相关论文、标准资料、漏洞信息”，并自动生成摘要和分类；同文件还要求系统与论文库、漏洞库等接口对接。因此，Practice不是course-only功能，可信外部证据属于原始需求范围。

### 2.2 Router明确支持Practice的显式外部required source

`backend/agent/router.py::requested_source_slots`在`practice_guide`意图下返回问题中显式命中的来源，否则才默认`["course"]`。当前来源词表包括：

- `paper`
- `standard`
- `vulnerability`
- `project`

所以Practice问题显式包含上述来源时，相应外部来源是既有、受支持的required source，不是测试专用魔法输入。

### 2.3 course+paper测试语义不是过时断言

`tests/test_agent_orchestration.py::test_practice_task_prioritizes_course_and_paper_evidence`要求：先有course检索证据、`paper_search`只执行一次、步骤顺序为`retrieve -> call_tool -> answer`、终态`COMPLETED`且最终引用覆盖两个真实来源。这与原始“实践指导+智能检索论文”的业务语义一致。

该测试当前还存在旧`PracticeGuide` provider和未落库course chunk两个夹具迁移问题，但8D-S1的只读证据同时证明，它在进入provider前因可信paper被course-only水合分支拒绝，stop reason为`PRACTICE_EVIDENCE_SCOPE_MISMATCH`。因此不能通过删除paper、把paper改成course或忽略required paper来调绿。

### 2.4 8C的“只水合course”不是“只允许course”

8C第4.1.1节约束的是本地course证据的完整chunk扩展：只有本次已检索、当前课程、已INDEXED且精确匹配document/chunk的内容可以在请求内临时水合。该约束防止第二次检索、越权和全文外溢。

已由现有Connector/MCP/工具可信链创建的外部`EvidenceItem`没有课程document/chunk，也不应访问课程数据库。它只能使用可信链已经产生的受控、净化、有限长度摘要。故正确解释为：

- course：严格授权后临时水合；
- 可信外部来源：不水合、不重新联网，只复用既有安全摘要；
- 未验证外部来源：fail closed。

## 3. 批准的混合证据契约

### 3.1 本地course evidence

保持8C和已实现逻辑不变：

1. 证据必须已经出现在本次`answer_evidence`中。
2. 必须属于`state.course_id`，document与chunk精确匹配且均为`INDEXED`。
3. 只在当前practice请求内水合完整chunk并经`SafetyGuard`净化、切分为有界span。
4. 完整chunk不得写回state、日志、observation、trace、异常、API citation或诊断快照。
5. 不得加入同文档其他chunk、同课程其他文档或任何未检索内容。

### 3.2 可信外部evidence

只允许`paper`、`standard`、`vulnerability`和`project`，并同时满足以下条件：

1. 证据已经由当前run的现有可信工具链创建；不得接受模型、请求调用者或测试直接自报的`trusted/source_id/URL/resource_type`。
2. `tool_name`必须存在于服务器`_TOOL_TO_SOURCE`映射，且映射结果与规范化后的`resource_type`一致。
3. 必须继续满足既有可信资源条件：server-set `trusted`、受信`origin_kind`、受信`delivery_mode`、HTTPS可定位URL、非空content hash以及title或external_id。
4. source_id必须能与当前run成功工具observation中由服务器记录的`evidence_ids`、对应tool_name和`provenance_verified=true`关联；仅字段外观相似不足以通过。
5. 只使用现有`EvidenceItem.text`，再次执行现有净化并保持既有长度边界；不读取外部全文。
6. 不访问课程数据库，不调用Connector/MCP，不重新联网，不扩大检索或工具范围。
7. 未知工具、类型不匹配、缺少当前run可信链关联、缺少provenance或不可定位的外部证据立即`PRACTICE_EVIDENCE_SCOPE_MISMATCH`并ABSTAIN。

### 3.3 混合span、七字段与来源覆盖

1. course和通过上述复核的外部证据分别生成服务器控制的`EvidenceSpan`；模型不能创建source_id或span_id。
2. `GroundedPracticeGuide`仍为七个字段，字段值仍为`list[GroundedSpanClaim]`，不增加proposal通道。
3. 允许的claim来源集合只包括`course`以及当前`state.required_sources`中显式要求且已验证的外部类型；拒绝未请求的额外外部类型。
4. 每个required source类型必须至少被一个有效claim覆盖；缺失时以既有grounding/source coverage语义拒答。
5. course可与required外部来源共同被引用；不得因存在外部required source而把合法course claim误判为unrequested source。
6. 七字段非空、显式步骤数、duplicate、伪造source、unknown/wrong span、source/span mismatch、quote篡改、identifier错绑、额外未绑定正文等校验全部保留。
7. 继续使用现有selected-span支持性检查和独立Practice Guard；不修改Citation Guard阈值或现有Grounded Guard分支。
8. 任一required source缺失或证据不足时继续ABSTAIN，不允许Template或无引用正文补齐。

实现应在`backend/agent/orchestrator.py`中采用practice专用允许来源与覆盖逻辑，避免改变普通`GroundedSpanAnswer`路径的既有语义。现有`CitationGuard.enforce(... verified_practice_bindings=...)`已能按最终answer中的引用类型执行required-source覆盖，因此本补充不授权修改Citation Guard。

## 4. 8D-S2精确允许范围

### 4.1 唯一允许修改的产品文件

- `backend/agent/orchestrator.py`

允许的实施内容仅为：

- 将practice证据准备分为“course授权水合”和“可信外部摘要验证/保留”两个分支；
- 为外部证据增加当前run可信工具链关联校验；
- 对两类证据构造请求内span；
- 为practice增加`course + required external`允许集合和required-source claim覆盖；
- 把`state.required_sources`传入practice专用校验，但不得改变普通grounded answer行为；
- 继续复用同一次检索和工具结果，不增加联网、工具调用或迟到写入。

### 4.2 唯一允许修改的测试文件

- `tests/test_practice_grounding_contract.py`
- `tests/test_agent_orchestration.py`
- `tests/test_phase3_e2e.py`

允许的测试迁移和新增仅为：

1. 增加course+paper混合正例，并断言七字段、六步骤、course与paper引用和required paper覆盖。
2. 增加forged/untrusted paper、工具类型不匹配、缺少当前run provenance关联、required paper缺失的拒答负例。
3. 证明外部证据不触发课程数据库水合、Connector/MCP二次调用或网络访问。
4. 迁移`SequenceProvider`的`GroundedPracticeGuide`分支，使其只从当前`LLMRequest.metadata.evidence_spans`选择合法source/span；其他Schema行为不变。
5. 为原编排测试创建与FakeRetriever一致、当前课程可授权且已INDEXED的持久化document/chunk；保留course+paper语义、一次paper工具调用、步骤顺序、COMPLETED和两个真实引用等核心断言。
6. 只扩充Phase3 `_seed_knowledge`为本地、隔离、防御性七字段六步骤资料；保留practice端点、assignment关联、citation存在及指向`5g_nas.md`等原断言。

### 4.3 禁止项与停止条件

明确禁止：

- 从测试删除paper、把paper改名为course、忽略required paper或将预期改成ABSTAIN；
- 相信模型或调用者自报的trusted、source_id、URL、resource_type或provenance；
- 对外部证据访问课程数据库、联网、抓取全文或再次调用Connector/MCP/工具；
- 修改Citation Guard、LLM Schema、router、Connector、MCP、公共API、数据库、前端、配置、超时或第四个测试文件；
- 降低七字段、步骤数、source/span、quote、support或Guard阈值；
- 回退8D既有修复，重跑H5/F01-F15或把既有PASS_SAVED重新验收。

若实现需要任何上述文件或行为，8D-S2必须立即输出`SCOPE_REVIEW_REQUIRED`并停止。

## 5. 8D-S2验证门

必须按以下顺序执行：

1. 两个原失败节点：
   - `tests/test_agent_orchestration.py::test_practice_task_prioritizes_course_and_paper_evidence`
   - `tests/test_phase3_e2e.py::test_phase3_full_demo_flow_offline`
2. 新增的course+paper正例、forged/untrusted/type mismatch/provenance缺失/required paper缺失负例，以及practice定向回归。
3. 使用全新隔离basetemp执行一次完整`pytest -q -p no:cacheprovider`。

放行条件：

- 所有上述节点和定向回归通过；
- 完整pytest为0 failed、0 errors；
- skipped、xfail和warnings逐项报告且不计作通过，不得新增未解释skip；
- 8D既有137 passed证据继续复用，不单独重跑；
- 正式数据库哈希、大小和业务计数不变；
- 不启动浏览器验收或8E。

只有全部通过后，8D-S2才可输出：

```text
IMPLEMENTATION_READY_FOR_TARGETED_ACCEPTANCE
command_8E_allowed=true
```

否则输出`IMPLEMENTATION_NOT_READY`；若超出本评审文件范围则输出`SCOPE_REVIEW_REQUIRED`。

## 6. 结论

原测试的course+paper语义符合原始需求，8D-S1识别出的冲突应通过窄范围产品澄清解决，不应通过削弱测试或降低Guard解决。批准进入8D-S2，但本命令到此停止，不自动实施。
