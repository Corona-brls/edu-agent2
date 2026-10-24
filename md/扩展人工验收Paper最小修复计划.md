# 扩展人工验收 Paper 最小修复计划

状态：`READY_FOR_REVERSE_REVIEW`  
下一门：`command_9I_allowed=true`

## 1. 依据、范围与裁决口径

本计划以9F唯一失败 run `096b7b4c-87c5-4d5d-89ff-77e6a8c6a6d3`、trace `d037fa2b-3af8-4f0c-9db3-191bede7f983`，以及9G的 `DIAGNOSIS_COMPLETE` 为依据。

当前只计一个 Paper 业务缺陷，其内部包含五个已证实机制：作者存储形状错配、长摘要下作者 span 不可达、作者未进入内容/hash、逐篇字段完整性缺失，以及查询与相关性未收敛。Citation Guard 行为正确；Multi-source 没有业务 run，只是依赖 Paper 阻断。

本计划不修改代码或测试，不运行 pytest、网络、模型、服务或浏览器。后续实现不得修改 Citation Guard、GroundedSpanAnswer/EvidenceSpan Schema、router/state、Connector/MCP、前端、权限、Practice、创新、teacher、配置、超时、工具预算或正式数据库 Schema，除非9I以新的不可替代证据批准范围增补。

## 2. 作者端到端契约

### 2.1 存储形状与旧数据兼容

新写入的 `authors_json` 必须是 JSON array，不再写成 `{"authors": [...]}`。作者使用一个共享的纯函数规范化：最多20人、每人最多160字符、只保留非空字符串、折叠空白并移除控制符/换行、保留 Unicode 名字、稳定去重且保持来源顺序。进入模型前仍经过现有不可信文本清洗。

`_external_resource_row` 同时兼容：

- 新格式 `list[str]`：返回有界 `list[str]`，标记当前 hash 覆盖作者。
- 旧格式 `{"authors": list}`：安全解包为 `list[str]`，但标记为 legacy/uncovered。
- 非法格式：返回空列表并 fail closed。

不做数据库迁移或批量重写。旧缓存仍可用于标题、摘要、链接等非作者事实；需要作者时，legacy/uncovered 记录不得生成作者 span，应返回 `PAPER_REQUIRED_FIELD_UNAVAILABLE`，等待正常 TTL 后的 verified live sync/upsert 写入新格式和新 hash。这样既兼容旧库，又不把未被原 provenance hash 覆盖的作者升级成可引用事实。

### 2.2 内容与 hash

`backend/resources/normalizer.py` 中的 `resource_content()` 和 `content_hash()` 使用同一份规范化作者列表。作者行进入索引文本，作者列表进入 hash payload；只改变作者时必须改变 hash。正常 verified sync 会按现有 upsert 更新旧行和 provenance，不新增表、列或迁移。

### 2.3 有界、服务器拥有的作者 span

本次不扩展 `EvidenceItem`、`EvidenceSpan` 或 `GroundedSpanAnswer` Schema。原因是现有 `EvidenceSpan.field=text` 足以承载可引用的作者原子事实，而 Schema 扩展会扩大所有 Grounded/Practice/API 路径的兼容面，9G没有证明其必要性。

对当前 hash 已覆盖作者的可信 paper：

1. `_resource_evidence_text()` 在摘要之前生成一个有界的服务器作者单元。
2. `_build_evidence_spans()` 或新的私有 bundle helper 为作者预留一个 `text` span，其余最多七个 text span 留给摘要/元数据，维持总上限8。
3. 同时生成只存在于本次请求 metadata 的 `span_semantics` 映射：作者为 `paper_authors`，摘要为 `paper_abstract`。
4. `paper_authors` 只能来自 trusted paper 的结构化 authors 字段、当前 hash 覆盖标志和服务器生成的第一个作者单元；摘要中伪造的 `Authors:` 文本、模型输出或调用者自报字段不能获得该语义。

模型仍必须选择合法 span 并产生抽取式原子 claim；服务器不生成、不替换、不补写作者 claim。

## 3. Paper 请求字段和数量契约

在 orchestrator 内增加私有、确定性的 `PaperRequestContract`，通用识别中英文别名：

- title：标题、title；
- authors：作者、author/authors；
- abstract clue：摘要、摘要线索、abstract、summary；
- link：链接、来源链接、URL、source link；
- 数量：阿拉伯数字或常用中文数字与“篇/paper/result”的组合。

不能匹配整条9F中文句、论文ID或某个固定主题。未明确要求的字段不强制，但所有选中来源始终需要可信 HTTPS provenance 和至少一个 grounded claim。

数量语义：

- 明确要求1至3篇：必须恰好满足。
- “最多N篇”：允许1至N篇。
- 未说明：最多3篇、至少1篇。
- 超过3篇或预计原子字段 claim 超过 `GroundedSpanAnswer` 的10条上限：生成前以 `PAPER_REQUEST_COUNT_UNSUPPORTED` 或 `PAPER_REQUEST_TOO_WIDE` 拒答，不在本次扩大 Schema。

对9F固定问题，每篇需要3个模型 claim：title、authors、abstract clue，共9条；link 由该 source 的 Citation DTO `source_url`/provenance HTTPS URL 满足，不要求模型把URL抄进正文。

### 3.1 生成前门

相关性过滤后，先检查每个候选的上游必需字段和可信链接。缺作者、缺摘要、URL不可信或legacy author hash未覆盖的候选不能用于满足对应字段。若剩余数量不足：

- 相关候选不足：`PAPER_RELEVANCE_INSUFFICIENT`；
- 有相关候选但必需字段缺失：`PAPER_REQUIRED_FIELD_UNAVAILABLE`。

两者均为 `ABSTAINED`，返回有界数量说明，不进入模型，不伪造字段或凑数。

### 3.2 候选完整性门

LLMRequest metadata 保存服务器计算的目标数量、请求字段、最终选中 source_id、各 source 的 title/author/abstract span IDs，以及 HTTPS citation URL 已验证标志。

`_validate_grounded_span_answer()` 增加 Paper 专用的逐 source 校验：

- title 请求必须有绑定该 source title span 的 claim；
- authors 请求必须有绑定该 source `paper_authors` span 的 claim；
- abstract clue 请求必须有绑定该 source `paper_abstract` span 的 claim；
- link 请求由同一 cited source 的可信 HTTPS Citation DTO 满足；
- 每个最终选中 paper 均须被引用，数量和去重身份与服务器选择一致。

模型遗漏字段时返回 `PAPER_FIELD_COVERAGE` 和脱敏的 `{source_id: [field enum...]}`；不得包含作者、摘要正文或候选原文。Citation Guard随后继续校验每个已有 claim 的 source/span 支持与相关性，职责不变且阈值不变。

## 4. 查询构造和跨语言相关性

### 4.1 可审计的主题查询

只对 `paper_search_real` 构建 Paper 查询：移除返回数量、字段格式、不要本地样例等输出指令，以及 Multi-source 中属于 NVD/GitHub 的从句；保留论文主题。优先保留引号内主题和连续 ASCII 技术短语，否则把有界、审计过的中英文通信/安全概念归一到 canonical English concepts，同时保留未知的重要主题词。

因此：

- 混合语言问题可直接抽取其中的英文技术主题；
- 全中文问题如“5G核心网攻击检测”可形成与英文论文元数据相同的 canonical profile/query；
- 不能只依赖中英文原始词面重合，也不能硬编码 `attack detection`、9F句子或arXiv ID。

若剥离后没有主题，返回 `PAPER_QUERY_EMPTY`，不盲目搜索。

### 4.2 单调用 over-fetch 与确定性收敛

仍只调用一次 `paper_search_real`，保持 `source=auto`、Semantic Scholar→arXiv fallback、live/fresh_cache 和 provenance 语义。工具参数使用：

`top_k = min(10, max(6, target_count * 3))`

不新增Connector/MCP调用，也不修改Connector代码、网络重试或超时。

服务端对返回项做确定性过滤、排序和去重：

1. 精确 DOI/arXiv/title locator 匹配优先；
2. canonical domain/entity 概念重合；
3. canonical security/task 概念重合；
4. 有界技术短语/token重合；
5. 现有 evidence score 只作末级 tie-break；
6. external_id 作稳定最终 tie-break。

若问题同时含 domain/entity 与 task/security 概念，候选必须每组至少命中一个；仅共同包含 `5G` 之类宽泛词不能通过。只有一组时必须命中该组。显式locator查询只能为精确匹配项绕过主题评分。

过滤后再按 canonical external identity、URL、content hash 去重并选取目标数量。被过滤论文不进入模型。相关论文不足时诚实拒答或按“最多N篇”语义返回可用数量，不得用S3式不相关论文补足。

## 5. 状态、重试与安全语义

错误必须可区分：

| 错误 | 含义 |
|---|---|
| `PAPER_REQUEST_COUNT_UNSUPPORTED` / `PAPER_REQUEST_TOO_WIDE` | 请求超出本次无Schema变化的有界能力 |
| `PAPER_QUERY_EMPTY` | 指令剥离后没有可审计主题 |
| `PAPER_RELEVANCE_INSUFFICIENT` | 一次工具结果中的相关可信论文不足 |
| `PAPER_REQUIRED_FIELD_UNAVAILABLE` | 相关论文缺少上游必需字段或作者hash覆盖 |
| `PAPER_FIELD_COVERAGE` | 模型候选遗漏选中论文的必需字段claim |
| `CONNECTOR_RATE_LIMIT` 等 | 保持现有第三方/transport分类 |
| 现有Guard错误 | 原样保留并fail closed |

`PAPER_FIELD_COVERAGE`、可重试的解析/Schema失败或现有 Grounded/Guard 失败最多使用一次现有 generation retry。retry必须：

- 复用完全相同的 selected evidence 与 span；
- 在实际 provider messages 中加入有界 source_id/field enums 或既有安全失败tuple；
- 不重复Planner、MCP、Connector或真实工具；
- 第二候选仍失败则按最终真实错误 `ABSTAINED`；
- DeepSeek失败不能由Template伪装成功。

## 6. 冻结文件白名单

产品文件：

1. `backend/storage/logger.py`
   - 仅允许修改作者 JSON 写入、旧格式读取、规范化和 transient hash-coverage标志。
2. `backend/resources/normalizer.py`
   - 仅允许增加纯作者规范化helper，并让 `resource_content`/`content_hash`覆盖相同作者值。
3. `backend/agent/orchestrator.py`
   - 仅允许 Paper 查询/over-fetch、作者span语义、相关性选择、生成前门、逐篇完整性校验、一次retry安全详情和上述错误语义。
   - 不得改变Practice和其他source语义。
4. `backend/llm/prompts.py`
   - 仅允许把服务器计算的 Paper 数量/字段、选中source、span semantics和retry指导送入真实 GroundedSpanAnswer messages。

测试文件：

1. `tests/test_j1_resource_provenance.py`
2. `tests/test_agent_orchestration.py`
3. `tests/test_r3_grounded_answer_contract.py`

明确禁止修改：Citation Guard、LLM schemas、router/state、resource service、arXiv/Semantic Scholar Connector、MCP、frontend、数据库迁移、配置、`.env`、超时、网络重试和工具预算。若9J发现需要白名单外文件，必须输出 `SCOPE_REVIEW_REQUIRED`，由9I重新批准。

## 7. 自动化测试矩阵

9J所有自动化测试只能使用离线受控fixture；真实固定问题留给9K，不得在 pytest 中调用 live Connector、DeepSeek、MCP进程或浏览器。

先红后绿至少覆盖：

- 两作者经verified Connector→SQLite→service/MCP形状→Agent仍为有界list并可引用；
- 旧 `{"authors": [...]}` 行可读，非作者路径兼容，作者请求在未刷新时fail closed；
- 作者变化改变内容和hash；
- 超过8句摘要仍有typed author span；恶意、重复、控制字符、超长作者被限制且不能伪造语义；
- 通用中英文请求字段/数量解析，不依赖完整9F句子；
- 全中文topic生成canonical English查询并命中英文fixture；
- 单次 `paper_search_real` 有界over-fetch，保持 `source=auto`；
- 三篇相关论文的title/author/abstract逐篇完整，link由正确Citation URL满足；
- S3式NOMA宽泛候选被过滤，不能靠共同`5G`通过；
- 相关结果不足、字段不足、重复paper、无摘要、URL不可信分别安全失败；
- 第一个候选漏一个author/abstract时仅重试生成一次并复用证据；第二次仍漏则拒答；
- 错source/span、伪造作者、prompt injection仍拒绝；
- 生成retry期间Planner/MCP/Connector/tool次数不增加。

修改文件定向测试通过后，再运行现有但不修改的兼容回归：

- `tests/test_j3_agent_evidence.py`
- `tests/test_l1_agent_grounding.py`
- `tests/test_s2_multi_source_citations.py`
- `tests/test_citation_guard.py`
- `tests/test_practice_grounding_contract.py`
- `tests/test_phase3_e2e.py`
- `tests/test_api_endpoints.py`
- `tests/connectors/test_connector_framework.py`

最后仅在稳定代码上使用全新隔离basetemp运行一次完整pytest，要求0 failed、0 errors；skipped、xfail和warnings单列，skip不算通过。

## 8. 实施顺序、兼容和回滚

9J顺序：保护工作区与不变量→加入有效红测→修作者存储/hash→修查询与相关性门→修span semantics与完整性/retry→定向测试→兼容测试→完整pytest→不变量报告。

不做数据库迁移和缓存批量删除。新 array 行可被旧 reader 的 `json.loads` 读取；旧 wrapper 行由新 reader兼容。回滚只恢复4个产品文件和3个测试文件，正常TTL/upsert继续管理缓存。

出现以下任一情况必须停止：需要第5个产品文件或第4个测试文件、需要改Schema/Guard/Connector/MCP/router/state/Practice/RBAC、需要写正式数据库，或双语相关性无法在受控正负fixture上稳定区分而又必须降低安全门。

## 9. 9I反向评审重点

9I必须挑战：

1. `paper_authors`语义是否只能由服务器结构化作者生成，摘要伪造是否完全无效。
2. legacy hash-coverage策略是否既兼容旧缓存又不会引用未纳入hash的作者。
3. 双语canonical profile是否支持中文问题检索英文论文，且仅共享`5G`不能通过。
4. 数量/字段contract是否严格受10条claim上限约束，没有暗中扩大Schema。
5. link是否由同一 cited source 的HTTPS Citation DTO满足，且不强迫URL正文。
6. over-fetch是否仍是一次工具调用，未改变fallback/cache/provenance。
7. 完整性retry是否真实进入provider messages、只含脱敏字段枚举且不重复上游调用。
8. Multi-source是否仍仅是兼容/9K依赖，而没有被倒推为已确认缺陷。

9I只有输出 `APPROVED_FOR_IMPLEMENTATION` 和 `command_9J_allowed=true` 才能实施。

## 10. 9J与9K门

9J全部代码回归通过时应输出：

```text
IMPLEMENTATION_READY_FOR_PAPER_ACCEPTANCE
command_9K_allowed=true
```

9K只执行最小真实页面矩阵：

1. 新隔离环境中通过真实学生页面一次提交固定Paper问题：
   `请使用真实论文搜索检索 5G core network attack detection，返回 3 篇论文的标题、作者、摘要线索和可核验来源链接；不要用本地论文样例代替真实联网结果。`
2. 只有Paper严格PASS后，再一次提交固定Multi-source问题：
   `请检索并综合真实的 5G 安全论文、NVD 中的 CVE-2021-44228 漏洞记录，以及 GitHub 上的 Open5GS 开源项目。请基于各自来源给出可核验的原子事实、逐项引用和实验建议。`

Paper PASS必须有一次真实工具调用、3篇互不重复且相关的论文、逐篇title/author/abstract grounded claims、同一来源的HTTPS Citation URL、DeepSeek且无fallback、Guard通过、无重复上游调用。若429或上游不可用，判第三方限制；若Paper语义失败，只诊断唯一Paper run，不执行Multi-source。Paper PASS后若Multi-source失败，永久保留Paper PASS并只诊断唯一Multi-source run。

两项通过后与9E、9D-S1和8V已有证据合并，才输出 `POST_V6_MANUAL_ACCEPTANCE_PASS`。

证据只保存简单JSON、测试统计、run_id/trace_id和必要截图，不设置ZIP成员数、集合指纹、目录总hash、固定文件数或全局mtime门。

9G两个只含合成数据、被host ACL锁定的临时目录是非阻断环境例外；后续不再修改其ACL或所有权，临时资源统一放到项目 `tmp` 下。
