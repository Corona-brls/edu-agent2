# 扩展人工验收 Paper 修复计划反向评审

总体裁决：`APPROVED_FOR_IMPLEMENTATION`  
放行：`command_9J_allowed=true`

本评审以9F唯一 Paper run、9G只读诊断和9H计划为依据。9H不能原样批准：其作者 claim 可能超过 `GroundedSpanClaim.claim_text=360`，作者 hash 覆盖判据不充分，三篇被错误设计为硬成功门，双语相关性规则与 `source=auto` 回退边界不够冻结，9J令牌和范围停止条件也有文档错误。下述替换条款完成这些修订后，计划可实施。

## 1. 独立复核结论

- 当前只计一个 Paper 业务缺陷，作者传递、hash、字段完整性、查询构造和相关性收敛是同一缺陷的关联机制，不拆成多个业务缺陷。
- Multi-source 尚无业务 run，只是依赖 Paper；9J只做兼容回归，9K在 Paper 通过后才做一次真实续验。
- Citation Guard 对既有原子 claim 的支持性与相关性判断正确。不得修改或放宽 Guard。
- 现有4个产品文件和3个测试文件可以形成最小闭环；无证据支持扩大到 Schema、Connector、MCP、数据库结构或前端。

## 2. 逐项裁决

| 计划项 | 对9H原案 | 修订后 | 主要理由 |
|---|---|---|---|
| 作者存储与旧数据兼容 | REVISE | APPROVE | list形状本身不能证明当前content hash覆盖作者，必须服务器重算并比对。 |
| 作者可引用span | REVISE | APPROVE | 完整作者列表可能超过360字符，必须生成可逐字引用的有界作者信息摘要。 |
| 逐篇字段完整性 | REVISE | APPROVE | 只校验最终选中论文；链接必须核对同一Citation DTO的真实HTTPS URL。 |
| 数量与不足语义 | REVISE | APPROVE | 实时结果允许1–3篇；不足必须显式、稳定区分，不能用不相关论文凑数。 |
| 双语查询与相关性 | REVISE | APPROVE | 必须冻结有界词典、短语归一化、停用词和最低命中规则，并覆盖多组离线fixture。 |
| `source=auto`与调用次数 | REVISE | APPROVE | Semantic Scholar成功但结果不相关时不得隐式再搜arXiv；9K只约束一次Agent工具和一次MCP请求。 |
| 同证据生成retry | APPROVE | APPROVE | 最多一次、复用同一选中证据且不重做上游；第二次失败仍拒答。 |
| 失败可观测性 | REVISE | APPROVE | 需稳定区分零相关、数量不足和字段缺失，不能留给下一轮Planner猜测。 |
| Multi-source兼容与9K问题 | REVISE | APPROVE | 每类只要求一条单span逐字支持事实，不要求无证据跨来源推理或实验建议。 |
| 文件白名单 | APPROVE | APPROVE | 4个产品文件＋3个测试文件足够；Guard、Schema、Connector、MCP均禁止修改。 |
| 9J/9K门禁文本 | REVISE | APPROVE | 成功令牌与范围停止条件按本评审统一。 |

## 3. 冻结的最终工程契约

### 3.1 作者存储、hash与旧数据

`authors_json`新写入为有界JSON数组。规范化最多20名作者、每名最多160字符，移除控制字符、折叠空白、保留Unicode、稳定去重并保持源顺序。旧 `{"authors": [...]}` 可安全读取；非法形状返回空列表并fail closed。

`authors_hash_covered=true` 只能由服务器按当前版本重新计算后得出，同时满足：

1. 回读作者是有效、非空的有界列表；
2. 用当前 `resource_content()`/`content_hash()` 规则从该行可信字段重新构造的hash，等于该行保存的 `content_hash`；
3. 现有provenance验证仍通过。

仅看到list形状不得设置该标志。旧wrapper、畸形JSON或hash不一致的行仍可供应被现有hash覆盖的标题、摘要和链接，但不能供应作者claim；正常verified sync/upsert可自然升级，不做数据库迁移或批量重写。

`resource_content()`和`content_hash()`必须包含同一份完整的“有界规范作者列表”，从而作者变化改变当前hash。这里“完整”指规范化上限内的列表，不是无限接受上游元数据。

### 3.2 有界作者信息摘要

不修改任何Schema。对当前hash验证通过的paper，服务器从结构化作者列表确定性生成一个 `paper_author_summary` text span：

- `exact_text`总长最多300字符，为 `claim_text=360` 留出安全余量；
- 只包含完整作者名，绝不在姓名中间截断；
- 全部可容纳时使用“Author information: …”；
- 不能全部容纳时使用“Author information (excerpt, showing k of n): …”，明确这是节选；
- 至少一名作者也无法安全容纳时，该论文作者字段不可用；不得切片、补写或改名。

作者span在最多8个text span中固定保留1个，其余最多7个给摘要/文本。语义映射由服务器根据可信结构化字段生成；摘要中出现 `Authors:`、模型自报或调用者自报不能获得 `paper_author_summary` 语义。

候选的作者claim必须逐字等于该作者摘要span的 `exact_text`，而非自由改写。模型仍负责选择合法source/span；服务器不得创建、替换或修补模型claim，也不得静默换span。节选span只能支持“作者信息节选”，不能支持“完整作者列表”。

### 3.3 Paper请求字段、最终选择集合和链接

Paper请求契约通用识别title、authors、abstract clue、link以及1–3的目标数量，不匹配整条9F句子、特定论文ID或固定主题。字段完整性只针对经过服务器相关性过滤、去重并最终选中的 `selected_paper_source_ids`：

- title绑定该论文title span；
- authors绑定该论文 `paper_author_summary` span；
- abstract clue绑定该论文abstract text span；
- link不要求模型把URL写进正文，但必须验证同一论文Citation DTO的实际 `source_url`：使用URL解析器确认scheme为HTTPS且有host，并核对它与所选EvidenceItem的provenance、external_id和content_hash身份一致。

不得用 `has_https_link=true` 之类布尔标志代替实际URL校验。过滤掉或未选中的paper不得进入最终LLM evidence catalog、verified bindings、renderer或citation cards，也不能被模型带回答案。

若模型遗漏某一选中paper的必需字段，候选以 `PAPER_FIELD_COVERAGE` 拒绝；最多一次同证据生成retry。服务器不得补写作者、摘要、标题或URL。

### 3.4 1–3篇与稳定终态

固定Paper问题中的“返回3篇”解释为目标数量和最大数量，不是必须凑足三篇的成功门。服务器最多选择3篇；至少有1篇真实、相关、字段完整的paper即可生成并完成：

- 3篇：`COMPLETED / ANSWERED`；
- 1–2篇：`COMPLETED`，并使用稳定的 `PAPER_PARTIAL_RELEVANCE_RESULTS` 或 `PAPER_PARTIAL_FIELD_RESULTS` 终态原因，保存requested/relevant/field-complete/selected计数；不得声称返回了3篇；
- 0篇相关：`ABSTAINED / PAPER_NO_RELEVANT_RESULTS`，不调用模型；
- 有相关paper但0篇满足必需上游字段：`ABSTAINED / PAPER_REQUIRED_FIELD_UNAVAILABLE`，不调用模型。

若同时存在相关性与字段短缺，服务器按上述计数记录两者，终态理由以直接阻止选择的阶段为准。所有分类在Paper预检/回答分支直接完成，不回到Planner猜测或发起第二次工具调用。数量不足不是伪造理由，也不能用NOMA等不相关结果补位。

### 3.5 查询、双语相关性和`source=auto`

只在Paper参数构造中剥离“返回几篇、包含哪些字段、不要使用本地样例”等输出指令和非Paper来源子句，保留显式技术短语、DOI、arXiv ID、CVE等定位符。最长短语优先归一化；未知显著主题词保留，禁止把任何问题替换成9F主题。

在orchestrator中冻结有界、可审计的网络安全/移动通信双语概念表、输出指令停用短语与停用词。至少包含独立的领域/实体组（如5G、core network/核心网、NAS/非接入层、network slicing/网络切片、RAN、SDN/NFV）和安全/任务组（如attack/攻击、detection/检测、authentication/认证、replay/重放、vulnerability/漏洞、intrusion/anomaly/入侵/异常、privacy/隐私），并使用稳定常量和版本标识。不得从LLM动态生成词典。

相关性规则：

1. 精确DOI/arXiv ID/精确标题定位只对匹配身份放行；
2. 请求同时含领域/实体和安全/任务概念时，候选必须两组各命中至少一个canonical概念；
3. 仅一组有意义概念时必须命中该组；
4. 仅共享 `5G` 或普通停用词不能通过；
5. 受控技术短语/token重合作为补充，connector score只作最终排序tie-break；
6. 稳定external_id作最后tie-break，随后去重并最多选3篇。

离线测试至少使用四组不同中文主题匹配英文论文（5G核心网攻击检测、NAS重放检测、网络切片DDoS缓解、5G认证异常检测），并包含NOMA/频谱效率等干扰项。规则必须支持中文问题→英文论文，而不是依赖同语言词面重合。

`source=auto`保持一次Agent层 `paper_search_real` 工具调用和一次MCP请求。MCP内部只有当Semantic Scholar发生连接失败、限流、transport或既有明确失败时才能按既有顺序回退arXiv；Semantic Scholar成功返回后，即便结果经服务器相关性门全部过滤，也不得在Agent层或验收层再发第二次工具调用去arXiv。9J不修改resource service、Connector或MCP。

### 3.6 Retry、Guard与隐私

相关性和上游字段预检在生成前完成。模型字段遗漏、结构化解析失败或现有Grounded/Guard失败最多触发一次最终生成retry；retry复用完全相同的selected evidence/spans，不重复Planner、MCP、Connector或真实工具。实际provider messages必须包含服务端生成、脱敏、有界的失败类型和缺失source/field枚举，不包含模型原文、完整摘要、密钥或内部对象。

第二候选仍失败则按真实最终错误ABSTAIN；不得恢复第一候选，不得第三次生成，不得Template伪装DeepSeek成功。Citation Guard继续逐claim fail closed，阈值和支持谓词不变。

## 4. 冻结文件白名单

产品文件仅允许：

1. `backend/storage/logger.py`：authors JSON新写入、旧形状安全读取、当前hash覆盖标志的服务器重算判定。
2. `backend/resources/normalizer.py`：有界作者规范化、`resource_content()`和`content_hash()`包含相同作者值。
3. `backend/agent/orchestrator.py`：Paper请求解析、查询构造、有界over-fetch参数、双语相关性/去重/选择、作者span语义、预检终态、逐篇字段和实际Citation URL校验、一次同证据retry。
4. `backend/llm/prompts.py`：把服务器冻结的selected sources、字段、作者摘要语义和安全retry指导送入真实GroundedSpanAnswer messages；prompt只是辅助，服务器校验是确定性门。

测试文件仅允许：

1. `tests/test_j1_resource_provenance.py`
2. `tests/test_agent_orchestration.py`
3. `tests/test_r3_grounded_answer_contract.py`

明确禁止修改 `backend/guards/citation_guard.py`、`backend/llm/schemas.py`、router/state、resource service、Semantic Scholar/arXiv Connector、MCP、前端、数据库Schema/迁移、配置、`.env`、超时、重试预算、全局工具预算、权限/RBAC、Practice、创新和teacher。

如果实施需要第八个文件、第五个产品文件或第四个测试文件，9J必须输出 `SCOPE_REVIEW_REQUIRED` 并停止。现有工作区其他历史改动不是9J授权范围。

## 5. 9J冻结测试矩阵

所有自动化测试使用离线受控fixture；真实固定问题只留给9K。

先红后绿：

- 新array作者真实SQLite写入/回读，旧wrapper兼容，非法JSON安全为空；
- list形状但当前hash重算不一致时 `authors_hash_covered=false`；verified resync后重算相等才为true；
- 仅作者变化使`resource_content`和hash变化；
- 长摘要仍产生≤300字符作者摘要span；超长列表标节选且不截断姓名；恶意文本不能升级为指令；
- 每个selected paper逐项title/作者摘要/abstract clue完整；未选论文不能进入messages、答案或citations；
- 同一paper Citation DTO真实HTTPS URL和provenance身份校验，布尔标志不足以通过；
- 3篇完整正例、1篇和2篇诚实partial正例；零相关、相关但字段全缺分别得到稳定错误；
- 正确span但遗漏字段仅重试一次；第二候选仍缺字段时安全拒答；
- 至少四组中文主题→英文论文正例和多组NOMA/无关5G负例；定位符、引号短语及未知显著词不被净化掉；
- `source=auto`仍只有一次Agent工具调用/一次MCP请求；首来源成功但结果不相关时不产生第二次Agent工具调用；既有上游失败→arXiv fallback兼容；
- 错source/span、伪造作者、无摘要、重复paper、prompt injection、无HTTPS provenance继续拒绝；
- 同证据retry不重复上游，DeepSeek失败不由Template接管。

定向文件通过后，运行现有兼容回归：GitHub、NVD、Standard、本地课程、course-only、Practice、Citation Guard、S2引用和三来源结构化链；Multi-source测试必须使用同一当前run的paper/vulnerability/project证据，不能拼历史单项PASS。最后在稳定代码上运行一次全新隔离完整pytest，要求0 failed、0 errors，skip/xfail/warnings单列且skip不算通过。

## 6. 9J实施、回滚和成功门

9J按“红测→作者存储/hash→查询/相关性→作者span与完整性→retry→定向与兼容→完整pytest”的顺序实施。正式数据库、`.env`和白名单外文件不得变化；不运行真实DeepSeek、Connector、服务或浏览器。

回滚仅恢复4个产品文件和3个测试文件；无Schema迁移，无数据库回滚。新array格式仍是合法JSON；旧wrapper继续由新reader兼容。不得批量删除缓存。

9J全部通过时唯一成功令牌为：

```text
IMPLEMENTATION_READY_FOR_PAPER_FINAL_ACCEPTANCE
command_9K_allowed=true
```

## 7. 9K冻结验收口径

9K只创建一个固定Paper run；Paper通过后再创建一个固定Multi-source run。不得重跑其他PASS、完整pytest或历史页面矩阵。

Paper固定问题保持：

`请使用真实论文搜索检索 5G core network attack detection，返回 3 篇论文的标题、作者、摘要线索和可核验来源链接；不要用本地论文样例代替真实联网结果。`

Paper PASS允许1–3篇真实、互不重复、相关且字段完整的论文。每篇必须有逐字受支持的title、作者信息摘要和abstract clue，并具有同一论文Citation DTO中的真实HTTPS URL；少于3篇时必须在稳定终态/计数中诚实表明不足。必须使用DeepSeek、`fallback_used=false`、Guard通过。9K只把Agent层 `paper_search_real=1` 和MCP请求一次作为调用次数门；Planner次数和MCP内部按既有 `source=auto` 发生的Connector尝试次数仅记录，不作为绝对失败条件。

Paper为0篇相关、0篇字段完整、429/上游不可用或产生产品语义失败时，按真实分支保存唯一run并停止；不得另建成功run。

Paper通过后，Multi-source固定问题改为：

`请检索真实的5G安全论文、NVD中的CVE-2021-44228，以及GitHub上的Open5GS项目；请分别从Paper、NVD和GitHub各给出一条由单一来源span逐字支持的原子事实并显示引用和可核验来源，不要进行跨来源推理，也不要补充无证据的实验建议。`

Multi-source必须在当前run取得三类可信证据，每类至少一条单span逐字支持事实和正确HTTPS引用；保持每类来源一项的原有语义。它不要求跨来源推理或无证据实验建议。最多一次同证据生成retry，不重复上游调用。

Paper和Multi-source均通过后，才与9E、9D-S1和8V证据合并为 `POST_V6_MANUAL_ACCEPTANCE_PASS`。429或第三方不可用保留 `PARTIAL / THIRD_PARTY_LIMITATION`；Paper语义失败只诊断Paper唯一run；Paper已PASS而Multi-source失败时永久保留Paper PASS并只诊断Multi-source唯一run。

## 8. 最终裁决

上述修订消除了9H中的过严三篇门、作者越界和hash误信，也给出了可离线证明的双语相关性、稳定终态和调用边界。修订没有降低Citation Guard、权限或外部来源真实性，也没有把Multi-source未执行误报为缺陷。

`APPROVED_FOR_IMPLEMENTATION`  
`command_9J_allowed=true`
