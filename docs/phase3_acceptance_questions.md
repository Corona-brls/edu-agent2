# 第三阶段最终验收问题对照表

## 学生端

| 验收问题 | 当前实现 | 验收证据 |
|---|---|---|
| 如何使用课程智能体完成一次从选题到实验方案的完整流程？ | 学生端“使用说明”、智能问答、实践指导、实验任务和创新启发形成完整流程 | `frontend/pages/student.py`、`frontend/pages/innovation.py`、`docs/phase3_demo_script.md` |
| 课程目标、考核方式和专题示例是什么？ | 学生端“使用说明”展示课程简介、课程目标、考核方式和专题示例 | `tests/test_security_phase3.py` |
| 请从真实论文源查找 5G NAS 安全相关研究，并显示来源和更新时间。 | Connector 支持 Semantic Scholar/arXiv；普通测试用 mock 缓存，live smoke 可访问真实源 | `backend/connectors/semantic_scholar.py`、`backend/connectors/arxiv.py`、`tests/connectors`、`pytest -m live -q` |
| 请查询一个适合作为课程实验分析样例的 CVE，并显示 CVSS、CWE 和 NVD 链接。 | NVD Connector 返回 CVE、CVSS、CWE、发布日期、更新时间和 NVD 链接 | `backend/connectors/nvd.py`、`tests/connectors/test_real_connectors.py`、`tests/test_phase3_e2e.py` |
| 3GPP TS 24.501 的官方元数据和本地正文分别来自哪里？ | 标准 Connector 返回官方元数据；本地正文来自教师上传或本地知识库文档 | `backend/connectors/standard_source.py`、`backend/resources/portal.py`、`tests/test_phase3_e2e.py` |
| 请为“基于区块链的隐蔽通信”生成实践方案并关联实验任务。 | `/api/practice/guide` 支持 `assignment_id` 和 `step_id`，学生端实践指导可携带当前实验步骤 | `tests/test_assignment_progress.py` |
| 我在实验的结果分析步骤遇到困难，请记录卡点并继续指导。 | `/api/stuck-reports` 记录主动卡点，教师端卡点分析可查看；实践指导继续使用任务上下文 | `backend/analytics/stuck_analysis.py`、`tests/test_assignment_progress.py` |
| 请根据我的当前方案推荐论文和开源项目，并提出三个启发式创新问题。 | 创新启发服务检索课程资料、外部资源和允许的 MCP 工具，生成案例、限制和 3-5 个启发式问题 | `backend/innovation/service.py`、`tests/test_innovation.py`、`tests/test_phase3_e2e.py` |
| 我回答创新问题后，系统能否继续同一创新会话并生成候选创新方向？ | 创新会话持久化，支持回答问题、追问和 finalize 输出候选创新方向 | `innovation_sessions`、`innovation_questions`、`innovation_ideas`、`tests/test_innovation.py` |

## 教师端

| 验收问题 | 当前实现 | 验收证据 |
|---|---|---|
| 本周有多少独立活跃学生，各自最近活跃时间是什么？ | 教师端第三阶段摘要和学生时间线 API 统计独立活跃学生、最近活跃和页面分布 | `backend/api/teacher_phase3.py`、`tests/test_teacher_phase3.py` |
| 某学生本周访问了哪些模块、资料和实验步骤？ | 学习事件记录 PAGE_VIEW、RESOURCE_SEARCH、RESOURCE_OPEN、QUESTION_ASKED、ASSIGNMENT_PROGRESS_UPDATED 等 | `backend/analytics/events.py`、`tests/test_learning_events.py` |
| 语义相近但表述不同的问题能否形成同一热点问题簇？ | 问题聚类服务使用 embedding/TF-IDF 与 DBSCAN 聚类 | `backend/analytics/clustering.py`、`tests/test_question_clustering.py` |
| 教师能否合并、拆分、重命名并批准问题簇？ | 聚类 API 支持 rename、merge、split、approve | `backend/main.py` 聚类接口、`tests/test_question_clustering.py` |
| 哪些实验步骤被主动报告为卡点，哪些只是系统推断的疑似卡点？ | 卡点分析区分 `active` 主动上报和 `suspected` 疑似卡点，并提供证据 | `backend/analytics/stuck_analysis.py`、`tests/test_assignment_progress.py` |
| 能否按课程、班级、模块和时间范围导出审核后的 FAQ？ | 教师端支持课程/班级/时间筛选，FAQ CSV 导出做公式注入防护 | `backend/api/teacher_phase3.py`、`tests/test_teacher_phase3.py` |

## 安全与降级

| 验收问题 | 当前实现 | 验收证据 |
|---|---|---|
| 学生账号访问教师统计接口是否返回 403？ | 教师统计、知识库管理、Connector 同步和聚类管理均使用角色依赖 | `tests/test_security_phase3.py`、`tests/test_auth_rbac.py` |
| Semantic Scholar 或 NVD 不可用时是否显示缓存或本地结果？ | Connector 失败时返回明确状态，资源门户可展示缓存或本地知识库结果 | `backend/resources/service.py`、`tests/connectors`、`tests/test_resource_portal.py` |
| 外部结果是否带来源、获取时间和缓存状态？ | 资源门户统一展示 `source_name`、`fetched_at`、`cache_status` 和来源链接 | `backend/resources/portal.py`、`tests/test_resource_portal.py` |
| 上传资料中的提示注入是否会触发未授权工具？ | Safety Guard 将文档和 Observation 视为不可信内容，Action Guard 拦截未授权或高风险工具 | `backend/guards/safety_guard.py`、`backend/guards/action_guard.py`、`tests/test_security_phase3.py` |
| 日志、配置页、CSV 和错误响应是否泄露密钥或令牌？ | `.env.example` 为空占位，系统状态不显示密钥，CSV 转义，测试扫描敏感片段 | `tests/test_security_phase3.py`、`docs/phase3_privacy_security.md` |

## 建议现场演示顺序

1. 运行 `pytest tests/test_phase3_e2e.py -q` 展示完整离线闭环。
2. 启动后端和 Streamlit 前端。
3. 学生登录，展示使用说明、资料检索、实践指导、实验任务、卡点和创新启发。
4. 教师登录，展示第三阶段摘要、学生时间线、聚类、卡点分析和 FAQ 导出。
5. 用学生账号访问教师统计接口，确认 403。
6. 说明真实论文源/NVD/DeepSeek 需要 `.env` 和 live smoke 显式启用。
