# 原始要求追踪表

本文档把老师原始要求逐项映射到当前模块、第三阶段任务和验收证据。第三阶段第十一步后，身份、课程资料、创新启发、问题聚类、实验进度、卡点分析、教师端教学摘要、学生端使用说明和安全边界已经形成可测试闭环。

## 总览

| 原始要求 | 当前模块 | 第三阶段实现/任务 | 验收证据 |
|---|---|---|---|
| ① 基于 MCP 模型开发教学智能体，辅助课程学习、实验和创新 | `mcp_server/`、`backend/tools/mcp_client.py`、`backend/agent/orchestrator.py`、`backend/llm/` | 已接入 MCP 工具目录、Agent Loop、DeepSeek/TemplateProvider、创新启发、实验步骤上下文 | `/api/tools`、Agent trace、tool_calls、run_id、引用与拒答测试 |
| ① 统一上下文交互 | `backend/agent/state.py`、`agent_runs`、`agent_steps`、`learning_sessions`、`learning_events` | 用户、课程、班级、学习会话、实验任务和 step_id 纳入上下文 | 同一 run_id 继续执行、学生时间线、实践指导关联实验步骤测试 |
| ① 支持外部工具、数据库、论文库、漏洞库接口 | `backend/connectors/`、`backend/resources/`、`mcp_server/tools.py` | 已实现 Connector 框架、Semantic Scholar、arXiv、NVD、标准元数据和资源缓存 | `tests/connectors`、`pytest -m live -q` 可选 smoke、缓存命中与降级测试 |
| ② 构建领域知识库 | `backend/rag/document_parser.py`、`backend/rag/vector_store.py`、`backend/rag/hybrid_retriever.py`、`documents`、`document_chunks` | 支持本地文档上传、切片、hash 去重、Chroma 索引和混合检索；外部资源可缓存并按需索引 | 文档解析测试、向量检索测试、引用定位字段 |
| ② 高效检索 | `HybridRetriever`、`ResourcePortalService` | TF-IDF 与 Chroma 分数归一化融合，资源门户支持关键词、专题、类型、年份、来源筛选 | Recall@K、知识库命中率、fallback_used、资源筛选与分页测试 |
| ③a 教师端访问监控 | `backend/analytics/events.py`、`activity.py`、`backend/api/teacher_phase3.py`、`frontend/pages/teacher.py` | 记录 LOGIN、PAGE_VIEW、RESOURCE_SEARCH、RESOURCE_OPEN、QUESTION_ASKED、实践、创新、进度、卡点和反馈事件；支持课程/班级/时间筛选 | `tests/test_teacher_phase3.py`、教师活动统计、学生时间线、权限测试 |
| ③a 查看学生访问情况、活跃度、学习轨迹 | `learning_sessions`、`learning_events` | heartbeat 合理累计 active_seconds，空闲超时不累计，按课程/班级聚合 | `tests/test_learning_events.py` |
| ③b 问题聚类 | `backend/analytics/clustering.py`、`question_clusters`、`question_cluster_members` | 使用 embedding 语义聚类，支持教师重命名、合并、拆分和批准；批准结果不被自动重建覆盖 | `tests/test_question_clustering.py` |
| ③c 统计分析：模块/实验卡壳 | `backend/analytics/stuck_analysis.py`、`assignments`、`assignment_steps`、`student_assignment_progress`、`stuck_reports` | 学生可更新实验进度、主动上报卡点；教师可查看主动和疑似卡点，并确认/忽略/解决 | `tests/test_assignment_progress.py`、教师端实验步骤卡点看板 |
| ③d FAQ 输出 | `QuestionClusteringService.export_faq`、教师端 CSV 导出 | FAQ 来源扩展到语义聚类，建议回答经过本地知识库检索和引用检查，CSV 做公式注入防护 | FAQ 导出测试、`tests/test_teacher_exports.py` |
| ④ 学生端基础模块：使用说明 | `frontend/pages/student.py`、README、操作文档 | 页面展示问答、资源、实践、创新和实验任务入口，并补充课程简介、考核方式、专题示例和网络安全实验边界 | `tests/test_security_phase3.py`、Streamlit 页面、学习事件 PAGE_VIEW |
| ④ 学生端基础模块：课程简介 | `courses`、`classes`、README | 已有课程/班级 API 与默认 demo course，可继续补页面化课程目标、考核方式和专题示例 | `/api/courses`、`/api/classes`、身份测试 |
| ④ 学生端基础模块：课程资料 | `ResourcePortalService`、`frontend/pages/student.py` | 统一展示课程文档、论文、标准、漏洞、开源项目和竞赛题目 | `tests/test_resource_portal.py` |
| ④ 核心模块：实践指导/问题引导 | `/api/practice/guide`、`CourseAnswerer`、实验任务接口 | 实践指导可携带 assignment_id 和 step_id，Agent 可根据当前步骤继续指导 | `tests/test_assignment_progress.py::test_practice_guide_can_attach_assignment_and_step` |
| ④ 核心模块：思维框架 | `backend/rag/answerer.py`、`PracticeGuideResponse` | 保持“研究目标 -> 方法设计 -> 工具选择 -> 实验步骤 -> 结果分析 -> 创新点”结构 | 实践指导接口和页面展示 |
| ④ 核心模块：智能检索 | `HybridRetriever`、`MCP tools`、`connectors` | 可检索本地知识库、论文、标准、漏洞、项目和竞赛数据，自动摘要与分类由资源服务和 Agent 组合完成 | 检索评测、Connector mock/live smoke |
| ④ 创新启发模块：案例推荐 | `backend/innovation/service.py`、`recommender.py`、`frontend/pages/innovation.py` | 独立创新会话，至少推荐论文、项目或竞赛案例中的两类，事实推荐带来源 | `tests/test_innovation.py` |
| ④ 创新启发模块：创新提问 | `innovation_questions`、`innovation_ideas` | 生成 3-5 个启发式问题，保存学生回答，最终输出候选创新方向、假设、实验方法、风险和证据 | `tests/test_innovation.py` |

## 当前模块映射

| 能力 | 当前文件或模块 |
|---|---|
| FastAPI API | `backend/main.py`, `backend/schemas.py` |
| Streamlit 前端 | `frontend/app.py`, `frontend/pages/student.py`, `frontend/pages/teacher.py`, `frontend/api_client.py` |
| 身份与权限 | `backend/auth/`, `backend/courses/` |
| 学习日志与迁移 | `backend/storage/logger.py` |
| 学习事件与活动统计 | `backend/analytics/events.py`, `backend/analytics/activity.py` |
| 第三阶段教师统计 API | `backend/api/teacher_phase3.py` |
| 实验任务与卡点分析 | `backend/analytics/stuck_analysis.py`, `assignments`, `assignment_steps`, `student_assignment_progress`, `stuck_reports` |
| 文档解析与知识库 | `backend/rag/document_parser.py`, `backend/rag/vector_store.py`, `backend/rag/hybrid_retriever.py` |
| LLM Provider | `backend/llm/base.py`, `backend/llm/deepseek_provider.py`, `backend/llm/template_provider.py`, `backend/llm/factory.py` |
| Agent Loop | `backend/agent/state.py`, `backend/agent/planner.py`, `backend/agent/orchestrator.py`, `backend/agent/observation.py` |
| MCP 工具 | `mcp_server/server.py`, `mcp_server/tools.py`, `backend/tools/mcp_client.py`, `backend/tools/tool_policy.py` |
| 资源 Connector | `backend/connectors/`, `backend/resources/` |
| 创新启发 | `backend/innovation/` |
| 问题聚类 | `backend/analytics/clustering.py` |
| 安全守卫 | `backend/guards/action_guard.py`, `backend/guards/citation_guard.py`, `backend/guards/safety_guard.py` |
| 评测与 smoke | `scripts/smoke_phase2.py`, `scripts/evaluate_phase2.py`, `tests/` |

## 第三阶段验收证据清单

- 测试：`pytest -q` 当前为 `187 passed, 2 skipped, 12 warnings`。
- 第十一步专项：`pytest tests/test_security_phase3.py tests/test_auth_rbac.py -q` 当前为 `18 passed, 1 warning`。
- 第十步专项：`pytest tests/test_teacher_phase3.py tests/test_auth_rbac.py -q` 当前为 `16 passed, 1 warning`。
- 第十二步 E2E：`pytest tests/test_phase3_e2e.py -q` 当前为 `1 passed, 1 warning`。
- 第九步专项：`pytest tests/test_assignment_progress.py -q` 当前为 `5 passed, 1 warning`。
- 文档：README、`docs/phase3_architecture.md`、本追踪表、`docs/phase3_testing_report.md`、`docs/phase3_final_report.md`、`docs/phase3_data_sources_licenses.md`、`docs/phase3_privacy_security.md`、`docs/phase3_final_architecture.md`、`docs/phase3_database_diagram.md` 和 `docs/phase3_demo_script.md`。
- 页面：学生端课程资料库、创新启发、实验任务；教师端访问监控、学习轨迹、语义聚类、实验步骤卡点。
- 数据：SQLite 中有用户、课程、班级、学习事件、资源缓存、Connector 同步、创新会话、实验进度、卡点和聚类记录。
- 安全：`.env.example` 不含密钥，`.env` 被忽略；教师接口有角色限制；CSV 导出有公式注入防护；外部工具调用可审计；学生不能调用 Connector 同步、知识库删除、聚类管理、教师统计或 MCP 调试入口；外部资源 URL 和提示注入内容会被过滤或拦截。
- 完整演示：`tests/test_phase3_e2e.py` 串联学生端、教师端、课程资料、外部 mock 元数据、本地引用、实验任务、创新启发、学习轨迹、聚类、FAQ 和权限拒绝。
