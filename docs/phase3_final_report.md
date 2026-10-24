# 第三阶段最终报告材料

## 项目概述

本项目已经从本地课程问答原型扩展为 `Streamlit + FastAPI + SQLite + Chroma + MCP + DeepSeek/TemplateProvider` 的课程智能体。系统支持学生学习、课程资料检索、实践指导、创新启发、实验进度、卡点反馈，以及教师端访问监控、学习轨迹、问题聚类、卡点分析和 FAQ 导出。

## 关键实现

- 身份与权限：本地 JWT、Argon2 密码哈希、student/teacher/admin 角色和课程/班级范围控制。
- 知识库：Markdown/TXT/PDF/DOCX 解析、SQLite 文档与 chunk 记录、Chroma 向量库、TF-IDF + 向量混合检索。
- Agent：有限步骤 Agent Loop、Action Guard、Citation Guard、Safety Guard、TemplateProvider 离线降级和 DeepSeek Provider。
- MCP 与 Connector：本地 MCP 工具、工具策略、MCP Gateway、Semantic Scholar/arXiv/NVD/标准元数据/GitHub Connector 框架。
- 学生端：使用说明、课程简介、课程资料库、智能问答、实践指导、创新启发和实验任务。
- 教师端：访问监控、学习轨迹、模型与工具统计、语义聚类、卡点分析、FAQ 审核和安全 CSV 导出。

## 验收证据

- `tests/test_phase3_e2e.py` 覆盖第三阶段完整离线演示流程。
- `tests/test_teacher_phase3.py` 覆盖教师端课程/班级/时间筛选、CSV 安全导出和权限隔离。
- `tests/test_security_phase3.py` 覆盖密钥泄露、角色权限、提示注入、CSV 注入、缓存降级和高风险工具限制。
- `tests/connectors` 覆盖 Connector 缓存、限流、超时、失败和向量同步合同。
- `tests/test_auth_rbac.py` 覆盖登录、令牌、密码哈希、角色边界和旧数据回填。
- `docs/phase3_acceptance_questions.md` 对老师最终验收问题逐项映射到代码模块、测试和演示证据。

## 限制

- 默认使用 SQLite 与本地 Chroma，适合课程演示和单机部署；多人生产环境建议迁移到 PostgreSQL、对象存储和任务队列。
- 普通测试默认不访问公网；真实 Connector 与 DeepSeek 需要本机 `.env` 配置密钥并显式运行 live smoke。
- 标准文档 Connector 只返回元数据和官方链接，不抓取受版权保护全文。
- 网络安全实验只支持授权教学、资料检索、防御分析和本地实验设计，不生成公网攻击执行步骤。

## 后续工作

- 增加教师可视化编辑实验任务模板。
- 增加更细粒度的课程资料审核流。
- 将长任务同步、文档索引和 Connector 同步迁移到后台队列。
- 在真实班级中持续校准问题聚类阈值和 FAQ 审核流程。
