# 第三阶段最终架构图

```mermaid
flowchart TD
    Student[学生端 Streamlit] --> API[FastAPI]
    Teacher[教师端 Streamlit] --> API
    API --> Auth[JWT/RBAC/课程班级范围]
    API --> Events[学习会话与学习事件]
    API --> Agent[有限 Agent Loop]
    API --> Resources[课程资料库 Resource Portal]
    API --> Innovation[创新启发服务]
    API --> Assignments[实验任务与进度]
    API --> TeacherAnalytics[教师统计与聚类]
    Agent --> LLM[DeepSeek 或 TemplateProvider]
    Agent --> RAG[Hybrid Retriever]
    Agent --> MCP[MCP Client Gateway]
    MCP --> Tools[MCP Tools]
    Resources --> Connectors[Semantic Scholar/arXiv/NVD/标准/GitHub]
    Resources --> ResourceCache[(SQLite external_resources)]
    RAG --> Docs[(SQLite documents/chunks)]
    RAG --> Vector[(Chroma)]
    Events --> DB[(SQLite)]
    Assignments --> DB
    TeacherAnalytics --> DB
    Innovation --> DB
```

## 运行边界

- 普通离线模式：TemplateProvider、本地知识库、本地 MCP fallback、SQLite 和缓存资源。
- DeepSeek 模式：`.env` 配置 `DEEPSEEK_API_KEY` 后启用真实 Planner/Answer Provider。
- Live Connector 模式：显式设置 live smoke 环境变量后访问真实论文源或 NVD。
