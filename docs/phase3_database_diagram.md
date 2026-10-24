# 第三阶段数据库图

```mermaid
erDiagram
    users ||--o{ class_members : joins
    courses ||--o{ classes : contains
    classes ||--o{ class_members : has
    courses ||--o{ questions : scopes
    classes ||--o{ questions : scopes
    questions ||--o{ student_feedback : receives
    questions ||--o{ question_cluster_members : clusters
    question_clusters ||--o{ question_cluster_members : has
    courses ||--o{ documents : owns
    documents ||--o{ document_chunks : splits
    documents ||--o{ index_jobs : indexes
    users ||--o{ learning_sessions : starts
    learning_sessions ||--o{ learning_events : records
    users ||--o{ agent_runs : owns
    agent_runs ||--o{ agent_steps : traces
    agent_runs ||--o{ tool_calls : calls
    agent_runs ||--o{ model_calls : calls
    external_resources ||--o{ connector_sync_runs : cached_by
    users ||--o{ innovation_sessions : creates
    innovation_sessions ||--o{ innovation_questions : asks
    innovation_sessions ||--o{ innovation_ideas : outputs
    courses ||--o{ assignments : has
    assignments ||--o{ assignment_steps : contains
    users ||--o{ student_assignment_progress : updates
    users ||--o{ stuck_reports : reports
```

## 关键表

- 身份课程：`users`, `courses`, `classes`, `class_members`
- 学习日志：`questions`, `conversations`, `learning_sessions`, `learning_events`, `student_feedback`
- Agent 追踪：`agent_runs`, `agent_steps`, `model_calls`, `tool_calls`, `agent_traces`
- 知识库：`documents`, `document_chunks`, `index_jobs`
- 外部资源：`external_resources`, `connector_sync_runs`
- 创新启发：`innovation_sessions`, `innovation_questions`, `innovation_ideas`
- 教师分析：`question_clusters`, `question_cluster_members`, `assignments`, `assignment_steps`, `student_assignment_progress`, `stuck_reports`
