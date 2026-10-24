# 网络安全课程智能体开发参考报告

## 1. 项目定位

项目建议名称：

**面向网络安全实践课程的 MCP-RAG 教学智能体系统**

本项目分为两个阶段完成。

第一阶段是**本地可演示版**，目标是先完成一个能跑通、能演示、能写报告的课程智能体系统。技术路线尽量轻量，采用 `Streamlit + FastAPI + SQLite + 本地 Markdown 知识库 + TF-IDF/关键词检索 + 工具注册层`。这一阶段不追求完整生产级平台，而是优先保证系统闭环完整、功能可展示、代码容易理解、报告容易撰写。

第二阶段是**完整增强版**，目标是在第一阶段基础上升级为更完整的 MCP-RAG 智能体平台。增强内容包括 DeepSeek LLM、标准 MCP Server、真实论文库/漏洞库/标准库、向量数据库、轻量 Agent 编排、用户权限和部署方案等。第二阶段先做好设计，等本地版稳定后再逐步扩展。

---

## 2. 参考项目

### 2.1 EduAgent

项目链接：

[https://github.com/hukdis/EduAgent](https://github.com/hukdis/EduAgent)

参考重点：

- 教育场景 RAG 架构。
- FastAPI 后端服务。
- 前端交互页面。
- MCP 工具注册。
- 引用回答。
- 教师反馈与评测。
- 课程智能体的整体系统分层。

本项目借鉴方式：

- 借鉴“课程问答 + 工具调用 + 教师分析”的总体架构。
- 借鉴“回答必须带引用来源”的可信 RAG 思路。
- 借鉴工具能力统一注册、统一调用的设计方式。
- 不直接复制代码，只参考架构思想。

### 2.2 TeaChat

项目链接：

[https://github.com/time1527/TeaChat](https://github.com/time1527/TeaChat)

参考重点：

- 中文教育智能系统。
- Multi-Agent RAG。
- 学科知识组织。
- BM25 + FAISS 混合检索。
- 教学链路设计。
- 面向学生学习过程的问答与引导。

本项目借鉴方式：

- 借鉴“知识定位 -> 知识讲解 -> 实践巩固”的教学流程。
- 借鉴中文课程资料的组织方式。
- 借鉴高频问题和知识点聚类思路。
- 第一阶段不做复杂 Multi-Agent，只用规则路由模拟教学链路。

### 2.3 openai_rag_project

项目链接：

[https://github.com/channico/openai_rag_project](https://github.com/channico/openai_rag_project)

参考重点：

- 最小 RAG 闭环。
- Agent Tool。
- MCP Server 示例。
- 引用回答。
- 拒答机制。
- RAG 工程最小实现路线。

本项目借鉴方式：

- 先完成最小可运行 RAG。
- 每个回答返回 `answer`、`citations`、`confidence`、`abstained`。
- 低证据问题拒答，避免编造。
- 后续再逐步升级到标准 MCP 和更复杂 Agent。

---

## 3. 总体开发策略

本项目不要一开始就做复杂系统。推荐采用“先本地闭环，再逐步增强”的方式。

### 3.1 第一阶段目标

第一阶段要完成：

- 后端可以启动。
- 前端可以打开。
- 本地知识库可以检索。
- 学生可以提问。
- 系统可以根据资料回答。
- 回答可以显示引用来源。
- 没有资料的问题可以拒答。
- 实践指导可以生成结构化方案。
- SQLite 可以记录提问日志。
- 教师端可以查看统计和 FAQ。
- 项目可以用 8 个验收问题完整演示。

### 3.2 第二阶段目标

第二阶段再考虑：

- 标准 MCP Server。
- Chroma / FAISS / pgvector 向量数据库。
- 真实论文检索。
- 真实 CVE / NVD 漏洞库查询。
- 真实标准文档检索。
- DeepSeek LLM 生成与结构化输出。
- 轻量 Agent Orchestrator 编排。
- React 前端。
- 用户权限。
- Docker 部署。

---

## 4. 第一阶段：本地可演示版功能设计

### 4.1 学生端功能

学生端主要服务学生学习、实验和创新实践。

功能包括：

- 课程介绍：展示课程目标、考核方式、专题示例、资料入口。
- 智能问答：学生输入问题，系统基于本地知识库回答。
- 引用展示：每个回答展示来源文件、标题和引用片段。
- 实践指导：学生输入专题后，系统生成实践方案。
- 专题导航：支持 5G NAS、区块链隐蔽通信、LLM 漏洞检测、课程智能体四类专题。
- 资料检索：支持检索课程资料、论文样例、标准样例、漏洞样例。

实践指导输出结构：

```text
研究目标
方法设计
工具选择
实验步骤
结果分析
创新点
风险与注意事项
```

### 4.2 教师端功能

教师端主要服务课程管理、答疑和教学优化。

功能包括：

- 总提问数统计。
- 最近提问记录。
- 高频模块统计。
- 拒答次数统计。
- 高频问题汇总。
- 学生卡点分析。
- FAQ 导出。
- 工具状态查看。

教师端的价值：

- 发现学生集中卡在哪些专题。
- 自动整理课堂答疑问题。
- 辅助教师调整课程内容。
- 为实践报告提供系统闭环证明。

### 4.3 后端功能

后端负责业务逻辑和数据处理。

功能包括：

- 提供统一 API。
- 加载本地知识库。
- 执行检索。
- 生成回答。
- 判断是否拒答。
- 调用工具注册层。
- 保存提问日志。
- 提供教师统计数据。

---

## 5. 第一阶段技术架构

### 5.1 推荐技术栈

必选技术：

```text
Python 3.10+
FastAPI
Uvicorn
Streamlit
SQLite
scikit-learn
jieba
pandas
requests
pydantic
python-multipart
```

各技术用途：

| 技术 | 用途 |
|---|---|
| FastAPI | 后端 API 服务 |
| Uvicorn | 启动 FastAPI |
| Streamlit | 快速构建前端页面 |
| SQLite | 保存学习日志 |
| scikit-learn | TF-IDF 检索 |
| jieba | 中文分词 |
| pandas | 教师端统计 |
| requests | 前端调用后端 |
| pydantic | 请求和响应结构 |
| python-multipart | 文件上传 |

### 5.2 第一阶段暂不使用的技术

第一阶段建议暂不使用：

```text
React
Docker
Chroma
FAISS
LangGraph
标准 MCP SDK
真实论文 API
真实漏洞 API
```

原因：

- React 开发量较大，第一版用 Streamlit 更快。
- Docker 第一版不是必须，容易增加环境问题。
- Chroma / FAISS 后续可替换，第一版用 TF-IDF 足够演示。
- LangGraph 适合复杂 Agent，第一版用规则路由即可。
- 标准 MCP SDK 可以后续升级，第一版先做工具注册层。

---

## 6. 推荐目录结构

```text
course-agent/
├── README.md
├── requirements.txt
├── .env.example
├── backend/
│   ├── __init__.py
│   ├── main.py
│   ├── schemas.py
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   ├── retriever.py
│   │   └── answerer.py
│   ├── agent/
│   │   ├── __init__.py
│   │   └── router.py
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── registry.py
│   │   └── mcp_demo_server.py
│   ├── storage/
│   │   ├── __init__.py
│   │   └── logger.py
│   └── data/
│       └── app.db
├── frontend/
│   ├── app.py
│   ├── api_client.py
│   └── pages/
│       ├── student.py
│       └── teacher.py
├── knowledge_base/
│   ├── course_intro.md
│   ├── 5g_nas.md
│   ├── blockchain_covert.md
│   ├── llm_vuln_detection.md
│   ├── cve_samples.md
│   └── standards_samples.md
├── docs/
│   ├── architecture.md
│   ├── demo_script.md
│   └── report_outline.md
└── tests/
    └── test_demo_questions.py
```

---

## 7. 后端 API 设计

### 7.1 接口列表

```text
GET  /api/health
POST /api/chat
POST /api/practice/guide
POST /api/knowledge/upload
GET  /api/teacher/analytics
GET  /api/teacher/faq
GET  /api/tools
```

### 7.2 `/api/health`

用途：

检查后端服务是否正常。

返回示例：

```json
{
  "status": "ok",
  "service": "course-agent"
}
```

### 7.3 `/api/chat`

用途：

学生智能问答。

请求示例：

```json
{
  "question": "5G NAS 认证流程有哪些关键风险点？",
  "user_id": "demo_student"
}
```

返回示例：

```json
{
  "answer": "5G NAS 认证流程的关键风险点主要包括身份暴露、认证上下文处理不当、重放风险、异常状态迁移等。本地知识库中建议从认证、加密、会话管理三个角度进行分析。",
  "citations": [
    {
      "source": "5g_nas.md",
      "title": "5G NAS 安全风险",
      "snippet": "NAS 信令涉及认证、加密、会话管理等安全相关流程。"
    }
  ],
  "confidence": 0.82,
  "abstained": false,
  "module": "5G NAS",
  "intent": "course_qa"
}
```

### 7.4 `/api/practice/guide`

用途：

根据学生输入的专题生成实践指导。

请求示例：

```json
{
  "topic": "基于区块链的隐蔽通信",
  "user_id": "demo_student"
}
```

返回内容包括：

```text
研究目标
方法设计
工具选择
实验步骤
结果分析
创新点
```

### 7.5 `/api/knowledge/upload`

用途：

上传本地知识库资料。

第一阶段只支持简单 Markdown / TXT 上传即可。

### 7.6 `/api/teacher/analytics`

用途：

返回教师端统计数据。

返回内容包括：

```text
总提问数
拒答次数
模块分布
最近问题
平均置信度
```

### 7.7 `/api/teacher/faq`

用途：

返回高频问题和 FAQ。

第一阶段可用简单关键词聚类或相同问题计数，不需要复杂聚类模型。

### 7.8 `/api/tools`

用途：

返回工具注册层中的工具列表。

返回示例：

```json
[
  {
    "name": "course_retriever",
    "description": "检索课程知识库",
    "risk": "LOW",
    "enabled": true
  },
  {
    "name": "vuln_lookup",
    "description": "检索本地漏洞样例",
    "risk": "LOW",
    "enabled": true
  }
]
```

---

## 8. RAG 流程设计

### 8.1 基础流程

```text
用户问题
-> 中文分词
-> TF-IDF/关键词检索
-> Top-K 文档片段召回
-> 简单重排
-> 计算置信度
-> 生成引用回答
-> 低置信度拒答
-> 写入 SQLite 日志
```

### 8.2 返回字段

每次问答都应该返回：

```text
answer
citations
confidence
abstained
module
intent
```

字段说明：

| 字段 | 含义 |
|---|---|
| answer | 最终回答 |
| citations | 引用来源 |
| confidence | 置信度 |
| abstained | 是否拒答 |
| module | 命中的课程模块 |
| intent | 用户意图 |

### 8.3 拒答机制

以下情况需要拒答：

- 检索分数过低。
- 知识库没有相关资料。
- 用户询问具体论文结论，但本地没有论文来源。
- 用户询问具体标准条款，但本地没有对应标准片段。
- 用户要求编造漏洞编号或不存在的 CVE。

拒答示例：

```text
当前本地知识库没有足够证据支持这个结论。建议补充相关论文、标准文档或漏洞样例后再查询。
```

### 8.4 知识库文件说明

```text
course_intro.md
课程目标、考核方式、专题介绍、系统能力说明。

5g_nas.md
5G NAS 概念、认证流程、安全风险、实验建议。

blockchain_covert.md
区块链隐蔽通信、地址派生、交易筛选、群组通信机制。

llm_vuln_detection.md
LLM 漏洞检测、幻觉检测、PoC 校验、漏洞报告核验。

cve_samples.md
本地漏洞样例和 CVE 示例。

standards_samples.md
标准文档样例和 3GPP TS 24.501 相关说明。
```

---

## 9. Agent 路由设计

第一阶段不做复杂多智能体，只做轻量规则路由。

### 9.1 意图类型

```text
course_qa
practice_guide
paper_search
vuln_lookup
standard_lookup
teacher_analytics
unknown
```

### 9.2 路由规则示例

```text
包含“怎么设计实验”“实践方案”“研究目标” -> practice_guide
包含“论文”“研究方向”“文献” -> paper_search
包含“漏洞”“CVE”“PoC” -> vuln_lookup
包含“标准”“3GPP”“TS 24.501” -> standard_lookup
其他问题 -> course_qa
```

### 9.3 教学链路

借鉴 TeaChat 的教学链路，回答尽量采用：

```text
知识定位
知识讲解
实践指导
拓展建议
引用来源
```

---

## 10. 工具注册层设计

第一阶段采用“类 MCP 工具层”，不强制接入标准 MCP SDK。

### 10.1 工具列表

```text
course_retriever      课程资料检索
paper_search          本地论文样例检索
vuln_lookup           本地漏洞样例检索
standard_lookup       标准片段检索
mermaid_generator     生成 Mermaid 架构图/流程图
```

### 10.2 工具结构

```json
{
  "name": "course_retriever",
  "description": "检索课程知识库",
  "risk": "LOW",
  "enabled": true
}
```

### 10.3 报告表述

可在报告中这样描述：

```text
本地版采用 MCP 思想设计统一工具注册层，所有外部能力通过标准化工具接口暴露。后续可替换为 MCP stdio server，与 Agent 通过标准协议通信。
```

---

## 11. SQLite 日志设计

### 11.1 questions 表

```text
id
user_id
question
answer_summary
module
intent
confidence
abstained
created_at
```

### 11.2 retrieval_logs 表

```text
id
question_id
source
score
snippet
```

### 11.3 faq_items 表

```text
id
cluster_label
representative_question
count
module
updated_at
```

### 11.4 教师端统计内容

```text
总提问数
拒答次数
高频模块
最近问题
高频 FAQ
卡点模块
FAQ 导出
```

---

## 12. 前端页面设计

### 12.1 Streamlit 页面结构

第一阶段建议只做一个 Streamlit 应用，用侧边栏切换页面：

```text
学生端
教师端
工具状态
项目说明
```

### 12.2 学生端页面

包含：

- 课程介绍卡片。
- 专题选择。
- 问答输入框。
- 回答展示区。
- 引用来源展示区。
- 实践指导生成区。

### 12.3 教师端页面

包含：

- 总提问数指标。
- 拒答次数指标。
- 模块分布图。
- 最近问题列表。
- 高频 FAQ 表格。
- FAQ 导出按钮。

### 12.4 工具状态页面

展示：

- 工具名称。
- 工具描述。
- 风险等级。
- 是否启用。

---

## 13. 第二阶段：完整增强版设计

第二阶段暂不实现，只作为后续升级方向。

### 13.0 DeepSeek LLM

- 通过统一 `LLMProvider` 接口接入 DeepSeek。
- DeepSeek 只根据检索证据生成答案，不直接访问数据库和文件。
- API Key 从本机 `.env` 读取，仓库仅保留空白占位。
- 未配置密钥或调用失败时回退到本地 `TemplateProvider`。

### 13.1 标准 MCP

升级内容：

- 使用标准 MCP stdio server。
- 每个工具独立暴露为 MCP tool。
- Agent 通过 MCP client 调用工具。
- 工具支持风险等级和启用开关。

扩展工具：

```text
paper_search_real      查询真实论文接口
cve_lookup_real        查询 NVD/CVE 数据
standard_fetcher       标准文档检索
github_search          查找开源项目/代码
web_fetch              受控网页读取
diagram_generator      Mermaid 图生成
```

### 13.2 真实知识库

升级内容：

- Chroma / FAISS / pgvector。
- PDF / Word / Markdown 多格式解析。
- embedding 模型。
- 文档增量索引。
- 文档版本管理。
- 引用定位到页码或章节。

### 13.3 有限自主 Agent 架构

完整架构采用单 Orchestrator 的受控决策循环：

```text
用户目标
-> DeepSeek Planner 选择下一动作
-> 后端 Action Guard 校验
-> 执行 RAG 或 MCP 工具
-> 写入 Observation
-> DeepSeek 继续决策
-> 回答 / 追问 / 拒答 / 达到步骤上限
```

原则：

- 回答必须经过检索。
- 无证据时拒答。
- 引用来源必须展示。
- 高风险工具默认关闭。
- 最大步骤和工具调用次数必须受配置限制。

### 13.4 完整前端

可选升级：

- React + Vite + Tailwind。
- 学生端、教师端、知识库管理端分离。
- SSE 流式回答。
- Mermaid 图渲染。
- 文件上传和索引状态展示。

### 13.5 部署升级

可选升级：

- Docker Compose。
- PostgreSQL + pgvector。
- Nginx。
- 用户权限。
- API Key 管理。
- 后端日志和监控。

---

## 14. VSCode 协作开发步骤

下面是后续和 Codex 协作开发时可以直接使用的命令模板。

### 第一步：生成项目骨架

给 Codex 的命令：

```text
第一步：根据我们确定的本地可演示版架构，创建课程智能体项目骨架。要求包含 backend、frontend、knowledge_base、docs、tests、requirements.txt 和 README，不要先实现复杂功能。
```

目标：

- 创建项目目录。
- 创建基础模块。
- 创建 README。
- 创建示例知识库文件。

### 第二步：配置 Python 环境

给 Codex 的命令：

```text
第二步：检查并配置 Python 环境，生成 requirements.txt，创建 .env.example，并告诉我在 VSCode 终端中如何安装依赖。
```

预期命令：

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 第三步：实现本地知识库和 RAG

给 Codex 的命令：

```text
第三步：实现本地知识库加载、文档切片、TF-IDF 检索、引用生成和拒答逻辑。要求返回 answer、citations、confidence、abstained、module。
```

验收：

```text
输入：5G NAS 认证流程有哪些关键风险点？
输出：有回答、有引用、有置信度。
```

### 第四步：实现 FastAPI 后端

给 Codex 的命令：

```text
第四步：实现 FastAPI 后端接口，包括 /api/chat、/api/practice/guide、/api/knowledge/upload、/api/teacher/analytics、/api/teacher/faq、/api/tools、/api/health。
```

启动命令：

```powershell
uvicorn backend.main:app --reload --port 8000
```

### 第五步：实现 SQLite 学习日志

给 Codex 的命令：

```text
第五步：实现 SQLite 学习日志，记录每次提问、命中模块、置信度、是否拒答和引用来源，并让教师端接口可以读取统计结果。
```

验收：

- 连续问 5 个问题。
- `/api/teacher/analytics` 能返回统计。
- `/api/teacher/faq` 能返回高频问题。

### 第六步：实现学生端 Streamlit

给 Codex 的命令：

```text
第六步：实现 Streamlit 学生端页面，包括课程介绍、专题导航、智能问答、实践指导和引用来源展示，所有功能通过 FastAPI 调用。
```

启动命令：

```powershell
streamlit run frontend/app.py
```

### 第七步：实现教师端 Streamlit

给 Codex 的命令：

```text
第七步：实现 Streamlit 教师端页面，包括提问统计、模块分布、拒答统计、高频问题、FAQ 导出和最近问题列表。
```

验收：

- 学生端提问后，教师端能看到统计变化。
- FAQ 可以导出。

### 第八步：实现工具注册层

给 Codex 的命令：

```text
第八步：实现工具注册层，注册 course_retriever、paper_search、vuln_lookup、standard_lookup、mermaid_generator，并在 /api/tools 和前端页面展示。
```

验收：

- `/api/tools` 显示工具名称、描述、风险等级、启用状态。
- 问答时能根据意图选择本地工具。

### 第九步：联调启动

给 Codex 的命令：

```text
第九步：联调后端和前端，启动项目并按 8 个验收问题测试，修复启动、接口、页面和日志统计问题。
```

验收问题：

```text
这个课程智能体能帮我做什么？
5G NAS 认证流程有哪些关键风险点？
我想做基于区块链的隐蔽通信，怎么设计实验？
帮我找和 LLM 漏洞检测相关的研究方向。
有没有可以作为漏洞检测样例的 CVE？
3GPP TS 24.501 里 NAS 安全相关内容在哪里？
请总结一篇知识库不存在的论文结论。
连续提问后，教师端是否能看到高频问题和模块统计？
```

### 第十步：生成报告材料

给 Codex 的命令：

```text
第十步：生成项目 README、系统架构图、Mermaid 流程图、答辩演示脚本、创新点总结和测试记录，要求能直接用于课程实践报告。
```

报告材料包括：

- 项目背景。
- 参考项目说明。
- 系统架构。
- 核心模块。
- RAG 流程。
- 工具注册/MCP 思想。
- 教师端学习分析。
- 测试问题与结果。
- 后续升级方向。

---

## 15. 验收问题

第一阶段固定验收问题：

```text
这个课程智能体能帮我做什么？
5G NAS 认证流程有哪些关键风险点？
我想做基于区块链的隐蔽通信，怎么设计实验？
帮我找和 LLM 漏洞检测相关的研究方向。
有没有可以作为漏洞检测样例的 CVE？
3GPP TS 24.501 里 NAS 安全相关内容在哪里？
请总结一篇知识库不存在的论文结论。
连续提问后，教师端是否能看到高频问题和模块统计？
```

---

## 16. 环境工具清单

### 16.1 第一阶段必需工具

```text
VSCode
Python 3.10+
VSCode Python 扩展
PowerShell
Git，可选
```

### 16.2 Python 依赖

```text
fastapi
uvicorn
streamlit
requests
pydantic
scikit-learn
jieba
pandas
python-multipart
```

### 16.3 第二阶段可选工具

```text
Node.js 20+
Docker Desktop
Chroma 或 FAISS
PostgreSQL + pgvector
MCP Python SDK
DeepSeek API Key
```

### 16.4 VSCode 推荐使用方式

```text
1. 打开项目根目录 course-agent
2. 选择 .venv 作为 Python Interpreter
3. 打开两个终端
4. 终端一启动 FastAPI
5. 终端二启动 Streamlit
6. 浏览器打开 Streamlit 页面测试
```

---

## 17. 第一阶段验收标准

完成后应满足：

- 后端可启动。
- 前端可打开。
- 本地知识库可检索。
- 问答有引用。
- 低证据问题会拒答。
- 实践指导能生成结构化方案。
- SQLite 能记录提问。
- 教师端能显示统计和 FAQ。
- README 说明清楚。
- 能用 8 个验收问题完整演示。

---

## 18. 第二阶段验收标准

后续增强版应满足：

- 标准 MCP Server 可运行。
- 向量数据库可替换本地 TF-IDF。
- DeepSeek 可基于检索证据生成带引用回答。
- DeepSeek 不可用时可回退本地模板回答。
- 真实论文、漏洞、标准数据可接入。
- 轻量 Agent 编排可解释并可查看 trace。
- 支持部署和复现实验。
- 可支撑完整课程实践报告。

---

## 19. 推荐最终报告创新点写法

可以在报告中总结为：

```text
本项目面向网络安全实践课程，设计并实现了一个 MCP-RAG 教学智能体系统。
系统通过本地知识库检索、引用回答、低证据拒答和工具注册机制，降低了大模型在课程问答中的幻觉风险。
同时，系统记录学生学习问题并生成教师端统计与 FAQ，形成了“学生学习 -> 智能问答 -> 教师分析 -> 课程优化”的闭环。
第一阶段实现本地可演示原型，第二阶段可扩展为 DeepSeek、标准 MCP、真实知识库和轻量 Agent 协同的完整平台。
```

---

# 第二阶段详细开发补充报告

## 20. 第一阶段目标状态说明

第一阶段“本地可演示版”完成后应具备以下能力。以下内容是验收目标，不代表项目当前已经实现：

- FastAPI 后端可以启动。
- Streamlit 前端可以打开。
- 本地 Markdown / TXT 知识库可以加载、切片和检索。
- 学生端支持智能问答、专题导航、实践指导和引用来源展示。
- 教师端支持总提问数、拒答次数、模块分布、最近问题、FAQ 和 CSV 导出。
- SQLite 已记录学习日志，包括问题、模块、意图、置信度、是否拒答和引用来源。
- 工具注册层可以注册 `course_retriever`、`paper_search`、`vuln_lookup`、`standard_lookup`、`mermaid_generator`。
- 8 个验收问题可以完成演示。

因此，第二阶段不是推倒重写，而是在第一阶段基础上升级为更完整的 MCP-RAG 智能体平台。

第二阶段开发原则：

```text
保留第一阶段可运行闭环
-> 拆分标准 MCP Server
-> 接入真实知识库和向量数据库
-> 引入 DeepSeek 和轻量 Agent 编排
-> 增强前端和知识库管理
-> 增加权限、配置、部署和报告材料
```

---

## 21. 第二阶段总体目标

第二阶段目标是把第一阶段的本地原型升级为“完整增强版 MCP-RAG 教学智能体系统”。

### 21.1 核心增强方向

第二阶段需要实现：

- 标准 MCP Server，而不只是类 MCP 工具注册层。
- 向量数据库，用于替换或增强 TF-IDF 检索。
- 真实知识库管理，支持 Markdown、TXT、PDF、Word 等资料解析。
- 真实论文、漏洞、标准资料的本地导入和可选联网检索。
- DeepSeek LLM 接入层，负责意图判断、基于证据生成回答和结构化实践指导。
- 轻量 Agent 编排，包括意图路由、课程导师、实践指导、研究助手、工具调用和教师分析等逻辑角色。
- 更完整的知识库管理界面。
- 更规范的日志、配置、权限和部署方案。
- 可直接用于课程实践报告的系统架构图、流程图、测试记录和创新点材料。

### 21.2 第二阶段不建议一次完成的内容

第二阶段可以分多轮逐步完成，不建议一次性同时做完：

- 不建议一开始就同时升级 React、Docker、pgvector、标准 MCP 和复杂多 Agent。
- 不建议一开始就接入多个真实外部 API。
- 不建议一开始就做复杂登录权限。
- 不建议破坏第一阶段 Streamlit + FastAPI 的可演示能力。

推荐策略：

```text
先增加统一 LLM 接口和 DeepSeek Provider
再把 MCP Server 和本地工具协议标准化
再把检索层升级为向量数据库
再增强知识库管理和文件解析
再引入轻量 Agent 编排
最后考虑 React 前端、Docker 和部署
```

---

## 22. 第二阶段推荐技术架构

### 22.1 总体架构

```text
用户
  -> Streamlit / React 前端
  -> FastAPI 应用服务
  -> 会话与权限层
  -> Agent Orchestrator
       -> 加载 AgentState
       -> DeepSeek Planner 生成 ActionDecision
       -> Action Guard / Tool Policy 校验动作
       -> 执行 retrieve / call_tool / ask_user / answer / abstain
       -> 将执行结果写入 Observation
       -> 更新 AgentState 并继续下一轮决策
       -> 达到完成条件或最大三轮后结束
  -> Hybrid Retriever / MCP Client Gateway
       -> course_retriever
       -> paper_search
       -> vuln_lookup
       -> standard_lookup
       -> mermaid_generator
  -> Knowledge Layer（Markdown / TXT / PDF / Word / Chroma / SQLite）
  -> Evidence Bundle 与 Observation
  -> Citation Guard / Safety Guard
  -> 答案、引用、run_id、trace_id、决策和工具调用轨迹
  -> Teacher Analytics
```

### 22.2 DeepSeek LLM 接入层

第二阶段正式使用 DeepSeek API 作为大模型。DeepSeek 是回答生成和轻量智能路由的核心，但不直接读取文件、数据库或执行工具；所有外部能力必须经过 Agent Orchestrator 和 MCP 工具策略层。

DeepSeek 官方 API 文档：[https://api-docs.deepseek.com/](https://api-docs.deepseek.com/)。本项目使用其 OpenAI 兼容接口，默认 `base_url` 为 `https://api.deepseek.com`。模型名始终从环境变量读取，后续若官方模型名称变化，只修改 `.env`，不修改业务代码。

DeepSeek 在系统中承担两个职责：

- Planner：根据目标、历史步骤、可用工具和已有证据决定下一动作。
- Answer Composer：证据充分时生成最终回答或结构化实践方案。

推荐调用链：

```text
学生目标
-> 创建 AgentRun 和 AgentState
-> DeepSeek 输出结构化 ActionDecision
-> 后端校验并执行动作
-> 形成 Observation 和 Evidence Bundle
-> DeepSeek 根据观察结果继续决策
-> 回答 / 追问 / 拒答 / 达到循环上限
-> Citation Guard 检查最终回答与引用
-> 保存每一步决策、工具调用和观察结果
-> 返回前端
```

第一阶段和第二阶段应共用统一接口：

```text
LLMProvider
├── TemplateProvider    第一阶段离线模板生成和故障降级
└── DeepSeekProvider    第二阶段真实大模型生成
```

DeepSeek 不可用、API Key 未配置、请求超时或超过重试次数时，系统自动回退到 `TemplateProvider`，返回检索片段和引用，不应导致整个问答接口失败。

DeepSeek Provider 需要支持：

- 从环境变量读取 API Key，禁止把密钥写入代码、数据库或日志。
- 普通文本回答和 JSON 结构化回答。
- 请求超时、有限重试和错误分类。
- 温度、最大输出长度和证据片段数量配置。
- 记录模型名、耗时、Token 用量、成功状态和错误类型。
- 禁止在没有证据时编造论文、CVE 或标准条款。
- 提供 `decide_next_action(state, available_tools)` 结构化决策接口。
- 决策只允许 `retrieve`、`call_tool`、`ask_user`、`answer`、`abstain` 五种动作。
- 模型只能提出动作，不能直接执行工具；所有动作由后端策略层校验和执行。

Agent 循环配置：

```env
AGENT_MAX_STEPS=3
AGENT_MAX_TOOL_CALLS=3
AGENT_TIMEOUT_SECONDS=120
AGENT_ALLOW_ASK_USER=true
```

### 22.3 第一阶段到第二阶段的模块演进

| 第一阶段模块 | 第二阶段升级方向 |
|---|---|
| `backend/tools/registry.py` | 标准 MCP Server 工具定义 |
| `backend/rag/retriever.py` | Hybrid Retriever：TF-IDF + 向量检索 |
| `backend/rag/answerer.py` | `LLMProvider` + `DeepSeekProvider` + 模板降级 |
| `knowledge_base/*.md` | 支持 Markdown / TXT / PDF / DOCX 文件库 |
| SQLite 学习日志 | 保留 SQLite，后续可升级 PostgreSQL |
| Streamlit 学生端 | 增加知识库管理页、工具调用轨迹页 |
| Streamlit 教师端 | 增加学习卡点、FAQ 聚类、导出报告 |
| 规则路由 | DeepSeek 驱动的有限自主 Agent Loop，暂不强制引入 LangGraph |

---

## 23. 第二阶段目录结构建议

在当前项目基础上扩展，建议目录如下：

```text
course-agent/
├── backend/
│   ├── main.py
│   ├── schemas.py
│   ├── agent/
│   │   ├── router.py
│   │   ├── schemas.py
│   │   ├── state.py
│   │   ├── planner.py
│   │   ├── orchestrator.py
│   │   ├── course_tutor.py
│   │   ├── practice_guide.py
│   │   ├── research_assistant.py
│   │   └── teacher_analytics.py
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── factory.py
│   │   ├── deepseek_provider.py
│   │   ├── template_provider.py
│   │   ├── prompts.py
│   │   └── schemas.py
│   ├── rag/
│   │   ├── loader.py
│   │   ├── retriever.py
│   │   ├── vector_store.py
│   │   ├── hybrid_retriever.py
│   │   ├── document_parser.py
│   │   └── answerer.py
│   ├── tools/
│   │   ├── registry.py
│   │   ├── mcp_client.py
│   │   └── tool_policy.py
│   ├── guards/
│   │   ├── action_guard.py
│   │   ├── citation_guard.py
│   │   └── safety_guard.py
│   ├── core/
│   │   ├── config.py
│   │   ├── errors.py
│   │   └── logging.py
│   ├── storage/
│   │   ├── logger.py
│   │   ├── knowledge_index.py
│   │   └── migrations.py
│   └── data/
│       ├── app.db
│       └── vector_store/
├── mcp_server/
│   ├── __init__.py
│   ├── server.py
│   ├── tools.py
│   ├── schemas.py
│   └── README.md
├── frontend/
│   ├── app.py
│   ├── api_client.py
│   └── pages/
│       ├── student.py
│       ├── teacher.py
│       ├── knowledge.py
│       └── tools.py
├── knowledge_base/
├── datasets/
│   ├── papers/
│   ├── cve/
│   └── standards/
├── docs/
│   ├── architecture.md
│   ├── operation_guide.md
│   ├── phase2_design.md
│   ├── mcp_design.md
│   ├── agent_design.md
│   └── report_outline.md
├── tests/
│   ├── test_demo_questions.py
│   ├── test_api_endpoints.py
│   ├── test_mcp_tools.py
│   ├── test_mcp_client.py
│   ├── test_observation.py
│   ├── test_vector_retriever.py
│   ├── test_llm_provider.py
│   ├── test_agent_decision.py
│   ├── test_action_guard.py
│   ├── test_citation_guard.py
│   ├── test_agent_orchestration.py
│   └── test_agent_resume.py
├── requirements.txt
├── requirements-phase2.txt
├── .env.example
└── README.md
```

---

## 24. 第二阶段环境工具清单

### 24.1 必需工具

```text
Python 3.10+
VSCode
PowerShell
Git
FastAPI
Streamlit
SQLite
pytest
```

### 24.2 第二阶段新增 Python 依赖

建议新增 `requirements-phase2.txt`：

```text
mcp
chromadb
sentence-transformers
langchain-text-splitters
openai
pypdf
python-docx
beautifulsoup4
lxml
httpx
tenacity
python-dotenv
networkx
```

说明：

| 依赖 | 用途 |
|---|---|
| `mcp` | 标准 MCP Server / Client |
| `chromadb` | 本地向量数据库 |
| `sentence-transformers` | 本地 embedding 模型 |
| `langchain-text-splitters` | 文档切片 |
| `openai` | 使用 OpenAI 兼容接口调用 DeepSeek API |
| `pypdf` | PDF 解析 |
| `python-docx` | Word 文档解析 |
| `beautifulsoup4` | HTML / 网页片段解析 |
| `httpx` | 异步 HTTP 请求 |
| `tenacity` | 外部接口重试 |
| `python-dotenv` | `.env` 配置读取 |
| `networkx` | 可选：知识点关系图 |

### 24.3 可选工具

```text
Node.js 20+
Docker Desktop
PostgreSQL
pgvector
React + Vite
Tailwind CSS
```

第一轮第二阶段不强制安装 Docker、Node、PostgreSQL。建议先完成 Python 版增强，再考虑前端和部署升级。

### 24.4 外部 API Key 配置

```text
DEEPSEEK_API_KEY=
NVD_API_KEY=
SEMANTIC_SCHOLAR_API_KEY=
```

`DEEPSEEK_API_KEY` 必须保持为空白占位，最后由项目使用者在本机 `.env` 中填写。`.env` 必须加入 `.gitignore`，仓库只提交 `.env.example`。NVD 和 Semantic Scholar 仍是后续联网工具的可选配置。

---

## 25. 第二阶段配置文件设计

建议更新 `.env.example`：

```text
APP_NAME=course-agent
API_HOST=127.0.0.1
API_PORT=8000
STREAMLIT_SERVER_PORT=8501

KNOWLEDGE_BASE_DIR=knowledge_base
SQLITE_DB_PATH=backend/data/app.db

VECTOR_STORE_PROVIDER=chroma
VECTOR_STORE_DIR=backend/data/vector_store
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-flash
LLM_TEMPERATURE=0.2
LLM_MAX_OUTPUT_TOKENS=2000
LLM_TIMEOUT_SECONDS=60
LLM_MAX_RETRIES=2
LLM_MAX_CONTEXT_CHUNKS=6
LLM_FALLBACK_PROVIDER=template

AGENT_MAX_STEPS=3
AGENT_MAX_TOOL_CALLS=3
AGENT_TIMEOUT_SECONDS=120
AGENT_ALLOW_ASK_USER=true

MCP_SERVER_COMMAND=python
MCP_SERVER_ARGS=mcp_server/server.py

ENABLE_REAL_PAPER_SEARCH=false
ENABLE_REAL_CVE_LOOKUP=false
ENABLE_REAL_STANDARD_FETCH=false

NVD_API_KEY=
SEMANTIC_SCHOLAR_API_KEY=
```

说明：

- `DEEPSEEK_MODEL` 必须从环境变量读取，便于模型名称变化时直接修改配置。
- 当前设计使用 DeepSeek 官方 OpenAI 兼容 API，不接入其他大模型 Provider。
- `DEEPSEEK_API_KEY` 只在本机 `.env` 中填写，日志和前端配置页只能显示“已配置/未配置”。
- 没有 API Key 时系统仍可使用 `TemplateProvider` 完成本地检索演示。

---

## 26. 第二阶段数据库设计

第一阶段已有：

```text
questions
retrieval_logs
faq_items
```

第二阶段建议新增：

### 26.1 documents 表

```text
id
filename
file_type
source_path
source_kind
title
module
version
content_hash
indexed
created_at
updated_at
```

### 26.2 document_chunks 表

```text
id
document_id
chunk_index
title
text
metadata_json
vector_id
created_at
```

### 26.3 tool_calls 表

```text
id
question_id
run_id
step_id
tool_name
arguments_json
result_summary
risk
success
created_at
```

### 26.4 agent_runs 表

记录一次从学生目标到完成、追问、拒答或失败的完整智能体运行。

```text
id
conversation_id
question_id
status
current_step
max_steps
tool_call_count
requires_user_input
pending_question
stop_reason
started_at
finished_at
```

`status` 统一使用：

```text
RUNNING
REQUIRES_INPUT
COMPLETED
ABSTAINED
STEP_LIMIT_REACHED
FAILED
```

### 26.5 agent_steps 表

```text
id
run_id
step_number
action
tool_name
arguments_json
reason
observation_summary
success
created_at
```

### 26.6 agent_traces 表

```text
id
question_id
run_id
agent_name
step_name
input_summary
output_summary
created_at
```

### 26.7 knowledge_sources 表

```text
id
name
source_type
description
enabled
created_at
```

### 26.8 conversations 表

```text
id
user_id
title
created_at
updated_at
```

### 26.9 messages 表

```text
id
conversation_id
role
content
question_id
created_at
```

### 26.10 model_calls 表

```text
id
question_id
run_id
step_id
provider
model
prompt_tokens
completion_tokens
latency_ms
success
error_type
created_at
```

不得在 `model_calls` 中保存 API Key。是否保存完整提示词应由配置控制，默认只保存摘要和统计信息。

### 26.11 index_jobs 表

```text
id
document_id
status
progress
error_message
started_at
finished_at
```

索引状态统一使用：

```text
PENDING
PARSING
CHUNKING
EMBEDDING
INDEXED
FAILED
DELETING
```

### 26.12 feedback 表

```text
id
question_id
user_id
rating
comment
created_at
```

数据库迁移必须使用版本化 migration，不允许每次启动时无条件删除并重建数据表。SQLite 本地运行建议启用 WAL 模式；知识库索引时应避免多个任务同时修改同一文档。

---

## 27. 标准 MCP Server 设计

### 27.1 MCP Server 目标

第一阶段是“类 MCP 工具注册层”，第二阶段应升级为标准 MCP Server。

目标：

- 每个工具以标准 MCP tool 暴露。
- FastAPI 后端通过 MCP client 调用工具。
- 工具有统一的名称、描述、输入 schema、输出 schema 和风险等级。
- 高风险工具默认关闭。

### 27.2 MCP 工具列表

```text
course_retriever
paper_search
vuln_lookup
standard_lookup
mermaid_generator
document_ingestor
knowledge_reindexer
```

### 27.3 工具输入输出设计

#### course_retriever

输入：

```json
{
  "query": "5G NAS 认证流程有哪些关键风险点？",
  "top_k": 5,
  "module": "5G NAS"
}
```

输出：

```json
{
  "results": [
    {
      "source": "5g_nas.md",
      "title": "5G NAS 安全专题",
      "snippet": "...",
      "score": 0.82
    }
  ]
}
```

#### paper_search

第一轮只检索本地论文样例。第二轮可接入 Semantic Scholar / arXiv。

输入：

```json
{
  "query": "LLM 漏洞检测 研究方向",
  "top_k": 5,
  "source": "local"
}
```

#### vuln_lookup

第一轮只检索本地 CVE 样例。第二轮可接入 NVD API。

输入：

```json
{
  "query": "CVE 漏洞检测样例",
  "top_k": 5,
  "allow_remote": false
}
```

#### standard_lookup

第一轮只检索本地标准样例。第二轮可导入真实 3GPP 标准 PDF。

输入：

```json
{
  "query": "3GPP TS 24.501 NAS security",
  "top_k": 5
}
```

#### mermaid_generator

输入：

```json
{
  "diagram_type": "flowchart",
  "topic": "MCP-RAG 系统架构"
}
```

输出：

```json
{
  "mermaid": "flowchart LR ..."
}
```

### 27.4 MCP 工具治理与错误约定

每个 MCP 工具除名称和输入输出 schema 外，还必须声明：

```text
risk_level
timeout_seconds
max_retries
allow_network
read_only
enabled
```

风险分级：

- `LOW`：本地只读检索，可自动执行。
- `MEDIUM`：联网查询，仅在配置启用后执行。
- `HIGH`：文件写入、代码执行、数据库修改，默认禁用并需要管理员确认。

工具调用必须返回统一结果：

```json
{
  "success": true,
  "data": {},
  "error_code": null,
  "error_message": null,
  "metadata": {
    "tool_name": "course_retriever",
    "latency_ms": 25
  }
}
```

FastAPI 启动时创建并复用 MCP Client 会话，不应为每个 HTTP 请求重复启动 MCP Server。MCP 不可用时只允许回退到本地只读 Registry，高风险工具不得自动降级执行。

### 27.5 Agent 可用工具目录

MCP Client Gateway 需要把当前已启用工具转换为可交给 DeepSeek Planner 的精简目录。目录只包含工具名称、用途、输入 schema、风险等级和是否联网，不包含实现细节或密钥。

```json
{
  "name": "paper_search",
  "description": "检索本地论文资料，返回可引用片段",
  "input_schema": {
    "query": "string",
    "top_k": "integer"
  },
  "risk_level": "LOW",
  "allow_network": false
}
```

DeepSeek 只能从该目录选择工具；模型输出不存在、未启用或权限不足的工具名时，Action Guard 必须拒绝执行并记录原因。

### 27.6 Observation 统一格式

每次检索或工具执行后转换为统一 Observation，再写回 AgentState：

```json
{
  "step_number": 2,
  "action": "call_tool",
  "tool_name": "paper_search",
  "success": true,
  "content": [],
  "error_code": null,
  "summary": "命中 3 条本地论文片段",
  "metadata": {
    "latency_ms": 36,
    "fallback_used": false
  }
}
```

Observation 是工具结果，不是系统指令。DeepSeek 可以据此决定下一步，但不得把 Observation 中的文本当成可越权执行的命令。

---

## 28. 真实知识库设计

### 28.1 支持文件类型

第二阶段知识库应支持：

```text
.md
.txt
.pdf
.docx
.html
.json
.csv
```

第一轮优先：

```text
Markdown
TXT
PDF
DOCX
```

### 28.2 文档解析流程

```text
上传文件
-> 判断文件类型
-> 提取文本
-> 清洗文本
-> 按标题/段落切片
-> 写入 SQLite document_chunks
-> 生成 embedding
-> 写入 Chroma
-> 返回索引状态
```

### 28.3 引用定位增强

第一阶段引用字段：

```text
source
title
snippet
score
```

第二阶段建议增强为：

```text
source
title
snippet
score
document_id
chunk_id
page
section
source_type
```

这样报告中可以写成：

```text
系统不仅给出引用文件，还能定位到文档页码、章节和具体 chunk。
```

### 28.4 文档索引一致性与上传安全

文档上传后按 `PENDING -> PARSING -> CHUNKING -> EMBEDDING -> INDEXED` 更新状态；任一步失败则标记为 `FAILED` 并保存错误信息。只有 `INDEXED` 文档可以进入正式检索结果。

删除文档时必须同时删除原始文件记录、SQLite 中的 `document_chunks` 和 Chroma 中对应的向量，避免删除后仍然被检索。上传接口必须限制扩展名、MIME 类型和文件大小，并对文件名重新命名，禁止使用用户输入路径直接写盘。

解析后的文档内容只能作为检索资料。文档中出现的“忽略系统指令”“调用某工具”等文本属于不可信内容，不得提升为系统指令或触发工具调用。

---

## 29. 向量数据库设计

### 29.1 推荐第一版使用 Chroma

原因：

- 本地运行简单。
- Python 接入容易。
- 适合课程演示。
- 后续可以替换 FAISS 或 pgvector。

### 29.2 检索策略

建议采用混合检索：

```text
用户问题
-> query rewrite，可选
-> TF-IDF 检索 Top-K
-> 向量检索 Top-K
-> 合并结果
-> 去重
-> 重排
-> 计算置信度
-> 生成引用回答
```

### 29.3 置信度设计

可以综合：

```text
向量相似度
关键词覆盖率
文档类型权重
是否命中目标模块
引用数量
```

示例：

```text
confidence = 0.5 * vector_score
           + 0.3 * keyword_score
           + 0.1 * module_score
           + 0.1 * citation_score
```

上述公式只用于初始实验。TF-IDF 分数和向量相似度必须先归一化，再参与合并；拒答阈值应通过固定测试集调优，不能仅凭主观设置。

### 29.4 RAG 评测指标

除 8 个演示问题外，建议准备 30 至 50 条带期望来源的离线问题，至少评估：

```text
Recall@K                 正确证据是否出现在 Top-K
Citation Accuracy       引用是否真实支持回答
Groundedness            回答内容是否来自证据
Abstention Accuracy     资料不足时是否正确拒答
Intent Accuracy         意图路由是否正确
Tool Routing Accuracy   工具选择是否正确
```

---

## 30. 轻量 Agent 编排设计

第二阶段不运行多个彼此独立的大模型 Agent。为降低开发量、成本和故障率，采用一个 `AgentOrchestrator` 实现受控的有限自主循环。课程导师、实践指导和研究助手是不同任务配置，不是独立进程。

### 30.1 AgentState

每次运行维护统一状态：

```text
run_id
conversation_id
user_goal
intent
current_step
max_steps
tool_call_count
available_tools
messages
evidence
observations
requires_user_input
pending_question
final_answer
stop_reason
```

状态只由后端更新。DeepSeek 接收状态的必要摘要，不得直接修改数据库状态。

### 30.2 ActionDecision

DeepSeek Planner 每一轮必须返回可由 Pydantic 校验的结构化决策：

```json
{
  "action": "call_tool",
  "tool_name": "paper_search",
  "arguments": {
    "query": "区块链隐蔽通信研究方法",
    "top_k": 5
  },
  "reason": "需要论文证据来设计实验方法",
  "task_complete": false
}
```

允许动作只有：

```text
retrieve      使用混合检索获取课程证据
call_tool     从已启用 MCP 工具目录中选择一个工具
ask_user      信息不足，需要学生补充后继续同一个 AgentRun
answer        证据充分，生成最终回答
abstain       无法获得足够证据，明确拒答
```

### 30.3 有限自主 Agent Loop

```python
for step_number in range(1, max_steps + 1):
    decision = planner.decide_next_action(state, available_tools)
    action_guard.validate(decision, state, available_tools)

    if decision.action == "retrieve":
        observation = hybrid_retriever.search(decision.arguments)
    elif decision.action == "call_tool":
        observation = tool_gateway.call(decision.tool_name, decision.arguments)
    elif decision.action == "ask_user":
        return requires_user_input(decision, state)
    elif decision.action == "answer":
        return compose_and_verify_answer(state)
    elif decision.action == "abstain":
        return abstained_response(decision.reason)

    state.add_step(decision, observation)

return stopped_by_step_limit(state)
```

Agent 在每次检索或工具调用后观察结果，并由 DeepSeek 决定继续调用其他工具、请求用户补充、生成答案或拒答。最大默认三轮，避免无限循环和不可控费用。

### 30.4 Action Guard 与停止条件

动作执行前必须检查：

- 动作是否属于允许枚举。
- 工具是否存在、启用且当前角色有权限。
- 参数是否符合工具 schema。
- 是否超过最大步骤、工具调用次数或总超时。
- 是否试图根据知识库提示注入调用高风险工具。
- `answer` 是否已经具备最低数量和最低分数的证据。

停止条件：

```text
COMPLETED             已完成回答并通过引用检查
REQUIRES_INPUT        等待学生补充信息，可继续原 run_id
ABSTAINED             证据不足或请求不受支持
STEP_LIMIT_REACHED    达到最大步骤，返回已有证据和限制说明
FAILED                出现不可恢复错误
```

### 30.5 回答配置与降级顺序

课程问答使用“知识定位、知识讲解、实践提示、引用来源”模板；实践指导使用“研究目标、方法设计、工具选择、实验步骤、结果分析、创新点、风险与注意事项”模板；论文问题在没有原文时不得总结具体结论。

任一步骤失败都必须记录 `run_id`、`trace_id`、步骤和错误类型。降级顺序固定为：

```text
DeepSeek 不可用 -> TemplateProvider 基于已有证据生成摘要
MCP 不可用 -> 本地只读 Registry
Chroma 不可用 -> TF-IDF 检索
证据不足 -> abstain，不调用 DeepSeek 补全事实
```

DeepSeek 生成最终回答前只接收经过裁剪的 Evidence Bundle，并使用 `[S1]`、`[S2]` 引用编号；Citation Guard 验证引用编号存在且片段支持对应结论。

---

## 31. 第二阶段 API 设计

在第一阶段 API 基础上新增：

```text
POST /api/knowledge/reindex
GET  /api/knowledge/documents
GET  /api/knowledge/documents/{document_id}
DELETE /api/knowledge/documents/{document_id}
GET  /api/knowledge/jobs/{job_id}

POST /api/tools/call
GET  /api/tools/calls

GET  /api/agent/traces
GET  /api/agent/traces/{question_id}
GET  /api/agent/runs/{run_id}
POST /api/agent/runs/{run_id}/continue

POST /api/conversations
GET  /api/conversations/{conversation_id}/messages
POST /api/feedback

POST /api/mermaid
GET  /api/system/config
GET  /api/system/health/dependencies
```

### 31.1 `/api/knowledge/reindex`

用途：

重新解析知识库并重建向量索引。

### 31.2 `/api/tools/call`

用途：

手动调试 MCP 工具。

### 31.3 `/api/agent/traces`

用途：

查看 Agent 执行轨迹，便于演示系统可解释性。

### 31.4 `/api/chat` 第二阶段响应契约

第二阶段保留原字段，并增加：

```json
{
  "answer": "...",
  "citations": [],
  "confidence": 0.82,
  "abstained": false,
  "module": "5G NAS",
  "intent": "course_qa",
  "conversation_id": "...",
  "question_id": 12,
  "run_id": "...",
  "trace_id": "...",
  "run_status": "COMPLETED",
  "current_step": 2,
  "requires_user_input": false,
  "pending_question": null,
  "provider": "deepseek",
  "model": "deepseek-flash",
  "fallback_used": false,
  "tool_calls": []
}
```

当 Agent 选择 `ask_user` 时，`/api/chat` 返回 `run_status=REQUIRES_INPUT`、`requires_user_input=true` 和 `pending_question`。学生补充信息后调用 `/api/agent/runs/{run_id}/continue`，后端必须恢复原 AgentState，不重新创建无关任务。

`GET /api/agent/runs/{run_id}` 返回运行状态、当前步骤、决策摘要、工具调用摘要和停止原因，用于前端展示；普通学生只能查看自己的运行记录。

### 31.5 权限和错误响应

- 学生可调用问答、实践指导、会话和反馈接口。
- 教师可查看聚合统计和 FAQ，不默认查看其他学生的完整敏感输入。
- 管理员才能上传、删除、重建知识库和手动调试 MCP 工具。
- 所有错误使用统一 `error_code`、`message`、`trace_id`，不得把 API Key、完整系统提示词或内部堆栈返回前端。

---

## 32. 第二阶段前端页面设计

### 32.1 保留现有页面

```text
学生端
教师端
工具状态
项目说明
```

### 32.2 新增页面

```text
知识库管理
Agent 轨迹
MCP 工具调试
系统配置
```

### 32.3 知识库管理页

功能：

- 上传 PDF / DOCX / MD / TXT。
- 查看文档列表。
- 查看索引状态。
- 手动重建索引。
- 查看文档 chunk。

### 32.4 Agent 轨迹页

功能：

- 查看最近问题。
- 查看路由意图。
- 查看每一步 `ActionDecision` 和 `Observation`。
- 查看调用了哪些工具。
- 查看引用检查结果。
- 查看是否触发拒答。
- 查看停止原因和是否达到步骤上限。

### 32.5 MCP 工具调试页

功能：

- 显示 MCP Server 状态。
- 显示工具列表。
- 选择工具并输入参数。
- 显示工具返回结果。

### 32.6 模型状态与安全显示

系统配置页只显示：

- 当前 Provider：`deepseek` 或 `template`。
- 当前模型名称。
- API Key：仅显示“已配置/未配置”，绝不返回密钥内容。
- DeepSeek、MCP、Chroma 和 SQLite 的健康状态。
- 最近一次模型调用是否使用 fallback。

知识库管理、MCP 工具调试和完整 Agent 轨迹页面应限制为管理员或教师角色。第一轮本地演示如果暂不实现登录，必须通过配置开关隐藏管理页面，并在报告中说明权限功能属于第二阶段收尾项。

### 32.7 学生补充信息与继续执行

当后端返回 `REQUIRES_INPUT` 时，学生端显示 Agent 提出的补充问题和输入框。提交后调用 `/api/agent/runs/{run_id}/continue`，继续原任务并保留此前证据、工具结果和步骤轨迹。

前端需要区分以下状态：

```text
正在规划
正在检索
正在调用工具
等待学生补充
正在生成答案
已完成
已拒答
达到步骤上限
执行失败
```

第一轮可以在请求结束后一次性展示轨迹，不强制实现 WebSocket 或 SSE 实时推送。

---

## 33. 第二阶段开发步骤

下面的步骤用于在 VSCode 中逐步与 Codex 协作。每一步都必须先检查现有代码，保留用户已经完成的功能和未提交修改；完成实现后运行本步骤测试和已有回归测试，不得只创建空文件或只写接口说明。

当前进度：根据项目使用者的说明，第二阶段第一步至第五步已经执行并完成审查复核。继续开发时从第六步开始；除非后续集成测试发现明确缺陷，否则不要重做前五步。

每次发布开发命令时，建议在结尾统一附加：

```text
先阅读现有代码和测试，不要覆盖已经完成的功能。实现后运行相关测试与完整回归测试；若测试失败，定位并修复后再结束。最后说明修改了哪些文件、如何启动、测试结果和仍存在的限制。不要在代码、日志或提交中写入真实 API Key。
```

### 第一步：建立第二阶段依赖和配置基线（已执行并审查复核）

给 Codex 的命令：

```text
第一步：检查第一阶段项目的目录、依赖和测试基线。新增 requirements-phase2.txt，在 .env.example 中加入向量数据库、embedding、MCP 和外部数据源的配置占位，确认 .env 已加入 .gitignore，并更新 README 的第二阶段说明。不得升级或删除第一阶段仍在使用的依赖。运行原有测试并记录基线结果。
```

完成判定：

```text
requirements-phase2.txt 可以安装
.env.example 不包含真实密钥
.env 已被 Git 忽略
第一阶段原有测试全部通过
```

### 第二步：实现统一文档解析层（已执行并审查复核）

给 Codex 的命令：

```text
第二步：实现 backend/rag/document_parser.py，定义统一 Document/Page 数据结构，支持 Markdown、TXT、PDF、DOCX 文本提取，保留 filename、page、section、source_type 等元数据。限制不支持的扩展名和最大文件大小；解析失败时返回明确的领域错误，不允许静默返回空文档。为每种格式增加最小测试样例。
```

完成判定：

```text
四种格式均可提取文本和来源元数据
空文件、损坏文件和不支持格式有明确错误
用户文件名不能造成路径穿越
解析层单元测试通过
```

### 第三步：实现文档元数据、切片和索引状态（已执行并审查复核）

给 Codex 的命令：

```text
第三步：通过版本化 migration 扩展 SQLite，新增 documents、document_chunks 和 index_jobs 表。实现内容 hash 去重、文档与 chunk 外键关系、PENDING/PARSING/CHUNKING/EMBEDDING/INDEXED/FAILED 状态流转，以及失败原因记录。不要通过删除数据库重建表。增加上传、重复上传、失败索引和级联删除测试。
```

完成判定：

```text
上传文档后 documents 和 index_jobs 有记录
切片后 document_chunks 保留页码和章节
重复内容可通过 hash 识别
失败任务状态为 FAILED 且有错误信息
数据库升级不破坏第一阶段日志
```

### 第四步：接入 Chroma 向量数据库（已执行并审查复核）

给 Codex 的命令：

```text
第四步：实现 backend/rag/vector_store.py，使用 Chroma 和本地 sentence-transformers embedding 模型，对 document_chunks 建立持久化向量索引。定义 add、query、delete_by_document、reset_collection 接口，保证 SQLite chunk_id 与 Chroma vector_id 一致。模型和存储目录从配置读取，增加临时目录下的隔离测试。
```

完成判定：

```text
可以创建和重新打开持久化 Chroma 索引
可以批量插入并查询相似 chunk
删除文档时对应向量同步删除
重复索引不会产生重复向量
向量检索测试不依赖真实 DeepSeek API
```

### 第五步：实现可评测的混合检索（已执行并审查复核）

给 Codex 的命令：

```text
第五步：检查已有 TF-IDF 与 Chroma 实现，新增或完善 backend/rag/hybrid_retriever.py。分别获取关键词 Top-K 和向量 Top-K，对两类分数归一化，按 chunk_id 去重后使用可配置权重融合并重排。统一返回 RetrievalResult，字段至少包含 document_id、chunk_id、source、title、snippet、page、section、source_type、keyword_score、vector_score、final_score。实现拒答阈值配置，但本步骤不调用大模型。增加 tests/test_hybrid_retriever.py，并建立一组带期望来源的小型检索评测数据。
```

实施约束：

- 不直接把未归一化的 TF-IDF 和向量相似度相加。
- 检索结果只允许来自 `INDEXED` 文档。
- Chroma 不可用时回退到 TF-IDF，并在结果 metadata 标记 fallback。
- 保留第一阶段检索接口，避免现有 API 失效。

测试与完成判定：

```powershell
pytest tests/test_hybrid_retriever.py -q
pytest -q
```

```text
原有 8 个演示问题继续通过
PDF/DOCX 新知识能够被检索
引用结果包含完整定位字段
无关问题低于阈值并触发拒答
```

### 第六步：新增 DeepSeek LLM 接入层

给 Codex 的命令：

```text
第六步：为项目实现统一 LLM Provider 和 Agent 决策接口，并把 DeepSeek 作为第二阶段唯一真实大模型。新增 backend/llm/base.py、factory.py、deepseek_provider.py、template_provider.py、prompts.py、schemas.py，新增 backend/agent/schemas.py，同时新增 backend/core/config.py 和 backend/core/errors.py。使用 openai Python SDK 的 OpenAI 兼容客户端调用 DeepSeek。除普通回答和 Pydantic 结构化回答外，实现 decide_next_action(state_summary, available_tools) -> ActionDecision。ActionDecision 只允许 retrieve、call_tool、ask_user、answer、abstain，包含 tool_name、arguments、reason、task_complete。DeepSeek 只能提出动作，不能直接执行工具。base_url、model、temperature、timeout、retry、最大输出长度、AGENT_MAX_STEPS、AGENT_MAX_TOOL_CALLS 和 AGENT_TIMEOUT_SECONDS 全部从环境变量读取。DEEPSEEK_API_KEY 在 .env.example 中必须为空。TemplateProvider 负责未配置密钥或 DeepSeek 不可用时的降级。使用 mock client 编写 tests/test_llm_provider.py 和 tests/test_agent_decision.py，测试不得产生真实 API 费用。
```

必须创建或修改：

```text
backend/llm/__init__.py
backend/llm/base.py
backend/llm/factory.py
backend/llm/deepseek_provider.py
backend/llm/template_provider.py
backend/llm/prompts.py
backend/llm/schemas.py
backend/agent/schemas.py
backend/core/config.py
backend/core/errors.py
requirements-phase2.txt
.env.example
.gitignore
tests/test_llm_provider.py
tests/test_agent_decision.py
```

配置要求：

```env
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-flash
LLM_TEMPERATURE=0.2
LLM_MAX_OUTPUT_TOKENS=2000
LLM_TIMEOUT_SECONDS=60
LLM_MAX_RETRIES=2
LLM_FALLBACK_PROVIDER=template
AGENT_MAX_STEPS=3
AGENT_MAX_TOOL_CALLS=3
AGENT_TIMEOUT_SECONDS=120
AGENT_ALLOW_ASK_USER=true
```

测试与完成判定：

```powershell
pytest tests/test_llm_provider.py -q
pytest -q
```

```text
未配置 DEEPSEEK_API_KEY 时自动使用 TemplateProvider
配置密钥时 Factory 返回 DeepSeekProvider
模拟成功、超时、限流、服务异常和非法 JSON 均有测试
日志、异常和 API 返回中不出现密钥
结构化实践指导可以通过 Pydantic 校验
五种 ActionDecision 均能通过结构化解析
不存在的 action、缺少工具名或非法参数会被 schema 拒绝
DeepSeek 只返回决策，不直接执行检索或 MCP 工具
```

### 第七步：实现标准 MCP Server 和工具策略

给 Codex 的命令：

```text
第七步：新增或完善 mcp_server/server.py、tools.py、schemas.py，以及 backend/tools/tool_policy.py。把 course_retriever、paper_search、vuln_lookup、standard_lookup、mermaid_generator 暴露为标准 MCP tools。每个工具定义严格输入输出 schema、面向 Agent 的用途描述、风险等级、超时、是否联网、是否只读和启用开关。实现 get_agent_tool_catalog()，只向 DeepSeek Planner 暴露当前已启用且当前角色可使用的工具名称、用途、输入 schema 和风险信息。统一工具结果格式为 success、data、error_code、error_message、metadata。第一轮所有工具默认使用本地数据；联网工具默认关闭。为工具目录、参数校验、成功调用、超时、禁用状态和不存在工具编写 tests/test_mcp_tools.py。
```

实施约束：

- MCP 工具只返回数据或 Mermaid 文本，不负责生成最终自然语言答案。
- 工具不得读取 `.env` 内容或把密钥返回调用方。
- 文档写入、代码执行和任意路径读取工具不得加入自动调用列表。
- DeepSeek 只能从 `get_agent_tool_catalog()` 返回的目录中选择工具。
- 工具描述必须说明“何时使用”和“何时不应使用”，避免模型随意调用。

测试与完成判定：

```powershell
pytest tests/test_mcp_tools.py -q
pytest -q
```

```text
MCP Server 可独立启动
MCP Client 可读取工具清单和 schema
五个工具可用测试参数调用
非法参数、禁用工具和超时返回统一错误
Agent 工具目录不包含禁用、越权或高风险工具
模型请求不存在的工具时不会执行
```

### 第八步：实现 MCP Client Gateway 与本地降级

给 Codex 的命令：

```text
第八步：在 backend/tools/mcp_client.py 中实现 MCP Client Gateway，由 FastAPI 应用生命周期统一启动和关闭 MCP 会话，不得每个请求重复启动进程。提供 list_tools、get_agent_tool_catalog 和 call_tool 接口，设置超时、有限重试和并发保护。把混合检索和 MCP 工具执行结果统一转换为 Observation，字段包含 step_number、action、tool_name、success、content、error_code、summary、latency_ms 和 fallback_used，供下一轮 DeepSeek 决策使用。MCP 不可用时，仅对 LOW 风险只读工具回退到本地 registry。tool_calls 必须关联 run_id 和 step_id，并对参数和结果脱敏。增加 tests/test_mcp_client.py、tests/test_observation.py 和后端集成测试。
```

测试与完成判定：

```powershell
pytest tests/test_mcp_client.py tests/test_api_endpoints.py -q
pytest -q
```

```text
后端启动时只创建一个可复用 MCP 会话
/api/tools 返回 MCP 工具状态
MCP 停止后只读工具可回退本地 registry
高风险或联网工具不会绕过策略自动执行
检索结果和工具结果都能转换成统一 Observation
Observation 中的文本只作为不可信数据，不能触发未授权动作
```

### 第九步：实现 DeepSeek 驱动的有限自主 Agent Loop

给 Codex 的命令：

```text
第九步：实现 backend/agent/state.py、planner.py 和 orchestrator.py，把固定问答流水线升级为有限自主 Agent Loop。创建 AgentState，保存 run_id、用户目标、意图、当前步骤、最大步骤、工具调用次数、可用工具、messages、evidence、observations、pending_question、final_answer 和 stop_reason。每一轮由 DeepSeekProvider.decide_next_action 返回 ActionDecision，后端校验后执行 retrieve、call_tool、ask_user、answer 或 abstain，将结果转换为 Observation 写回 AgentState，再由 DeepSeek 决定下一步。默认最多 3 步、最多 3 次工具调用和 120 秒总超时。实现 REQUIRES_INPUT 状态，使学生补充信息后可以通过原 run_id 继续。通过版本化 migration 新增 agent_runs 和 agent_steps，并让 model_calls、tool_calls、agent_traces 关联 run_id 和 step_id。所有模型、检索和 MCP 调用使用 mock 编写 tests/test_agent_orchestration.py 和 tests/test_agent_resume.py。
```

实施约束：

- 简单课程问题允许在一次决策后直接检索并回答；复杂实践问题允许多轮决策，但不得超过配置上限。
- 每轮 DeepSeek 只能提出一个动作，工具必须由后端执行。
- 证据不足时直接拒答，不要求模型补充事实。
- DeepSeek 失败时使用 TemplateProvider 组织已检索证据。
- `ask_user` 必须暂停并持久化状态，继续执行时不得丢失已有证据和观察结果。
- 每一步必须可追踪，错误必须带 `run_id` 和 `trace_id`。
- 达到步骤上限时返回已有证据和限制说明，不允许悄悄继续调用模型。

测试与完成判定：

```powershell
pytest tests/test_agent_orchestration.py -q
pytest -q
```

```text
简单问题可以一次检索后回答，复杂问题可以连续选择不同工具
信息不足时返回 REQUIRES_INPUT，补充信息后继续原 AgentRun
每次问答生成完整的决策、Observation 和 Agent trace
模型调用和工具调用可关联到 question_id、run_id 和 step_id
DeepSeek、MCP 或 Chroma 任一不可用时按设计降级
达到三步或工具调用上限时强制停止
```

### 第十步：实现 Action Guard、Citation Guard、Safety Guard 和 Agent API

给 Codex 的命令：

```text
第十步：新增 backend/guards/action_guard.py、citation_guard.py 和 safety_guard.py。Action Guard 在动作执行前校验 action 枚举、工具是否存在和启用、参数 schema、角色权限、步骤上限、工具调用上限、总超时和高风险操作；DeepSeek 请求不存在或未授权工具时拒绝执行。Citation Guard 要求最终回答使用 [S1]、[S2] 引用，校验引用存在、来源可定位且关键事实受证据支持。Safety Guard 把上传文档、Observation 和工具结果视为不可信内容，阻止提示注入影响系统规则。升级 /api/chat 响应，增加 run_id、run_status、current_step、requires_user_input、pending_question、conversation_id、question_id、trace_id、provider、model、fallback_used 和 tool_calls；新增 GET /api/agent/runs/{run_id} 与 POST /api/agent/runs/{run_id}/continue。统一 API 错误格式。增加 tests/test_action_guard.py、引用、提示注入、暂停恢复和 API 契约测试。
```

测试与完成判定：

```powershell
pytest tests/test_action_guard.py tests/test_citation_guard.py tests/test_api_endpoints.py -q
pytest -q
```

```text
回答引用可以定位到 document_id、chunk_id、页码或章节
伪造引用会被拦截或触发拒答
知识库中的提示注入文本不能改变系统规则
前端所需响应字段稳定且有 schema 测试
非法动作、越权工具和超过上限的调用会被 Action Guard 拦截
REQUIRES_INPUT 可以通过原 run_id 恢复执行
```

### 第十一步：增强知识库、模型状态和轨迹前端

给 Codex 的命令：

```text
第十一步：在 Streamlit 中增强知识库管理、模型状态和 Agent 轨迹页面。支持上传 PDF/DOCX/MD/TXT、查看 documents 和 index_jobs 状态、失败原因、手动重建索引和删除文档；显示 DeepSeek 是否已配置、当前模型、MCP/Chroma/SQLite 健康状态和 fallback 状态，但绝不显示 API Key。学生问答页面展示 Agent 当前状态、每一步 ActionDecision、Observation、引用定位、工具调用摘要、run_id 和 trace_id。当后端返回 REQUIRES_INPUT 时显示补充问题输入框，提交后调用 /api/agent/runs/{run_id}/continue，继续原任务。管理页面通过角色或配置开关限制访问。前端只能调用 FastAPI，不得直接访问 SQLite、Chroma、DeepSeek 或 MCP。第一轮采用请求完成后一次性展示轨迹，不强制实现 SSE。
```

完成判定：

```text
上传后可看到完整索引进度和失败信息
删除文档后 SQLite 与 Chroma 均无残留
DeepSeek 未配置时页面清楚显示本地降级模式
问答页面可展开查看引用和执行轨迹
学生可以补充信息并继续同一个 AgentRun
前端能区分完成、等待输入、拒答、步骤上限和失败状态
```

### 第十二步：增强教师端分析与反馈闭环

给 Codex 的命令：

```text
第十二步：扩展教师端和统计 API，增加 DeepSeek 调用成功率与平均耗时、fallback 次数、平均 Agent 步数、平均工具调用次数、最常选择的动作与工具、REQUIRES_INPUT 次数、STEP_LIMIT_REACHED 次数、知识库命中率、拒答率、模块卡点、FAQ 聚类和学生反馈统计。新增 POST /api/feedback。统计接口只返回课程所需聚合数据，默认不暴露其他学生完整问题；CSV 导出进行公式注入防护。为统计口径、AgentRun 状态聚合和权限边界增加测试。
```

完成判定：

```text
教师端可查看模型、工具、检索和学习分析指标
教师端可查看 Agent 步数、停止原因和工具选择路径统计
学生评价能关联 question_id
FAQ 与测试记录可安全导出
无管理权限时不能调用知识库删除和 MCP 调试接口
```

### 第十三步：端到端联调、评测和报告材料

给 Codex 的命令：

```text
第十三步：完成第二阶段端到端联调和 Agent 自主决策验收。先在不配置 DEEPSEEK_API_KEY 的状态验证 TemplateProvider 离线闭环，再由我在本机 .env 填写密钥后验证 DeepSeek 模式。除第一阶段 8 个问题、第二阶段验收问题和 30 至 50 条离线检索评测集外，必须测试以下 Agent 行为：简单问题一次检索后回答；复杂实践问题连续选择两个不同工具；信息不足时 ask_user；用户补充后继续原 run_id；工具失败后选择其他工具或降级；模型请求非法工具时被 Action Guard 拦截；证据不足时 abstain；达到最大步骤和工具调用次数时强制停止。输出 Recall@K、Citation Accuracy、Groundedness、Abstention Accuracy、Intent Accuracy、Action Accuracy 和 Tool Routing Accuracy。补充 README、启动脚本、故障排查、数据来源与许可证说明、Agent Loop Mermaid 图、测试记录、演示脚本和 phase2_report_materials.md。不得在报告或截图中泄露 API Key。
```

建议验证命令：

```powershell
pytest -q
uvicorn backend.main:app --reload --port 8000
streamlit run frontend/app.py
```

最终完成判定：

```text
离线 TemplateProvider 模式可以完整启动和演示
DeepSeek 模式可以生成有证据、有引用的回答
DeepSeek/MCP/Chroma 故障时降级路径有效
DeepSeek 能根据 Observation 动态决定继续检索、调用工具、追问、回答或拒答
复杂任务具有两个以上可解释步骤，简单任务不会无意义循环
暂停和继续使用同一个 run_id，状态不会丢失
达到最大步骤、调用次数或超时后能够安全停止
上传、索引、检索、问答、日志、教师统计形成闭环
完整测试通过且 README 可以让新环境复现
报告包含架构、关键实现、评测结果、限制和后续工作
```

---

## 34. 第二阶段验收问题

在第一阶段 8 个问题基础上新增：

```text
请从我上传的 PDF 中总结 5G NAS 安全相关内容，并给出页码引用。
我上传了一篇 LLM 漏洞检测论文，请只根据论文内容总结研究方法。
请检索本地 CVE 样例，判断哪些适合作为课程实验。
请展示这次回答调用了哪些 MCP 工具。
请展示这次问答的 Agent 执行轨迹。
请把课程智能体系统架构生成为 Mermaid 图。
请重新索引知识库后再回答 3GPP 标准相关问题。
教师端能否看到工具调用统计和学生卡点模块？
在没有配置 DEEPSEEK_API_KEY 时，系统能否自动回退到本地模板回答？
配置 DeepSeek 后，同一个问题能否生成带 [S1]、[S2] 引用的答案？
关闭 MCP Server 后，本地只读检索是否仍可工作并显示 fallback？
上传包含“忽略系统指令”文本的资料后，系统是否仍遵守原有安全规则？
询问知识库不存在的 CVE 或标准条款时，系统是否明确拒答？
对于“设计区块链隐蔽通信实验”，Agent 是否会根据观察结果连续选择课程检索和论文检索，而不是直接生成答案？
当实验环境信息不足时，Agent 是否进入 REQUIRES_INPUT 并提出明确问题？
补充实验环境后，系统是否通过同一个 run_id 继续执行？
当 DeepSeek 请求不存在或未授权的工具时，Action Guard 是否拒绝执行？
当连续三步仍未完成任务时，Agent 是否以 STEP_LIMIT_REACHED 安全停止？
```

验收时同时检查：

```text
答案质量
引用真实性
拒答正确性
工具调用是否符合风险策略
DeepSeek 调用失败时的降级行为
Agent trace、model_calls、tool_calls 是否能够关联
agent_runs、agent_steps 是否完整记录决策、Observation 和停止原因
前端和日志是否泄露 API Key
```

---

## 35. 第二阶段报告创新点写法

```text
在第一阶段本地可演示 MCP-RAG 教学智能体基础上，第二阶段进一步引入 DeepSeek LLM、标准 MCP Server、向量数据库和有限自主 Agent Loop。
系统将课程知识库、论文样例、漏洞样例和标准文档统一解析为可检索知识单元。DeepSeek Planner 根据学生目标、可用工具和历史 Observation，在受控动作空间内自主选择检索、MCP 工具、追问、回答或拒答；后端 Action Guard 负责权限、参数和循环边界校验。
系统先构建带来源定位的 Evidence Bundle，再由 DeepSeek 基于证据生成回答，并通过 Citation Guard 校验引用，从而降低大模型幻觉风险。
同时，系统通过 Agent 轨迹、工具调用日志和教师端学习分析，实现了从“学生提问”到“证据检索”、从“工具调用”到“教师优化”的可解释闭环。
系统在 DeepSeek、MCP 或向量数据库不可用时仍具备明确的本地降级路径，既能支撑网络安全实践课程的课堂演示，也具备扩展到真实论文库、CVE/NVD 数据和标准文档检索的工程基础。
```

---

# 第三阶段：项目要求完整实现版开发参考报告

## 36. 第三阶段定位

第二阶段已经形成可运行、可演示、可评测的本地 MCP-RAG 教学智能体。第三阶段不推倒第二阶段，也不以更换前端或部署技术为目标，而是补齐老师原始要求中尚未完全产品化的部分，使系统从“本地完整智能体原型”升级为“逐条覆盖课程项目要求的完整版本”。

第三阶段重点完成：

- 真实用户、课程、班级和角色维度。
- 学生访问、活跃度和学习轨迹监控。
- 真实论文库、漏洞库和标准资料接口。
- 学生端课程资料库独立页面。
- 独立的创新启发与案例推荐模块。
- 学生问题的语义聚类和教师人工修正。
- 实验任务、学习进度和卡点分析。
- 面向网络安全内容的工具权限、隐私和审计。
- 覆盖原始需求的端到端验收与最终报告材料。

第三阶段继续使用：

```text
Streamlit
FastAPI
SQLite
Chroma
DeepSeek
MCP Server / Client Gateway
现有 Agent Loop
```

第三阶段暂不强制：

```text
React 重写
PostgreSQL 迁移
Docker 部署
Redis / Celery
公网部署
复杂 OAuth 单点登录
```

这些内容不属于老师原始要求的核心，可在项目答辩后作为工程化扩展。

---

## 37. 原始要求与第三阶段功能映射

| 原始项目要求 | 第二阶段现状 | 第三阶段补充 | 最终验收证据 |
|---|---|---|---|
| MCP 教学智能体 | 已有 MCP、Agent Loop、DeepSeek | 扩展真实外部数据工具和状态管理 | 工具调用轨迹、MCP 状态、Agent Run |
| 外部工具/数据库接口 | 本地工具和接口架构 | 真实 Connector、缓存、同步任务 | 在线 smoke、缓存回退、来源链接 |
| 论文库 | 本地论文样例 | Semantic Scholar、arXiv | 学生端搜索结果和真实来源 |
| 漏洞库 | 本地 CVE 样例 | NVD API | CVE 详情、更新时间、NVD 链接 |
| 标准文档库 | 本地标准片段 | 官方元数据链接与教师上传全文 | 标准编号、版本、页码引用 |
| 领域知识库 | 已有 Chroma 混合检索 | 外部资料规范化入库和来源治理 | 多来源检索与引用 |
| 访问监控 | 提问日志和最近问题 | 用户、课程、会话、学习事件 | 活跃用户、访问次数、最近活跃 |
| 学习轨迹 | Agent 和问题日志 | 页面、资料、任务、实践事件时间线 | 单个学生学习轨迹页 |
| 问题聚类 | 高频 FAQ | embedding 语义聚类和人工修正 | 聚类列表、代表问题、FAQ 导出 |
| 卡点分析 | 模块提问统计 | 实验步骤、进度、主动卡点上报 | 实验步骤卡点看板 |
| 学生基础模块 | 已有课程介绍和问答 | 完整使用说明与资料门户 | 页面演示 |
| 实践指导 | 已有结构化实践指导 | 与任务进度、真实资料和创新模块连接 | 完整实践案例 |
| 智能检索 | 已有本地 RAG | 本地加真实外部混合检索 | 论文、标准、漏洞分类摘要 |
| 案例推荐 | 问答式推荐 | 独立推荐数据与页面 | 论文、项目、竞赛案例列表 |
| 创新提问 | 实践指导中的创新点 | 独立创新会话和连续启发式追问 | innovation session 轨迹 |

---

## 38. 第三阶段总体架构

```text
学生 / 教师 / 管理员
        |
        v
Streamlit 前端
  -> 登录与课程选择
  -> 学生课程资料库
  -> 实践指导与实验进度
  -> 创新启发工作台
  -> 教师监控、聚类、卡点和 FAQ
        |
        v
FastAPI 应用服务
  -> Auth / RBAC
  -> Course Service
  -> Learning Event Service
  -> Resource Service
  -> Innovation Service
  -> Assignment Progress Service
  -> Question Clustering Service
  -> Teacher Analytics Service
        |
        +-------------------------------+
        |                               |
        v                               v
现有 Agent Orchestrator           Connector Registry
  -> DeepSeek Planner              -> Semantic Scholar
  -> RAG / MCP                      -> arXiv
  -> Action Guard                   -> NVD
  -> Citation Guard                 -> GitHub（可选 Token）
  -> Safety Guard                   -> Standard Source
        |                               |
        +---------------+---------------+
                        v
               Normalized Resource Layer
               -> external_resources
               -> resource_cache
               -> connector_sync_runs
               -> Chroma 索引
                        |
                        v
               SQLite + Chroma + 本地文件
```

### 38.1 架构原则

- 所有外部数据源通过 Connector 接口接入，业务代码不直接依赖具体 API。
- 所有外部结果转换成统一 `ExternalResource`，保留来源和更新时间。
- 默认优先查询本地缓存，必要时再访问外部 API。
- 外部接口失败不能导致整个问答系统不可用。
- 学习行为、问题、任务和创新会话必须关联真实用户和课程。
- 普通学生不能访问教师统计、知识库管理和工具调试接口。
- 网络安全相关高风险操作仍由 Action Guard 和 Tool Policy 控制。

---

## 39. 第三阶段目录结构建议

在现有项目基础上新增，不移动已经稳定的第二阶段模块：

```text
course-agent/
├── backend/
│   ├── auth/
│   │   ├── service.py
│   │   ├── dependencies.py
│   │   ├── password.py
│   │   └── schemas.py
│   ├── courses/
│   │   ├── service.py
│   │   └── schemas.py
│   ├── analytics/
│   │   ├── events.py
│   │   ├── activity.py
│   │   ├── clustering.py
│   │   ├── stuck_analysis.py
│   │   └── schemas.py
│   ├── connectors/
│   │   ├── base.py
│   │   ├── registry.py
│   │   ├── schemas.py
│   │   ├── semantic_scholar.py
│   │   ├── arxiv.py
│   │   ├── nvd.py
│   │   ├── github.py
│   │   └── standard_source.py
│   ├── resources/
│   │   ├── service.py
│   │   ├── repository.py
│   │   ├── normalizer.py
│   │   └── schemas.py
│   ├── innovation/
│   │   ├── service.py
│   │   ├── prompts.py
│   │   ├── recommender.py
│   │   └── schemas.py
│   ├── assignments/
│   │   ├── service.py
│   │   ├── progress.py
│   │   └── schemas.py
│   ├── api/
│   │   ├── auth.py
│   │   ├── courses.py
│   │   ├── events.py
│   │   ├── resources.py
│   │   ├── innovation.py
│   │   ├── assignments.py
│   │   └── teacher_phase3.py
│   └── storage/
│       └── migrations/
├── frontend/
│   └── pages/
│       ├── login.py
│       ├── resources.py
│       ├── innovation.py
│       ├── assignments.py
│       ├── learning_trace.py
│       └── teacher_phase3.py
├── datasets/
│   ├── competitions/
│   └── recommendation_seed/
├── tests/
│   ├── connectors/
│   ├── test_auth_rbac.py
│   ├── test_learning_events.py
│   ├── test_resource_portal.py
│   ├── test_innovation.py
│   ├── test_question_clustering.py
│   ├── test_assignment_progress.py
│   └── test_phase3_e2e.py
└── docs/
    ├── phase3_architecture.md
    ├── phase3_test_report.md
    └── final_demo_script.md
```

---

## 40. 第三阶段依赖和配置

### 40.1 新增依赖

在 `requirements-phase3.txt` 中基于原依赖增量添加：

```text
PyJWT
pwdlib[argon2]
respx
```

说明：

| 依赖 | 用途 |
|---|---|
| `PyJWT` | 本地课程系统身份令牌 |
| `pwdlib[argon2]` | 密码哈希，数据库不保存明文密码 |
| `respx` | 模拟外部 HTTP API，避免测试消耗配额 |

`httpx`、`tenacity`、`scikit-learn`、`sentence-transformers`、`pandas` 已在前两阶段使用，不重复引入功能相同的框架。

### 40.2 `.env.example` 配置

```env
# 身份与课程
JWT_SECRET_KEY=
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=480
ENABLE_DEMO_ACCOUNTS=true

# 外部数据源总开关
EXTERNAL_CONNECTOR_MODE=hybrid
EXTERNAL_CACHE_TTL_HOURS=24
EXTERNAL_HTTP_TIMEOUT_SECONDS=20
EXTERNAL_HTTP_MAX_RETRIES=2
EXTERNAL_RESOURCE_INDEX_ENABLED=true

# 论文
ENABLE_SEMANTIC_SCHOLAR=true
SEMANTIC_SCHOLAR_API_KEY=
ENABLE_ARXIV=true

# 漏洞
ENABLE_NVD=true
NVD_API_KEY=

# 开源项目
ENABLE_GITHUB_SEARCH=false
GITHUB_TOKEN=

# 标准资料
ENABLE_STANDARD_REMOTE_METADATA=true
STANDARD_DOCUMENT_MODE=uploaded_local

# 聚类
QUESTION_CLUSTER_MIN_SIZE=2
QUESTION_CLUSTER_DISTANCE_THRESHOLD=0.35
QUESTION_CLUSTER_LOOKBACK_DAYS=90

# 学习事件
LEARNING_EVENT_RETENTION_DAYS=365
SESSION_IDLE_TIMEOUT_MINUTES=30
ACTIVITY_HEARTBEAT_SECONDS=60
```

安全要求：

- `.env.example` 中所有密钥保持为空。
- 真实密钥只写入本机 `.env`。
- `.env` 必须在 `.gitignore` 中。
- `JWT_SECRET_KEY` 使用随机长字符串，不与 DeepSeek API Key 共用。
- 系统配置接口只返回“已配置/未配置”，不得返回密钥内容。

---

## 41. 第三阶段数据库设计

所有表必须通过版本化 migration 增加，禁止删除第二阶段数据库重建。

### 41.1 users 表

```text
id
username
display_name
password_hash
role              student / teacher / admin
enabled
created_at
last_login_at
```

### 41.2 courses、classes、class_members 表

```text
courses
- id
- name
- description
- assessment_method
- created_at

classes
- id
- course_id
- name
- term
- teacher_id
- created_at

class_members
- id
- class_id
- user_id
- joined_at
```

### 41.3 learning_sessions 表

```text
id
user_id
course_id
class_id
started_at
last_active_at
ended_at
active_seconds
client_session_id
```

### 41.4 learning_events 表

```text
id
session_id
user_id
course_id
class_id
event_type
page_name
module
resource_id
assignment_id
metadata_json
created_at
```

事件类型统一使用枚举：

```text
LOGIN
PAGE_VIEW
RESOURCE_SEARCH
RESOURCE_OPEN
QUESTION_ASKED
PRACTICE_GUIDE_GENERATED
INNOVATION_SESSION_STARTED
INNOVATION_QUESTION_ANSWERED
ASSIGNMENT_PROGRESS_UPDATED
STUCK_REPORTED
FEEDBACK_SUBMITTED
```

### 41.5 external_resources 表

```text
id
external_id
resource_type       paper / standard / vulnerability / project / competition
source_name
title
authors_json
abstract
url
published_at
tags_json
module
raw_metadata_json
fetched_at
expires_at
content_hash
rights_note
indexed
vector_id
```

对 `(source_name, external_id)` 建立唯一约束，防止重复缓存。

### 41.6 connector_sync_runs 表

```text
id
connector_name
query
status
result_count
cache_hit
error_code
error_message
started_at
finished_at
```

### 41.7 innovation_sessions、innovation_questions、innovation_ideas 表

```text
innovation_sessions
- id
- user_id
- course_id
- topic
- current_solution
- status
- created_at
- updated_at

innovation_questions
- id
- session_id
- sequence_number
- question
- student_answer
- rationale
- created_at

innovation_ideas
- id
- session_id
- title
- problem
- proposed_change
- validation_method
- risks
- evidence_json
- created_at
```

### 41.8 assignments、assignment_steps、student_assignment_progress 表

```text
assignments
- id
- course_id
- title
- module
- description
- created_at

assignment_steps
- id
- assignment_id
- step_order
- title
- description

student_assignment_progress
- id
- assignment_id
- step_id
- user_id
- status
- progress_percent
- updated_at
```

### 41.9 stuck_reports 表

```text
id
user_id
assignment_id
step_id
module
description
resolved
created_at
resolved_at
```

### 41.10 question_clusters、question_cluster_members 表

```text
question_clusters
- id
- course_id
- label
- representative_question
- member_count
- status            generated / reviewed / approved
- created_at
- updated_at

question_cluster_members
- id
- cluster_id
- question_id
- similarity_score
```

教师人工合并、拆分、重命名后，不得被下一次自动聚类无条件覆盖。

### 41.11 现有第二阶段表的课程上下文升级

第三阶段需要通过 migration 为现有表补充必要关联：

```text
questions          增加 course_id、class_id、session_id
conversations      增加 course_id、class_id
agent_runs         增加 user_id、course_id、class_id
feedback           增加 course_id
documents          增加 course_id、visibility
```

迁移时把现有演示数据回填到默认 demo course 和 demo class，并保留原主键、问题、引用、轨迹和时间戳。不得通过删除旧数据解决外键问题。

---

## 42. 外部数据源 Connector 设计

### 42.1 统一接口

```python
class ExternalConnector:
    name: str

    async def health_check(self) -> ConnectorHealth: ...
    async def search(self, query: ResourceQuery) -> list[ExternalResource]: ...
    async def get_by_id(self, external_id: str) -> ExternalResource | None: ...
```

`ResourceQuery` 至少包含：

```text
query
resource_type
module
year_from
year_to
page
page_size
allow_remote
```

每个 Connector 必须实现：

- 请求超时和有限重试。
- API Key 从环境变量读取。
- HTTP 状态码到领域错误的转换。
- 速率限制和缓存优先策略。
- 结果规范化、去重和来源标注。
- 可配置地把论文摘要、漏洞描述、标准元数据和项目说明写入 Chroma，并记录 `vector_id`。
- mock contract test。
- 可选 live smoke test，默认不在普通 `pytest` 中执行。

### 42.2 Semantic Scholar 与 arXiv

- Semantic Scholar 用于论文元数据、摘要、作者、年份和来源链接。
- arXiv 用于公开预印本检索，并作为论文接口降级来源。
- 不宣称已获得论文全文时，不生成“基于全文”的结论。
- 相同 DOI、arXiv ID 或高度相似标题需要去重。

### 42.3 NVD

- 根据 CVE 编号、关键词、CWE 和时间范围检索。
- 展示 CVE ID、描述、CVSS、CWE、发布日期、更新时间和 NVD 来源链接。
- 缓存结果并保存 `fetched_at`，避免重复消耗 API 配额。
- 漏洞数据只用于课程学习和授权实验，不自动执行利用代码。

### 42.4 标准资料

- `standard_source` 保存标准编号、标题、版本、官方链接和本地文档状态。
- 不自动批量抓取受版权保护的标准全文。
- 教师上传合法获得的标准 PDF 后，使用现有解析、切片、Chroma 和引用定位能力。
- 回答必须区分“外部元数据”和“本地上传正文证据”。

### 42.5 GitHub 和竞赛题目

- GitHub Connector 用于推荐开源项目，默认关闭，配置 Token 后启用。
- 展示仓库名称、描述、语言、更新时间和来源链接，不自动执行仓库代码。
- 竞赛题目使用教师维护的本地结构化数据集，避免依赖不稳定网页抓取。

### 42.6 与 MCP 的连接

扩展 MCP 工具：

```text
paper_search_real
vuln_lookup_real
standard_metadata_lookup
github_project_search
competition_case_search
```

Agent 默认先查询本地知识库；只有本地证据不足、工具已启用且策略允许时，才能调用真实外部工具。所有外部工具调用必须记录来源、缓存状态、耗时和错误。

---

## 43. 学生端第三阶段设计

### 43.1 使用说明与课程信息

独立页面展示：

- 智能问答、实践指导、资料检索和创新启发的使用方法。
- 课程目标、考核方式、专题示例和安全要求。
- 引用、置信度、拒答和工具调用轨迹的含义。
- 网络安全实验必须在授权、本地或隔离环境中进行的提示。

### 43.2 课程资料库

资料类型：

```text
课程文档
论文
标准
漏洞
开源项目
竞赛题目
```

支持：

- 关键词、专题、类型、年份和来源筛选。
- 本地与外部来源统一结果列表。
- 摘要、分类、来源链接和最后更新时间。
- 打开资料时记录 `RESOURCE_OPEN` 事件。
- 外部接口不可用时显示缓存结果和更新时间。

### 43.3 实践指导与任务进度

- 将实践方案关联到课程、实验任务和学生。
- 学生可以更新当前步骤、完成比例和备注。
- 学生可以主动提交“我卡在这一步”。
- Agent 根据当前步骤、已有证据和卡点继续指导。
- 教师端按实验和步骤统计卡点。

### 43.4 创新启发工作台

```text
选择专题
-> 描述当前方案
-> 检索论文、漏洞、标准、项目和竞赛案例
-> 推荐相关案例
-> 识别当前方案限制
-> 生成 3 至 5 个启发式问题
-> 学生回答
-> Agent 根据回答继续追问或收敛
-> 输出候选创新方案和验证方法
```

输出至少包括：

```text
相关案例及来源
已有方案的限制
启发式问题
候选创新方向
需要验证的假设
建议实验方法
风险和伦理注意事项
```

创新启发不能凭空编造论文、开源项目、竞赛题目或漏洞编号。事实型推荐必须带来源；纯思路建议应明确标记为“模型建议”。

---

## 44. 教师端第三阶段设计

### 44.1 访问监控

按课程、班级和时间范围显示：

```text
独立活跃学生数
访问次数
最近活跃时间
活跃天数
页面访问分布
资源打开次数
问答次数
实践指导次数
创新会话次数
```

“活跃度”必须在报告中说明统计口径。例如：某天产生至少一个有效学习事件视为活跃一天；连续心跳之间的间隔超过空闲阈值不计入活跃时间。

### 44.2 学习轨迹

教师可以按学生查看时间线：

```text
登录
访问课程模块
检索和打开资料
提出问题
生成实践方案
更新实验进度
报告卡点
参与创新启发
提交反馈
```

默认只展示教学所需摘要，敏感问题全文需要更高权限。

### 44.3 语义问题聚类

推荐流程：

```text
选择时间范围和课程
-> 获取学生问题
-> 使用现有 embedding 生成向量
-> 使用余弦距离和 AgglomerativeClustering 或 DBSCAN 聚类
-> 小簇和噪声单独标记
-> 选择代表问题
-> DeepSeek 生成简短聚类标签
-> 教师合并、拆分、重命名和批准
-> 批准后的聚类生成 FAQ
```

聚类结果应显示成员数、代表问题、模块分布和最近出现时间。自动重新聚类不得覆盖教师批准的标签和人工调整结果。

### 44.4 卡点分析

卡点来源综合：

- 学生主动提交的 `stuck_reports`。
- 同一实验步骤的重复问题。
- 同一模块高拒答率。
- 同一步骤长时间未更新。
- 多次打开同一资料但任务进度未推进。

系统输出“疑似卡点”，教师可以确认或忽略，避免把普通浏览行为直接判定为困难。

### 44.5 FAQ 输出

支持按课程、班级、模块和时间范围导出：

```text
聚类标签
代表问题
问题数量
建议回答
引用来源
相关模块
教师审核状态
```

CSV 导出必须防止以 `=`, `+`, `-`, `@` 开头的单元格造成公式注入。

---

## 45. 第三阶段 API 设计

### 45.1 身份和课程

```text
POST /api/auth/login
GET  /api/auth/me
POST /api/auth/logout

GET  /api/courses
GET  /api/courses/{course_id}
GET  /api/classes
GET  /api/classes/{class_id}/members
```

第三阶段开始，后端从登录身份中确定 `user_id`，不再信任学生请求体中任意传入的 `user_id`。

### 45.2 学习事件

```text
POST /api/learning/sessions
POST /api/learning/sessions/{session_id}/heartbeat
POST /api/learning/sessions/{session_id}/end
POST /api/learning/events
GET  /api/teacher/activity
GET  /api/teacher/students/{user_id}/timeline
```

事件接口只接受允许的事件枚举，并由后端补充用户、课程和时间信息。

### 45.3 课程资料

```text
GET  /api/resources
GET  /api/resources/{resource_id}
POST /api/resources/search
GET  /api/connectors/status
POST /api/connectors/{connector_name}/sync
GET  /api/connectors/sync-runs
```

普通学生只能搜索和查看资料；教师和管理员才能触发同步任务。

### 45.4 创新启发

```text
POST /api/innovation/sessions
GET  /api/innovation/sessions/{session_id}
POST /api/innovation/sessions/{session_id}/answer
POST /api/innovation/sessions/{session_id}/finalize
```

### 45.5 实验任务与卡点

```text
GET  /api/assignments
GET  /api/assignments/{assignment_id}
PUT  /api/assignments/{assignment_id}/progress
POST /api/assignments/{assignment_id}/stuck
GET  /api/teacher/assignments/analytics
```

### 45.6 问题聚类与 FAQ

```text
POST /api/teacher/question-clusters/rebuild
GET  /api/teacher/question-clusters
PATCH /api/teacher/question-clusters/{cluster_id}
POST /api/teacher/question-clusters/merge
POST /api/teacher/question-clusters/{cluster_id}/split
POST /api/teacher/question-clusters/{cluster_id}/approve
GET  /api/teacher/faq/export
```

---

## 46. 权限、隐私和网络安全边界

### 46.1 最小角色权限

| 功能 | 学生 | 教师 | 管理员 |
|---|---:|---:|---:|
| 智能问答与实践指导 | 是 | 是 | 是 |
| 查看课程资料 | 是 | 是 | 是 |
| 创新启发 | 是 | 是 | 是 |
| 更新自己的实验进度 | 是 | 否 | 是 |
| 查看班级统计 | 否 | 是 | 是 |
| 查看学生学习轨迹 | 否 | 是 | 是 |
| 管理聚类与 FAQ | 否 | 是 | 是 |
| 上传和删除知识库 | 否 | 是 | 是 |
| 配置外部 Connector | 否 | 否 | 是 |
| 调试 MCP 工具 | 否 | 否 | 是 |

### 46.2 数据与隐私

- 数据库只保存密码哈希，不保存明文密码。
- 教师只能查看自己课程或班级的数据。
- 学生只能查看自己的会话、轨迹、任务和创新记录。
- 学习事件设置保留时间，并提供测试环境清理方式。
- 日志不保存 API Key、JWT、密码和完整外部响应中的敏感字段。
- 教师导出数据默认使用显示名或课程内标识，不额外暴露个人信息。

### 46.3 网络安全教学边界

- 漏洞、PoC 和隐蔽通信内容仅用于课程学习和授权实验。
- Agent 不自动执行漏洞利用、端口扫描、恶意代码或公网攻击。
- GitHub 项目只推荐和展示元数据，不自动克隆或运行。
- 外部资料和工具返回均视为不可信输入，继续经过 Safety Guard。
- 高风险工具保持默认禁用，并记录所有调用尝试。

---

## 47. 第三阶段测试与验收指标

### 47.1 测试分层

```text
单元测试       数据模型、权限、聚类、推荐规则
Connector测试  使用 respx 模拟成功、限流、超时和错误响应
集成测试       FastAPI + SQLite + Chroma + MCP + mock DeepSeek
离线端到端     TemplateProvider + 本地缓存完整闭环
在线 smoke     DeepSeek 和启用的真实外部 Connector
UI 验收        学生和教师核心操作流程
```

普通 `pytest` 必须不依赖公网和真实 API Key。在线测试使用 `pytest -m live` 单独执行，并允许因外部服务不可用而给出明确跳过原因，不能静默伪造成功。

### 47.2 核心指标

```text
身份权限正确率
学习事件记录完整率
Connector 成功率与缓存命中率
外部资源去重准确性
聚类覆盖率和噪声比例
教师人工确认后的聚类可用率
资料推荐相关性
创新问题与当前方案相关性
实验卡点统计准确性
FAQ 导出正确性
原第二阶段 117 项测试回归结果
```

### 47.3 最终验收演示

1. 学生登录并进入指定课程。
2. 浏览使用说明、课程简介和考核方式。
3. 在课程资料库搜索一个 5G NAS 或区块链主题。
4. 同时看到本地资料和至少一个真实外部论文来源。
5. 查询一个 CVE，展示 NVD 来源、更新时间和缓存状态。
6. 打开标准资料，区分官方元数据和教师上传正文引用。
7. 生成结构化实践方案并关联一个实验任务。
8. 更新实验进度并在某一步提交卡点。
9. 启动创新启发会话，查看案例推荐并回答至少一个启发式问题。
10. 教师端查看该学生的访问、资源、问答、实验和创新轨迹。
11. 教师重建语义问题聚类，修改标签并批准生成 FAQ。
12. 教师查看实验步骤卡点并导出 FAQ。
13. 关闭一个外部 Connector，证明缓存或本地检索仍可工作。
14. 使用学生账号尝试访问教师接口，确认返回 403。

---

## 48. 第三阶段 VSCode 协作开发步骤

以下命令可以按顺序直接发送给 VSCode 中的 Codex。每一步都必须先检查当前代码和第二阶段测试，不得覆盖已经完成的 Agent、RAG、MCP、前端和统计功能。

每次命令结尾统一附加：

```text
先阅读现有实现、数据库 migration 和测试，保留用户已有修改。使用 apply_patch 做人工编辑。完成后运行本步骤测试和完整回归测试，失败时定位并修复后再结束。说明修改文件、数据库变化、启动方式、测试结果和剩余限制。不要在代码、日志、测试数据、截图或提交中写入真实 API Key、JWT_SECRET_KEY 或密码。
```

### 第一步：建立第三阶段基线和需求追踪表

给 Codex 的命令：

```text
第三阶段第一步：审查当前第二阶段项目，不重写现有功能。记录当前 pytest 基线、DeepSeek smoke、离线评测和启动方式。新增 requirements-phase3.txt、docs/phase3_architecture.md 和 docs/requirements_traceability.md。把老师原始要求逐项映射到当前模块、第三阶段任务和验收证据。更新 .env.example，加入身份、外部 Connector、缓存、聚类和学习事件配置，所有密钥保持为空；确认 .env 被 .gitignore 忽略。只建立第三阶段骨架和配置，不先实现后续业务。
```

完成判定：

```text
第二阶段原有测试全部通过
requirements-phase3.txt 可以安装
需求追踪表覆盖老师原始要求①至④及创新启发模块
.env.example 无真实密钥
README 明确第二阶段已完成、第三阶段目标和启动方式
```

### 第二步：实现用户、课程、班级和最小 RBAC

给 Codex 的命令：

```text
第三阶段第二步：通过版本化 migration 新增 users、courses、classes、class_members 表，不删除或重建现有数据库。为第二阶段已有 questions、conversations、agent_runs、feedback、documents 补充 user/course/class/session/visibility 等必要关联，并把旧演示数据幂等回填到默认 demo course 和 demo class。使用 pwdlib[argon2] 保存密码哈希，使用 PyJWT 实现本地访问令牌。新增 backend/auth、backend/courses 和对应 API，实现 POST /api/auth/login、GET /api/auth/me、POST /api/auth/logout、GET /api/courses、GET /api/classes。定义 student、teacher、admin 权限依赖。第三阶段开始由后端根据令牌确定 user_id，不再信任请求体中任意传入的 user_id。提供幂等的 demo seed 命令，创建一个课程、一个班级、一个教师和两个学生，默认密码只用于本地演示并在首次启动提示修改。为迁移回填、登录、密码哈希、令牌过期、跨用户访问、教师班级范围和管理员权限编写 tests/test_auth_rbac.py。
```

实施约束：

- 不在数据库保存明文密码。
- JWT Secret 只从 `.env` 读取。
- 不为本地课程项目引入复杂 OAuth。
- 保持第二阶段接口兼容，但第三阶段页面必须使用登录身份。

测试与完成判定：

```powershell
pytest tests/test_auth_rbac.py -q
pytest -q
```

```text
三类角色权限正确
学生不能查看其他学生数据或教师接口
教师只能查看所属课程和班级
重复执行 seed 不会创建重复数据
旧 questions、Agent 轨迹和引用在 migration 后仍完整可查
第二阶段测试无回归
```

### 第三步：实现学习会话、访问事件和学习轨迹

给 Codex 的命令：

```text
第三阶段第三步：通过 migration 新增 learning_sessions 和 learning_events 表，定义允许的事件枚举。实现 backend/analytics/events.py、activity.py 和对应 API：创建会话、heartbeat、结束会话、写入学习事件、教师活动统计、学生学习时间线。前端登录后创建 client_session_id；页面访问、资料搜索与打开、提问、实践指导、创新会话、任务进度、卡点和反馈写入事件。后端自动补充 user_id、course_id、class_id 和时间，拒绝客户端伪造其他用户。active_seconds 只累计合理心跳间隔，超过 SESSION_IDLE_TIMEOUT 的间隔不计入活跃时间。为事件去重、空闲超时、权限和统计口径编写 tests/test_learning_events.py。
```

测试与完成判定：

```powershell
pytest tests/test_learning_events.py -q
pytest -q
```

```text
学生完整操作可以形成有序时间线
访问次数、最近活跃、活跃天数和页面分布计算正确
重复 heartbeat 不会无限累计活跃时间
学生不能提交其他用户的事件
```

### 第四步：实现统一 Connector、缓存和同步任务框架

给 Codex 的命令：

```text
第三阶段第四步：新增 backend/connectors/base.py、registry.py、schemas.py 和 backend/resources 的 repository、normalizer、service。定义 ExternalConnector、ResourceQuery、ExternalResource、ConnectorHealth 和统一领域错误。通过 migration 新增 external_resources 与 connector_sync_runs，对 source_name + external_id 建立唯一约束。实现缓存优先、TTL、内容 hash 去重、超时、有限重试、分页、来源和 fetched_at；提供配置开关，把允许索引的论文摘要、漏洞描述、标准元数据和项目说明写入现有 Chroma，并保存 indexed 与 vector_id，删除或过期刷新时同步处理旧向量。新增 /api/connectors/status、/api/connectors/{name}/sync、/api/connectors/sync-runs 和统一资源搜索服务。本步骤先使用 FakeConnector 验证框架，不接真实服务。使用 respx/mock 编写连接成功、超时、429、5xx、缓存命中、过期刷新、重复资源和向量同步测试。
```

实施约束：

- Connector 不直接写前端响应，先规范化为 ExternalResource。
- 外部失败返回明确状态并允许本地缓存降级。
- 同步操作只有教师或管理员可触发。

测试与完成判定：

```powershell
pytest tests/connectors -q
pytest -q
```

```text
FakeConnector contract test 全部通过
缓存有效期、去重和错误映射正确
外部缓存资源与 Chroma 向量保持一致
外部服务失败不影响本地 RAG 和已有问答
```

### 第五步：接入真实论文、漏洞和标准数据源并暴露 MCP 工具

给 Codex 的命令：

```text
第三阶段第五步：在统一 Connector 框架上实现 semantic_scholar.py、arxiv.py、nvd.py、standard_source.py，并可选实现 github.py。Semantic Scholar 和 arXiv 返回论文标题、作者、摘要、年份、DOI/arXiv ID 和来源链接并去重；NVD 返回 CVE、描述、CVSS、CWE、发布日期、更新时间和来源链接；standard_source 返回标准编号、版本、官方链接和本地上传状态，不抓取受版权保护全文。扩展 MCP 工具 paper_search_real、vuln_lookup_real、standard_metadata_lookup、github_project_search。Agent 只在本地证据不足且 Connector 启用时调用真实工具。所有普通测试使用 respx，不访问公网；新增 pytest live 标记进行可选真实 smoke，并在 README 说明 API Key、配额和失败行为。
```

测试与完成判定：

```powershell
pytest tests/connectors tests/test_mcp_tools.py -q
pytest -q
pytest -m live -q
```

```text
mock contract test 覆盖成功、空结果、限流、超时和服务错误
至少一个论文数据源和 NVD 完成真实 smoke，或明确记录外部服务不可用原因
结果有真实来源链接、fetched_at 和 cache_hit
关闭联网工具后自动使用本地资料或缓存
标准工具能区分元数据和本地正文
```

### 第六步：实现学生端课程资料库

给 Codex 的命令：

```text
第三阶段第六步：新增资源 API 和 Streamlit 课程资料库页面。统一展示课程文档、论文、标准、漏洞、开源项目和竞赛题目，支持关键词、专题、资料类型、年份和来源筛选；显示标题、摘要、标签、来源、外部链接、缓存状态和最后更新时间。竞赛题目使用 datasets/competitions 中的教师维护数据，不抓取不稳定网页。点击资料写入 RESOURCE_OPEN，搜索写入 RESOURCE_SEARCH。学生只能查看资料；同步和管理按钮仅教师或管理员可见。为筛选、分页、去重、权限、缓存降级和事件记录编写 tests/test_resource_portal.py。
```

测试与完成判定：

```powershell
pytest tests/test_resource_portal.py -q
pytest -q
```

```text
学生可独立浏览六类课程资料
本地和外部结果使用统一页面
断网时显示缓存结果和更新时间
搜索和打开行为进入学习轨迹
```

### 第七步：实现独立创新启发模块

给 Codex 的命令：

```text
第三阶段第七步：通过 migration 新增 innovation_sessions、innovation_questions、innovation_ideas。实现 backend/innovation/service.py、recommender.py、prompts.py 和 API。学生输入专题与当前方案后，调用现有 Agent、RAG 和允许的 MCP 工具检索论文、标准、漏洞、项目和竞赛案例；先输出带来源案例和当前方案限制，再生成 3 至 5 个与方案相关的启发式问题。学生回答后使用同一 innovation session 继续追问或收敛，最终输出候选创新方向、需要验证的假设、实验方法、风险和证据。事实推荐必须带来源，纯模型建议标记为“模型建议”。加入网络安全教学边界，不生成对公网目标的攻击执行步骤。编写 tests/test_innovation.py，模型和外部接口均使用 mock。
```

测试与完成判定：

```powershell
pytest tests/test_innovation.py -q
pytest -q
```

```text
创新启发是独立页面和持久化会话，不只是实践方案中的一个字段
至少推荐论文、项目或竞赛案例中的两类
能够连续问答并保存学生回答
最终创新方向包含证据和验证方法
不存在来源的案例不会被伪造
```

### 第八步：实现语义问题聚类、人工修正和 FAQ

给 Codex 的命令：

```text
第三阶段第八步：通过 migration 新增 question_clusters 和 question_cluster_members。实现 backend/analytics/clustering.py，读取指定课程和时间范围内的学生问题，复用现有 embedding，采用余弦距离与 AgglomerativeClustering 或 DBSCAN 做语义聚类，标记小簇和噪声，选择代表问题；使用 DeepSeek 生成简短标签，失败时使用关键词标签。实现聚类重建、列表、重命名、合并、拆分、批准和 FAQ 导出 API。自动重建不得覆盖教师批准的标签和成员调整。FAQ 建议回答必须经过本地知识库检索和引用检查。编写 tests/test_question_clustering.py，使用固定向量或 mock embedding 保证测试稳定。
```

测试与完成判定：

```powershell
pytest tests/test_question_clustering.py -q
pytest -q
```

```text
语义相近但措辞不同的问题能够归为同一簇
不相关问题不会被强行合并
教师可重命名、合并、拆分和批准
批准结果不会被下一次自动重建覆盖
FAQ 导出包含标签、代表问题、数量、回答和引用
```

### 第九步：实现实验任务、学生进度和卡点分析

给 Codex 的命令：

```text
第三阶段第九步：通过 migration 新增 assignments、assignment_steps、student_assignment_progress、stuck_reports。实现课程实验任务、步骤列表、学生进度更新和主动卡点上报 API 与 Streamlit 页面。将现有实践指导结果关联 assignment_id 和当前 step_id，使 Agent 可以根据当前步骤继续指导。实现 backend/analytics/stuck_analysis.py，综合主动上报、同一步骤重复问题、高拒答率和长期未推进生成“疑似卡点”，但不得仅凭一次页面访问判定卡点。教师可确认、忽略或标记已解决。编写 tests/test_assignment_progress.py 和卡点统计测试。
```

测试与完成判定：

```powershell
pytest tests/test_assignment_progress.py -q
pytest -q
```

```text
学生只能更新自己的任务进度
实践指导可以关联实验及步骤
主动卡点立即进入教师统计
疑似卡点有可解释原因并可由教师修正
教师可按实验、步骤、模块和班级查看卡点
```

### 第十步：完成教师端综合看板和访问监控

给 Codex 的命令：

```text
第三阶段第十步：扩展 Streamlit 教师端和 backend/api/teacher_phase3.py。增加课程与班级筛选、独立活跃学生、访问次数、活跃天数、最近活跃、页面分布、资料搜索与打开、问答、实践指导、创新会话、实验进度、卡点、语义聚类、FAQ 审核和导出。增加单个学生学习轨迹时间线，但默认展示教学摘要。所有统计明确时间范围和口径，避免把 Agent 内部步骤误计为学生访问。教师只能查看自己课程，管理员可查看全部。为统计 SQL、班级隔离、时间范围、CSV 公式注入和空数据页面编写测试。
```

测试与完成判定：

```powershell
pytest tests/test_teacher_phase3.py tests/test_auth_rbac.py -q
pytest -q
```

```text
教师端覆盖访问监控、学习轨迹、问题聚类、卡点分析和 FAQ 输出
课程、班级和时间筛选结果正确
CSV 导出不存在公式注入风险
教师不能跨课程查看未授权学生
```

### 第十一步：完成使用说明、安全审计和接口降级验证

给 Codex 的命令：

```text
第三阶段第十一步：完善学生端使用说明、课程简介、考核方式、专题示例和网络安全实验边界。审查第三阶段所有 API 的身份依赖、角色权限、输入校验、文件上传、外部 URL、日志脱敏和错误响应。验证学生不能调用 Connector 同步、知识库删除、聚类管理、教师统计和 MCP 调试接口。验证外部资源与文档中的提示注入不能提升权限或触发高风险工具。加入密钥扫描测试、403 权限测试、CSV 注入测试、缓存过期测试和外部 Connector 关闭时的本地降级测试。
```

测试与完成判定：

```powershell
pytest tests/test_security_phase3.py tests/test_auth_rbac.py -q
pytest -q
```

```text
使用说明覆盖所有学生端模块
敏感配置不会通过 API、日志或页面泄露
角色权限和课程隔离全部通过
禁用或故障 Connector 时系统仍可使用本地知识库
高风险网络安全请求受到现有 Guard 限制
```

### 第十二步：第三阶段端到端验收、评测和最终报告

给 Codex 的命令：

```text
第三阶段第十二步：执行完整回归与原始需求验收。保留第二阶段全部测试，并新增 tests/test_phase3_e2e.py，使用 demo 教师和学生完成登录、资料检索、真实或 mock 外部论文、NVD 漏洞、标准元数据、本地正文引用、实践任务、进度更新、卡点上报、创新启发、访问轨迹、语义聚类、教师修正、FAQ 导出和权限拒绝的完整流程。普通 pytest 必须离线可重复；真实 DeepSeek 和外部 Connector 使用单独 live smoke。输出第三阶段测试报告、需求追踪完成表、数据来源与许可证说明、隐私与安全说明、最终架构图、数据库图、演示脚本和项目限制。更新 README，让新环境可以从安装、migration、seed、启动到演示完整复现。
```

建议验证命令：

```powershell
pytest -q
pytest -m live -q
uvicorn backend.main:app --reload --port 8000
streamlit run frontend/app.py
```

最终完成判定：

```text
第二阶段全部测试继续通过
第三阶段新增测试全部通过
DeepSeek 与至少一个真实论文源、NVD 完成 live smoke 或有明确外部失败记录
学生端覆盖基础模块、课程资料、实践指导和创新启发
教师端覆盖访问监控、学习轨迹、语义聚类、卡点分析和 FAQ 导出
真实来源、缓存、引用、权限和降级路径可演示
需求追踪表中老师原始要求均有代码模块、测试和演示证据
README 和最终报告可以让新环境复现
```

---

## 49. 第三阶段最终验收问题

### 49.1 学生端

```text
如何使用课程智能体完成一次从选题到实验方案的完整流程？
课程目标、考核方式和专题示例是什么？
请从真实论文源查找 5G NAS 安全相关研究，并显示来源和更新时间。
请查询一个适合作为课程实验分析样例的 CVE，并显示 CVSS、CWE 和 NVD 链接。
3GPP TS 24.501 的官方元数据和本地正文分别来自哪里？
请为“基于区块链的隐蔽通信”生成实践方案并关联实验任务。
我在实验的结果分析步骤遇到困难，请记录卡点并继续指导。
请根据我的当前方案推荐论文和开源项目，并提出三个启发式创新问题。
我回答创新问题后，系统能否继续同一创新会话并生成候选创新方向？
```

### 49.2 教师端

```text
本周有多少独立活跃学生，各自最近活跃时间是什么？
某学生本周访问了哪些模块、资料和实验步骤？
语义相近但表述不同的问题能否形成同一热点问题簇？
教师能否合并、拆分、重命名并批准问题簇？
哪些实验步骤被主动报告为卡点，哪些只是系统推断的疑似卡点？
能否按课程、班级、模块和时间范围导出审核后的 FAQ？
```

### 49.3 安全与降级

```text
学生账号访问教师统计接口是否返回 403？
Semantic Scholar 或 NVD 不可用时是否显示缓存或本地结果？
外部结果是否带来源、获取时间和缓存状态？
上传资料中的提示注入是否会触发未授权工具？
日志、配置页、CSV 和错误响应是否泄露密钥或令牌？
```

---

## 50. 第三阶段报告总结与创新点写法

```text
本项目在本地 MCP-RAG 教学智能体的基础上，进一步构建了覆盖学生学习、实践指导、创新启发和教师教学分析的完整课程智能体平台。
系统通过统一 Connector 与 MCP 工具接入论文、漏洞、标准和开源项目等多源资料，并采用缓存、来源标注、引用校验和本地降级机制保证外部服务不稳定时仍可使用。
在学生端，系统将课程资料检索、结构化实践指导、实验进度和连续创新启发连接为完整学习链路；在教师端，系统通过真实身份、学习事件、语义问题聚类和实验卡点分析形成可解释的教学反馈闭环。
系统使用 DeepSeek 驱动的有限自主 Agent Loop 动态选择检索、MCP 工具、追问、回答或拒答，并通过 Action Guard、Citation Guard、Safety Guard、角色权限和审计日志控制工具风险与大模型幻觉。
最终系统实现了“学生目标 -> 多源证据 -> 智能体决策 -> 实践与创新指导 -> 学习轨迹 -> 教师分析 -> 课程优化”的闭环，完整覆盖课程项目要求。
```
