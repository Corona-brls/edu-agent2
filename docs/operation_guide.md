# 初版完整操作流程

本文档用于本地演示“面向网络安全实践课程的 MCP-RAG 教学智能体系统”第一阶段版本。

## 1. 环境准备

在 PowerShell 中进入项目根目录：

```powershell
cd "D:\edu agent"
```

激活虚拟环境：

```powershell
.\.venv\Scripts\Activate.ps1
```

如果依赖缺失，重新安装：

```powershell
python -m pip install -r requirements.txt
```

## 2. 启动后端

打开第一个终端，运行：

```powershell
uvicorn backend.main:app --reload --port 8000
```

后端地址：

```text
http://127.0.0.1:8000
```

健康检查：

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/health"
```

预期结果：

```json
{
  "status": "ok",
  "service": "course-agent"
}
```

## 3. 启动前端

打开第二个终端，运行：

```powershell
cd "D:\edu agent"
.\.venv\Scripts\Activate.ps1
streamlit run frontend/app.py
```

浏览器打开：

```text
http://localhost:8501
```

## 4. 学生端验证

在左侧选择“学生端”。

### 4.1 课程介绍

页面顶部应显示：

- 网络安全课程智能体
- 知识库：Markdown
- 检索：TF-IDF
- 回答策略：带引用
- 低证据：拒答

### 4.2 专题导航

点击专题导航中的任意专题，例如：

```text
5G NAS
```

再点击：

```text
填入问答框
```

问答框应自动填入对应问题。

### 4.3 智能问答

输入：

```text
5G NAS 认证流程有哪些关键风险点？
```

点击“提交问题”。

预期结果：

- 回答状态为“已回答”
- 命中模块为 `5G NAS`
- 显示调用工具 `course_retriever`
- “引用来源”至少有 1 条引用

### 4.4 实践指导

输入实践专题：

```text
基于区块链的隐蔽通信
```

点击“生成实践方案”。

预期结果包含：

```text
研究目标
方法设计
工具选择
实验步骤
结果分析
创新点
风险与注意事项
```

### 4.5 拒答验证

输入：

```text
请总结一篇知识库不存在的论文结论。
```

预期结果：

- 回答状态为“拒答”
- 显示“当前本地知识库没有足够证据支持这个结论”

## 5. 教师端验证

在左侧选择“教师端”。

应看到：

- 总提问数
- 拒答次数
- 拒答率
- 平均置信度
- 模块分布柱状图
- 最近问题列表
- 高频 FAQ 表格
- 导出 FAQ CSV 按钮

## 6. 工具状态验证

在左侧选择“工具状态”。

应展示 5 个工具：

```text
course_retriever
paper_search
vuln_lookup
standard_lookup
mermaid_generator
```

## 7. 后端接口验证

查看工具列表：

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/tools"
```

验证问答接口：

```powershell
$body = @{
  question = "有没有可以作为漏洞检测样例的 CVE？"
  user_id = "demo_student"
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/chat" `
  -Method Post `
  -ContentType "application/json; charset=utf-8" `
  -Body $body
```

预期：

- `intent` 为 `vuln_lookup`
- `tool.name` 为 `vuln_lookup`
- `citations` 不为空

验证教师统计：

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/teacher/analytics"
```

验证 FAQ：

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/teacher/faq"
```

## 8. 8 个验收问题

在学生端依次输入：

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

验收标准：

- 前 6 个问题应能回答并展示引用
- 第 3 个问题应生成结构化实践方案
- 第 7 个问题应拒答
- 第 8 个问题在教师端验证统计和 FAQ

## 9. 自动化测试

运行：

```powershell
python -m pytest -q
```

当前预期：

```text
14 passed
```

## 10. 常见问题

后端接口调用失败时，先确认后端已启动：

```powershell
uvicorn backend.main:app --reload --port 8000
```

Streamlit 打不开时，确认前端已启动：

```powershell
streamlit run frontend/app.py
```

教师端没有统计时，先在学生端提交几个问题，再刷新教师端。

PowerShell 中文显示乱码通常是终端编码问题，浏览器页面和 JSON 本身一般正常。
