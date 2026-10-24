# 第二阶段测试记录

测试环境：

- 操作系统：Windows
- 项目目录：`C:\Users\Asd\Desktop\edu agent`
- Python 虚拟环境：`.venv`
- 默认模式：TemplateProvider 离线模式，未使用真实 DeepSeek 费用

## 单元与集成测试

命令：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

结果：

```text
116 passed, 12 warnings
```

说明：warnings 来自第三方依赖的 deprecation 提示，不影响当前验收。

## Agent 行为验收

命令：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_agent_orchestration.py tests\test_agent_resume.py tests\test_action_guard.py -q
```

结果：

```text
14 passed
```

覆盖：

- 简单问题一次检索后回答
- 复杂任务连续调用两个不同工具
- 信息不足时 `ask_user`
- 补充信息后继续同一个 `run_id`
- 检索失败、模型失败后的降级
- 非法工具被 Action Guard 拦截
- 证据不足时拒答
- 达到最大步骤时强制停止

## 离线冒烟

命令：

```powershell
.\.venv\Scripts\python.exe scripts\smoke_phase2.py
```

结果：

```text
phase2 smoke ok: provider=template, model=local-template, run_status=COMPLETED
```

## 离线评测

命令：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_phase2.py --output docs\phase2_eval_results.json
```

结果：

```text
case_count: 40
Recall@K: 0.4118
Citation Accuracy: 1.0000
Groundedness: 1.0000
Abstention Accuracy: 0.7250
Intent Accuracy: 0.8500
Action Accuracy: 0.7000
Tool Routing Accuracy: 0.8750
```

## DeepSeek 模式待执行

由本机用户在 `.env` 填写 `DEEPSEEK_API_KEY` 后执行：

```powershell
.\.venv\Scripts\python.exe scripts\smoke_phase2.py --allow-deepseek
.\.venv\Scripts\python.exe scripts\evaluate_phase2.py --allow-deepseek --output docs\phase2_eval_results_deepseek.json
```

注意：报告、截图和提交内容不得包含 API Key。
