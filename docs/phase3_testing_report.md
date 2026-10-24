# 第三阶段测试报告

## 回归结果

本轮第三阶段第十二步新增离线端到端验收 `tests/test_phase3_e2e.py`，并保留第二阶段与第三阶段既有测试。

| 验证项 | 命令 | 结果 |
|---|---|---|
| 第三阶段 E2E | `.\.venv\Scripts\python.exe -m pytest tests\test_phase3_e2e.py -q` | `1 passed, 1 warning` |
| 完整回归 | `.\.venv\Scripts\python.exe -m pytest -q` | `187 passed, 2 skipped, 12 warnings` |
| Live smoke | `.\.venv\Scripts\python.exe -m pytest -m live -q` | `2 skipped, 187 deselected, 1 warning`，默认不访问公网 |

## E2E 覆盖

- demo 学生与教师登录。
- 课程与班级读取。
- mock 外部论文、NVD 漏洞、标准元数据和开源项目缓存展示。
- 课程资料检索与打开事件记录。
- 本地课程正文引用通过实践指导接口返回。
- 实验任务进度更新与卡点上报。
- 创新启发会话、学生回答与最终创新方向。
- 学生问题日志、语义聚类、教师重命名与批准。
- FAQ CSV 导出。
- 教师端摘要、学生学习轨迹、资源统计、问答统计、创新统计和卡点统计。
- 学生访问 Connector 同步、教师统计和知识库删除时被拒绝。

## Live Smoke 说明

普通 `pytest` 不访问公网，不产生 DeepSeek 或外部 Connector 费用。真实 DeepSeek、论文源和 NVD 验证使用单独 live smoke：

```powershell
$env:RUN_LIVE_CONNECTOR_TESTS="true"
.\.venv\Scripts\python.exe -m pytest -m live -q
```

如果外部服务限流、超时或网络不可用，测试会跳过并记录原因；不影响离线回归。
