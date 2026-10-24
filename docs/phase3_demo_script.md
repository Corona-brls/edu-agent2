# 第三阶段演示脚本

## 1. 初始化

```powershell
cd "D:\edu agent"
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-phase2.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-phase3.txt
Copy-Item .env.example .env
.\.venv\Scripts\python.exe -m backend.auth.seed
```

## 2. 验证

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pytest tests\test_phase3_e2e.py -q
```

可选真实 smoke：

```powershell
$env:RUN_LIVE_CONNECTOR_TESTS="true"
.\.venv\Scripts\python.exe -m pytest -m live -q
```

## 3. 启动

```powershell
.\scripts\start_backend.cmd
.\scripts\start_frontend.cmd
```

等价命令：

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload --port 8000
.\.venv\Scripts\python.exe -m streamlit run frontend/app.py
```

## 4. 学生演示

1. 使用 demo 学生登录。
2. 阅读“使用说明”和课程简介。
3. 在课程资料库搜索 `5G NAS`，打开一条论文或课程文档。
4. 在智能问答或实践指导中询问 5G NAS 本地实验设计。
5. 在实验任务中更新进度并上报卡点。
6. 进入创新启发页面，输入专题和当前方案，回答启发式问题并生成创新方向。

## 5. 教师演示

1. 使用 demo 教师登录。
2. 查看访问监控、资源搜索/打开、问答、实践指导、创新会话和实验进度。
3. 查看学生学习轨迹。
4. 重建问题聚类，重命名并批准热点问题。
5. 查看卡点分析并确认/忽略/解决卡点。
6. 导出 FAQ CSV，检查内容已做公式注入防护。
