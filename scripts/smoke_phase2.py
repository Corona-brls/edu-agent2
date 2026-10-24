"""Phase 2 smoke checks for TemplateProvider and optional DeepSeek mode."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-deepseek", action="store_true")
    args = parser.parse_args()

    if not args.allow_deepseek:
        os.environ["DEEPSEEK_API_KEY"] = ""

    from backend import main as api
    from backend.storage.logger import LearningLogger

    with tempfile.TemporaryDirectory(prefix="phase2_smoke_", ignore_cleanup_errors=True) as tmp:
        original_logger = api.learning_logger
        api.learning_logger = LearningLogger(Path(tmp) / "smoke.db")
        try:
            status = api.system_status()
            simple = _run_chat_smoke(api, allow_deepseek=args.allow_deepseek)
            run = api.get_agent_run(simple["run_id"])
            feedback = api.feedback(api.FeedbackRequest(question_id=simple["question_id"], rating=5, helpful=True))
            analytics = api.teacher_analytics()
        finally:
            api.learning_logger = original_logger

    assert status["provider"] in {"template", "deepseek"}
    assert simple["run_id"] and simple["trace_id"]
    if args.allow_deepseek:
        assert simple["run_status"] == "COMPLETED", simple
        assert simple["citations"], simple
    else:
        assert simple["citations"] or simple["abstained"]
    assert run["steps"]
    assert feedback["saved"] is True
    assert analytics["total_questions"] >= 1
    assert analytics["feedback_metrics"]["count"] >= 1
    print(
        "phase2 smoke ok:",
        {
            "provider": status["provider"],
            "model": status["model"],
            "run_status": simple["run_status"],
            "question_id": simple["question_id"],
            "steps": run["current_step"],
        },
    )


def _run_chat_smoke(api, *, allow_deepseek: bool):
    questions = [
        "5G NAS 认证流程有哪些关键风险点？",
        "请先检索课程知识库，再回答 5G NAS 认证流程有哪些关键风险点？",
        "课程知识库中 5G NAS 安全专题包含哪些风险？",
    ]
    last_result = None
    for question in questions if allow_deepseek else questions[:1]:
        result = api.chat(api.ChatRequest(question=question))
        last_result = result
        if result.get("run_status") == "COMPLETED" and result.get("citations"):
            return result
    return last_result


if __name__ == "__main__":
    main()
