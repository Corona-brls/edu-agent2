"""Evidence-grounded answer generation for the local demo RAG."""

from pathlib import Path

from backend.agent.router import route_intent
from backend.rag.retriever import LocalRetriever
from backend.tools.registry import select_tool_for_intent

ABSTAIN_MESSAGE = "当前本地知识库没有足够证据支持这个结论。建议补充相关资料后再查询。"


class CourseAnswerer:
    def __init__(self, base_dir: str | Path | None = None) -> None:
        self.retriever = LocalRetriever(base_dir)

    def answer(self, question: str, top_k: int = 3) -> dict:
        intent = route_intent(question)
        tool = select_tool_for_intent(intent)
        results = self.retriever.search(question, top_k=top_k)
        confidence = _confidence(results)
        should_abstain = (
            confidence < 0.35
            or not _has_enough_evidence(results)
            or _requires_strict_source(question, results)
        )

        if should_abstain:
            return {
                "answer": ABSTAIN_MESSAGE,
                "citations": results[:1],
                "confidence": min(confidence, 0.17),
                "abstained": True,
                "module": _module(results),
                "intent": intent,
                "tool": tool,
            }

        return {
            "answer": _compose_answer(question, intent, results),
            "citations": results,
            "confidence": confidence,
            "abstained": False,
            "module": _module(results),
            "intent": intent,
            "tool": tool,
        }

    def practice_guide(self, topic: str) -> dict:
        tool = select_tool_for_intent("practice_guide")
        results = self.retriever.search(topic, top_k=3)
        confidence = _confidence(results)
        module = _module(results)
        evidence = results[0]["snippet"] if results else "当前仅能基于课程通用实践流程生成初步方案。"
        return {
            "answer": _compose_practice_guide(topic, module, evidence),
            "citations": results,
            "confidence": confidence,
            "abstained": False,
            "module": module,
            "intent": "practice_guide",
            "tool": tool,
        }


def answer_question(question: str, base_dir: str | Path | None = None) -> dict:
    return CourseAnswerer(base_dir).answer(question)


def abstain_answer() -> str:
    return ABSTAIN_MESSAGE


def _confidence(results: list[dict]) -> float:
    if not results:
        return 0.0
    top_score = float(results[0]["score"])
    top_tfidf = float(results[0].get("tfidf_score", top_score))
    support_bonus = min(len(results), 3) * 0.03
    confidence = min(0.95, top_tfidf * 1.45 + support_bonus)
    return round(confidence, 2)


def _module(results: list[dict]) -> str:
    if not results:
        return "unknown"
    return str(results[0].get("module") or "课程资料")


def _requires_strict_source(question: str, results: list[dict]) -> bool:
    if "不存在" in question and ("论文" in question or "结论" in question):
        return True
    if "论文结论" in question or "具体论文结论" in question:
        return True
    strict_terms = ("精确条款", "编造")
    if not any(term in question for term in strict_terms):
        return False
    if not results:
        return True
    return float(results[0]["score"]) < 0.35


def _has_enough_evidence(results: list[dict]) -> bool:
    if not results:
        return False
    top = results[0]
    top_tfidf = float(top.get("tfidf_score", top["score"]))
    top_overlap = float(top.get("token_overlap", 0.0))
    return top_tfidf >= 0.08 and top_overlap >= 0.2


def _compose_answer(question: str, intent: str, results: list[dict]) -> str:
    module = _module(results)
    snippets = [item["snippet"] for item in results[:3]]
    knowledge = "；".join(snippets)
    guide = _guide_for(module, intent)
    extension = _extension_for(intent)
    return (
        f"知识定位：这个问题命中“{module}”模块，意图识别为 {intent}。\n\n"
        f"知识讲解：根据本地知识库，{knowledge}\n\n"
        f"实践指导：{guide}\n\n"
        f"拓展建议：{extension}\n\n"
        "引用来源：见 citations 字段中的来源文件、标题和片段。"
    )


def _guide_for(module: str, intent: str) -> str:
    if intent == "vuln_lookup":
        return "先核对漏洞编号、影响组件和验证思路，再判断是否适合作为课程实验样例。"
    if intent == "standard_lookup":
        return "先定位标准相关流程，再结合本地片段判断是否能支持具体结论。"
    if module == "5G NAS":
        return "可以从认证流程、加密协商、完整性保护和异常状态迁移四个角度展开实验。"
    if module == "区块链隐蔽通信":
        return "可以构造本地模拟交易，比较普通交易和隐蔽通信交易在地址、金额、频率上的差异。"
    if module == "LLM 漏洞检测":
        return "可以围绕漏洞报告核验、PoC 一致性检查和证据引用设计评估流程。"
    return "建议先明确研究问题，再整理资料证据、实验步骤和评价指标。"


def _extension_for(intent: str) -> str:
    if intent == "paper_search":
        return "第一阶段只检索本地样例，不连接真实论文库；如果需要精确论文结论，应补充论文原文。"
    if intent == "vuln_lookup":
        return "第一阶段不连接真实 CVE/NVD 数据库；没有本地证据时不应编造漏洞编号。"
    if intent == "standard_lookup":
        return "第一阶段只提供标准样例片段；没有具体条款时应拒绝给出精确条款结论。"
    return "后续可以把本地 TF-IDF 替换为向量数据库，并接入标准 MCP 工具。"


def _compose_practice_guide(topic: str, module: str, evidence: str) -> str:
    return (
        f"研究目标：围绕“{topic}”明确一个可验证的安全问题，结合“{module}”模块资料形成实验目标。\n\n"
        f"方法设计：先梳理概念和威胁模型，再基于本地样例设计对照实验。参考证据：{evidence}\n\n"
        "工具选择：第一阶段优先使用本地 Markdown 知识库、Python 脚本、日志记录和可视化表格。\n\n"
        "实验步骤：1. 收集资料；2. 定义输入输出；3. 构造正常样例和异常样例；4. 记录结果；5. 对比分析。\n\n"
        "结果分析：从准确性、可解释性、复现性和风险控制四个维度整理实验结果。\n\n"
        "创新点：突出课程智能体如何把检索、引用、拒答和教师分析结合成学习闭环。\n\n"
        "风险与注意事项：不要编造论文、标准条款或 CVE；所有结论尽量给出本地证据来源。"
    )
