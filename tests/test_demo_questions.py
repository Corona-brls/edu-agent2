import pytest


DEMO_KNOWLEDGE_FILES = {
    "course_intro.md": """# 课程智能体
课程智能体使用本地知识检索、引用和证据不足时拒答，帮助学生理解课程概念并设计安全实验。
""",
    "5g_nas.md": """# 5G NAS 认证与安全
5G NAS 认证流程需要关注身份校验、重放攻击、密钥协商、完整性保护和异常状态迁移风险。
实验应比较正常认证与重放样本，并记录检测率、误报率和处理延迟。
""",
    "blockchain_covert.md": """# 区块链隐蔽通信
区块链隐蔽通信教学实验只使用本地模拟交易，比较地址、金额和交易频率特征，不连接真实资产网络。
""",
    "llm_vuln_detection.md": """# LLM 漏洞检测
LLM 漏洞检测应核对漏洞报告、代码证据和测试结果，避免把模型推测当成已经验证的漏洞结论。
""",
    "cve_samples.md": """# CVE 教学样例
CVE 样例用于学习漏洞编号、影响组件、版本范围和缓解措施的核验方法，不代表实时漏洞数据库结果。
""",
    "standards_samples.md": """# 标准文档样例
3GPP TS 24.501 的 NAS 安全内容应按版本定位流程和条款；本地样例不能替代正式标准原文。
""",
}


DEMO_QUESTIONS = [
    "这个课程智能体能帮我做什么？",
    "5G NAS 认证流程有哪些关键风险点？",
    "我想做基于区块链的隐蔽通信，怎么设计实验？",
    "帮我找和 LLM 漏洞检测相关的研究方向。",
    "有没有可以作为漏洞检测样例的 CVE？",
    "3GPP TS 24.501 里 NAS 安全相关内容在哪里？",
    "请总结一篇知识库不存在的论文结论。",
    "连续提问后，教师端是否能看到高频问题和模块统计？",
]


@pytest.fixture
def demo_knowledge_dir(tmp_path):
    for filename, content in DEMO_KNOWLEDGE_FILES.items():
        (tmp_path / filename).write_text(content, encoding="utf-8")
    return tmp_path


def test_demo_questions_are_defined():
    assert len(DEMO_QUESTIONS) == 8


def test_local_knowledge_base_loads(demo_knowledge_dir):
    from backend.rag.loader import load_knowledge_chunks

    chunks = load_knowledge_chunks(demo_knowledge_dir)

    assert len(chunks) >= 6
    assert any(chunk.source == "5g_nas.md" for chunk in chunks)


def test_txt_knowledge_files_load(tmp_path):
    from backend.rag.loader import load_knowledge_chunks

    txt_file = tmp_path / "extra.txt"
    txt_file.write_text("# TXT 资料\n这是一个 TXT 知识库样例。", encoding="utf-8")

    chunks = load_knowledge_chunks(tmp_path)

    assert len(chunks) == 1
    assert chunks[0].source == "extra.txt"


def test_nas_question_returns_citations(demo_knowledge_dir):
    from backend.rag.answerer import answer_question

    result = answer_question("5G NAS 认证流程有哪些关键风险点？", demo_knowledge_dir)

    assert result["abstained"] is False
    assert result["module"] == "5G NAS"
    assert result["citations"]
    assert {"answer", "citations", "confidence", "abstained", "module", "intent"} <= set(result)


def test_missing_paper_conclusion_abstains(demo_knowledge_dir):
    from backend.rag.answerer import answer_question

    result = answer_question("请总结一篇知识库不存在的论文结论。", demo_knowledge_dir)

    assert result["abstained"] is True


def test_unrelated_question_abstains(demo_knowledge_dir):
    from backend.rag.answerer import answer_question

    result = answer_question("完全无关的问题：今天上海天气如何？", demo_knowledge_dir)

    assert result["abstained"] is True
