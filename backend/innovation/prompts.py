"""Prompt fragments and safety boundaries for innovation coaching."""

SAFETY_BOUNDARY = (
    "This course innovation coach supports defensive research, reproducible lab experiments, "
    "measurement design, and literature-grounded ideation. It must not provide public-target "
    "attack execution steps, exploit chaining instructions, stealth deployment guidance, or "
    "instructions to bypass authorization."
)

QUESTION_TEMPLATES = [
    "你的方案如何量化 {focus}，哪些指标可以证明改进不是偶然结果？",
    "如果把 {focus} 放到受控实验环境中，最小可复现实验应包含哪些对照组？",
    "当前方案在哪个假设上最脆弱，如何用课程知识库或公开元数据验证它？",
    "哪些检测、误报或合规边界会限制该方案落地？",
    "如果只能保留一个创新点，你会优先优化隐蔽性、可解释性、鲁棒性还是安全边界，为什么？",
]

MODEL_SUGGESTION_LABEL = "模型建议"
