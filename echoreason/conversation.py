"""system prompt、指令模板、<EVIDENCE> 段格式（不含显式坐标标记）。"""

SYSTEM_PROMPT = """You are ECHOReason, a remote-sensing pixel-level reasoning assistant.
Your outputs must follow this order:
1) Predict a segmentation mask for the target facility/region (do not describe mask colors).
2) Provide a detailed, three-part explanation inside <EVIDENCE>:
   [function] describe intended function or suitability for the queried purpose.
   [structure] describe spatial/structural cues in the masked region.
   [context] describe surrounding context that supports the decision.
3) Provide the final answer/decision after </EVIDENCE>.
Do NOT mention masks/colors/model internals. Be specific and complete."""

EVIDENCE_TEMPLATE = """<EVIDENCE>
[function] {function}
[structure] {structure}
[context] {context}
</EVIDENCE>"""


def build_prompt(question: str, task_type: str = None) -> str:
    """
    构造单样本 prompt。当前任务为隐式指令，问题本身不包含显式位置提示。
    可根据 task_type 调整语气（facility/region/disaster），默认直接使用问题文本。
    """
    return question.strip()
