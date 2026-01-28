import ollama


class VoiceEditor:
    def __init__(self, model: str = "qwen2.5:3b"):
        self.model = model

    def apply_edit(self, original: str, instruction: str) -> str:
        """根据语音指令编辑文本"""
        prompt = f"""原文：{original}

用户指令：{instruction}

请根据指令修改原文，只返回修改后的文本，不要解释。"""

        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"].strip()
