import re
from typing import Optional

from cog import BasePredictor, BaseModel, Input
from transformers import AutoModelForCausalLM, AutoTokenizer


class ModerationOutput(BaseModel):
    safety_label: str
    categories: str
    refusal: Optional[str] = None


class Predictor(BasePredictor):
    def setup(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained("./weights")
        self.model = AutoModelForCausalLM.from_pretrained(
            "./weights",
            torch_dtype="auto",
            device_map="auto",
        )

    def predict(
        self,
        prompt: str = Input(description="User message to moderate"),
        response: Optional[str] = Input(
            description="Assistant response to moderate (enables response moderation)",
            default=None,
        ),
        max_new_tokens: int = Input(
            description="Maximum number of tokens to generate",
            ge=1,
            le=256,
            default=128,
        ),
    ) -> ModerationOutput:
        messages = [{"role": "user", "content": prompt}]
        if response is not None:
            messages.append({"role": "assistant", "content": response})

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        raw = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()

        safety_match = re.search(r"Safety:\s*(.+)", raw)
        categories_match = re.search(r"Categories:\s*(.+)", raw)
        refusal_match = re.search(r"Refusal:\s*(.+)", raw)

        return ModerationOutput(
            safety_label=safety_match.group(1).strip() if safety_match else raw,
            categories=categories_match.group(1).strip() if categories_match else "None",
            refusal=refusal_match.group(1).strip() if refusal_match else None,
        )
