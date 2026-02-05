import json
from openai import OpenAI
from ..core.config import settings
from ..core.s3 import presigned_get_url
from .schema import ChartFacts, VisionResult, Mode
from .prompt import SYSTEM, user_prompt

client = OpenAI(api_key=settings.openai_api_key)

def analyze_image_key(image_id: str, s3_key: str, mode: Mode) -> VisionResult:
    img_url = presigned_get_url(s3_key, expires_sec=600)

    resp = client.responses.create(
        model=settings.vision_model,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": SYSTEM + "\n" + user_prompt(mode)},
                {"type": "input_image", "image_url": img_url, "detail": "high"},
            ],
        }],
        # If your model supports strict structured outputs in your setup,
        # you can enforce JSON more strongly; for now we parse defensively.
    )

    text = resp.output_text.strip()
    data = json.loads(text)
    facts = ChartFacts.model_validate(data)

    # simple gating
    keep_looking = facts.confidence < 0.55 or not facts.symbol or not facts.timeframe
    narration = build_narration(facts, mode, keep_looking)

    return VisionResult(
        image_id=image_id,
        mode=mode,
        chart_facts=facts,
        narration_text=narration,
        keep_looking=keep_looking,
    )

def build_narration(f: ChartFacts, mode: Mode, keep: bool) -> str:
    if keep:
        return "I can’t read enough from this screenshot with confidence. Keep looking."

    core = f"{f.symbol} on {f.timeframe}. Trend looks {f.trend}. "
    lvl = f"Support {f.levels.support[:2]} and resistance {f.levels.resistance[:2]}. "
    extra = " ".join(f.notes[:3])

    if mode == "f10":
        return core + lvl + "Momentum focus: watch the next resistance break; invalidate on support loss. " + extra
    if mode == "f9":
        return core + lvl + extra
    return core + lvl + "Patterns: " + ", ".join(f.patterns[:3]) + ". " + extra
