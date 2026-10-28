"""Standalone EmoFlow analysis for the installable Harness bundle."""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime

from emotion_adaptive_agent import EmotionAdaptiveAgent
from service_decision import ServiceMemory, decide_service


def analyze(text: str, history: list[str]) -> dict:
    if not isinstance(text, str) or not text.strip() or len(text) > 2000:
        raise ValueError("text must contain 1–2000 characters")
    if not isinstance(history, list) or len(history) > 4:
        raise ValueError("recent_user_messages must contain at most four messages")
    if any(not isinstance(item, str) or not item.strip() or len(item) > 2000 for item in history):
        raise ValueError("invalid history message")

    agent = EmotionAdaptiveAgent()
    for previous in history:
        agent.process(previous)
    result = agent.process(text)
    perception = result["layer1_perception"]
    reasoning = result["layer2_reasoning"]
    strategy = result["layer3_strategy"]

    memory = ServiceMemory()
    now = time.time()
    hour = datetime.now().hour
    for index, previous in enumerate(history):
        previous_time = now - (len(history) - index) * 60
        decide_service(previous, memory, now=previous_time, hour=hour)
    choice = decide_service(
        text, memory, primary=perception["primary"],
        arousal=perception["arousal"], now=now, hour=hour,
    ).as_dict()
    guidance = (
        "先回应用户当下感受，再直接给出一个可选、具体的轻量建议；不要连续提问或替用户执行服务。"
        if choice["mode"] == "offer"
        else "先贴近用户当前表达，简短陪伴；不主动给建议，必要时只问一个与当前感受有关的问题。"
    )
    if choice["no_questions"]:
        guidance += "用户明确不希望被提问，本轮不要提问。"

    return {
        "engine": "EmoFlow local four-layer engine",
        "emotion": {key: perception[key] for key in
                    ("primary", "confidence", "intensity", "valence", "arousal", "dominance", "source")},
        "trajectory": {key: reasoning[key] for key in
                       ("trend", "valence_slope", "dominant_pattern", "escalation_risk", "inferred_need")},
        "strategy": {key: strategy[key] for key in
                     ("name", "priority", "tone", "pace", "do_first", "avoid")},
        "reasoning_details": {
            "decayed_valence": reasoning.get("decayed", {}).get("valence"),
            "tom_need": reasoning.get("tom_need"),
            "tom_belief": reasoning.get("tom_belief"),
        },
        "decision_source": strategy.get("selection", {}).get("source", "local_rule"),
        "learning": {"enabled": False, "reason": "Q 学习模块未接入此插件"},
        "service_decision": choice,
        "reply_guidance": guidance,
        "limits": "情绪结果为演示性启发式判断，不是诊断。",
    }


def main() -> None:
    request = json.load(sys.stdin)
    result = analyze(request["text"], request.get("recent_user_messages") or [])
    json.dump(result, sys.stdout, ensure_ascii=False, allow_nan=False)


if __name__ == "__main__":
    main()
