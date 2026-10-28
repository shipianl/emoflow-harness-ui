"""Deterministic service-intent gate shared by the cloud A2A agent.

Emotion answers how the user may feel.  This module separately decides whether
the current message asks for a service, asks for advice, or asks the agent to
stay in companion mode.  Explicit current-turn intent wins over older memory.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


VALID_SERVICES = {"music", "walk", "quiet"}
PAUSE_SECONDS = 30 * 60
OFFER_COOLDOWN_SECONDS = 10 * 60


@dataclass
class ServiceMemory:
    enabled: bool = True
    paused_until: float = 0.0
    last_offer: float = 0.0
    preferred: str = ""
    blocked: set[str] = field(default_factory=set)
    rejected: str = ""
    rejected_until: float = 0.0


@dataclass(frozen=True)
class ServiceChoice:
    mode: str = "companion"
    service: str = ""
    reason: str = "需求不明确，保持陪伴"
    intent: str = "uncertain"
    no_questions: bool = False

    def as_dict(self):
        return {
            "mode": self.mode,
            "service": self.service,
            "reason": self.reason,
            "intent": self.intent,
            "no_questions": self.no_questions,
        }


def _target_service(text):
    if re.search(r"音乐|听歌|听.{0,3}歌|首歌|歌曲|歌单", text):
        return "music"
    if re.search(r"散步|走走|出去走|走一会", text):
        return "walk"
    if re.search(r"安静|独处|静一静|暂停", text):
        return "quiet"
    return ""


def _default_service(primary, arousal, text):
    if primary in {"sadness", "disappointment"}:
        return "music"
    if primary in {"anxiety", "anger", "frustration"} or arousal >= 0.6:
        return "quiet"
    if re.search(r"开会|上课|开车|没空|下雨|不方便出门|深夜|半夜", text):
        return "quiet"
    return "walk"


def _walk_is_unsuitable(text, hour):
    return hour < 7 or hour >= 21 or bool(
        re.search(r"开会|上课|开车|没空|下雨|不方便出门|不能出门|深夜|半夜", text)
    )


def decide_service(text, memory, primary="neutral", arousal=0.0, now=0.0, hour=12):
    """Choose one service or companion mode from current-turn intent.

    Rules deliberately use only explicit phrases from the current message.
    Emotion can choose between safe low-burden services after an advice request,
    but emotion alone never creates a service offer.
    """
    text = (text or "").strip()
    no_questions = bool(re.search(r"别.{0,6}(问|提问)|不要.{0,6}(问|提问)|不想.{0,6}(回答|被问)", text))
    reject = bool(re.search(
        r"(不要|不想|不用|别|不需要|不喜欢).{0,10}(建议|推荐|音乐|听歌|散步|走走|打扰)", text
    ))
    vent = bool(re.search(r"只想.{0,6}(聊|倾诉|说说)|听我说|陪我一会|陪着我", text))

    target = _target_service(text)
    request = bool(target and re.search(
        r"想听|听首|放首|播放|打开|想去|想出去|想散步|想走走|让我安静|想安静|帮我放|带我", text
    ))
    advice = bool(re.search(
        r"(给我|帮我|来个|能不能|可以).{0,10}(建议|办法|方法)|"
        r"有什么.{0,8}(建议|办法|方法)|怎么.{0,8}(放松|缓解|调整)|建议吗", text
    ))

    if reject:
        memory.paused_until = max(memory.paused_until, now + PAUSE_SECONDS)
        if target:
            memory.rejected = target
            memory.rejected_until = now + PAUSE_SECONDS
        return ServiceChoice(reason="用户明确拒绝建议，进入陪伴模式", intent="decline", no_questions=no_questions)

    if request or advice:
        # A new explicit request is the user's latest preference and therefore
        # overrides an older pause or cooldown.
        memory.paused_until = 0.0
        if not target:
            target = _default_service(primary, arousal, text)
        if target == "walk" and _walk_is_unsuitable(text, hour):
            target = "quiet"
        memory.last_offer = now
        return ServiceChoice(
            mode="offer",
            service=target,
            reason="用户本轮明确请求服务" if request else "用户本轮明确寻求建议",
            intent="request" if request else "advice",
            no_questions=no_questions,
        )

    if vent:
        memory.paused_until = max(memory.paused_until, now + PAUSE_SECONDS)
        return ServiceChoice(reason="用户希望倾诉或安静陪伴", intent="vent", no_questions=no_questions)

    if not memory.enabled or now < memory.paused_until:
        return ServiceChoice(reason="主动建议处于暂停期", intent="uncertain", no_questions=no_questions)
    if memory.last_offer and now - memory.last_offer < OFFER_COOLDOWN_SECONDS:
        return ServiceChoice(reason="主动建议处于冷却期", intent="uncertain", no_questions=no_questions)
    return ServiceChoice(no_questions=no_questions)


def fallback_advice(service):
    if service == "music":
        return "可以先选一首你很熟悉、节奏不太强的歌，从头听完一遍，这几分钟先不处理别的事。"
    if service == "walk":
        return "如果现在环境安全方便，可以先走五分钟，把注意力放在脚步和周围的声音上。"
    return "先给自己两分钟暂停：把手机放下，坐稳一点，让呼气比平时慢一些。"


def ensure_explicit_advice(response, choice):
    """Guarantee that an explicit request is not turned back into questioning."""
    if choice.mode != "offer":
        return response
    response = (response or "").strip()
    refusal = re.search(
        r"先不急着给建议|不急着给建议|先不建议|不想给你建议|"
        r"建议.*没什么用|更想先知道|先聊聊再", response
    )
    actionable = re.search(r"可以|先|试试|打开|听|走|放下|坐|写|喝|离开|整理|停下|闭眼|呼吸|吸气|呼气", response)
    if not response or refusal or not actionable:
        return fallback_advice(choice.service)
    # A requested suggestion should end with the action itself.  Asking
    # whether the user wants to do it adds friction and feels formulaic.
    response = re.sub(
        r"(?:\n+|(?<=[。！!]))\s*(?:要不要|愿不愿意|想不想|需要我)[^？?]*[？?]\s*$",
        "",
        response,
    ).rstrip()
    response = re.sub(
        r"[,，]\s*(?:做完|试完|完成后|之后).{0,12}(?:告诉我|跟我说|和我说).*$",
        "。",
        response,
    ).rstrip()
    response = re.sub(
        r"(?:\n+|(?<=[。！!]))\s*(?:做完|试完|完成后|之后).{0,12}(?:告诉我|跟我说|和我说).*$",
        "",
        response,
    ).rstrip()
    return response or fallback_advice(choice.service)
