# -*- coding: utf-8 -*-
"""
基于情绪感知的 Agent 输出优化 Demo
=====================================
四层架构: 情绪感知 → 情绪推理 → 策略适配 → 生成优化
核心技术: 情感条件解码 (Emotion-Conditioned Decoding) + 情绪轨迹追踪

运行方式:
  python emotion_adaptive_agent.py          # 交互模式
  python emotion_adaptive_agent.py --demo   # 演示模式（预设对话）
"""

import argparse
import json
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

# ---------------------------------------------------------------------------
# Layer 1: 情绪感知层
# ---------------------------------------------------------------------------


class Emotion(Enum):
    """GoEmotions 精简映射"""
    ANGER = "anger"
    FRUSTRATION = "frustration"
    ANXIETY = "anxiety"
    SADNESS = "sadness"
    CONFUSION = "confusion"
    DISAPPOINTMENT = "disappointment"
    NEUTRAL = "neutral"
    SATISFACTION = "satisfaction"
    GRATITUDE = "gratitude"
    EXCITEMENT = "excitement"


@dataclass
class EmotionSignal:
    """情绪感知层输出"""
    primary: Emotion
    confidence: float       # 0.0 ~ 1.0
    intensity: float        # 0.0 ~ 1.0
    valence: float          # -1.0 (消极) ~ +1.0 (积极)
    arousal: float          # 0.0 (平静) ~ 1.0 (激动)
    dominance: float = 0.0  # -1.0 (顺从/被动) ~ +1.0 (支配/主动) — PAD 第三维 (SELAgents)
    secondary: list[tuple[Emotion, float]] = field(default_factory=list)
    raw_text: str = ""
    source: str = "local"   # "local" (关键词/本地模型) | "deepseek" (LLM 感知)

    @property
    def is_negative(self) -> bool:
        return self.valence < -0.15

    @property
    def is_high_arousal(self) -> bool:
        return self.arousal > 0.6

    @property
    def is_high_dominance(self) -> bool:
        return self.dominance > 0.4

    @property
    def is_low_dominance(self) -> bool:
        return self.dominance < -0.4


class EmotionPerceptionLayer:
    """
    情绪感知层（本地快速路径）
    基于关键词 + 规则的轻量分类器（生产环境可替换为 BERT/LLM/本地小模型）
    输出: EmotionSignal(标签, 置信度, 强度, valence, arousal, dominance)

    相对 v3 的改进:
      - 字符二元组模糊匹配: 处理错别字/口语改写（ELMO 论文的 OOV 教训）
      - 置信度校准: 基于命中证据量，低置信度触发上层走 LLM 慢通道
      - PAD 第三维 dominance 估计（SELAgents）
    """

    # (情绪, valence, arousal, 关键词权重列表)
    EMOTION_PATTERNS: list[tuple[Emotion, float, float, list[tuple[str, float]]]] = [
        (Emotion.ANGER, -0.7, 0.85, [
            ("气死", 1.0), ("愤怒", 1.0), ("火大", 0.95), ("可恶", 0.85),
            ("垃圾", 0.8), ("sb", 0.9), ("傻逼", 0.9), ("妈的", 0.85),
            ("无语", 0.6), ("够了", 0.7), ("滚", 0.9),
        ]),
        (Emotion.FRUSTRATION, -0.5, 0.7, [
            ("烦", 0.85), ("崩溃", 0.9), ("受不了", 0.85), ("怎么又", 0.8),
            ("老是", 0.7), ("总是不行", 0.85), ("试了好多次", 0.8),
            ("搞不定", 0.85), ("卡住", 0.75), ("折腾", 0.7),
        ]),
        (Emotion.ANXIETY, -0.4, 0.65, [
            ("担心", 0.85), ("害怕", 0.9), ("紧张", 0.8), ("焦虑", 1.0),
            ("万一", 0.75), ("会不会", 0.7), ("不确定", 0.7),
            ("心里没底", 0.85), ("慌", 0.8), ("不安", 0.85),
        ]),
        (Emotion.SADNESS, -0.6, 0.25, [
            ("难过", 0.9), ("伤心", 0.9), ("哭", 0.85), ("失落", 0.8),
            ("孤独", 0.85), ("想哭", 0.9),
            ("不开心", 0.75), ("心累", 0.8), ("没意思", 0.7),
        ]),
        (Emotion.CONFUSION, -0.1, 0.4, [
            ("不懂", 0.85), ("不明白", 0.85), ("什么意思", 0.8),
            ("搞不懂", 0.85), ("怎么回事", 0.75), ("看不懂", 0.8),
            ("蒙", 0.7), ("混乱", 0.7), ("？？", 0.85),
        ]),
        (Emotion.DISAPPOINTMENT, -0.35, 0.3, [
            ("怎么这样", 0.8), ("算了", 0.65), ("差太多了", 0.8),
            ("白费", 0.8), ("还不如", 0.7), ("就这", 0.65),
            ("本以为", 0.7), ("期待落空", 0.9),
        ]),
        (Emotion.GRATITUDE, 0.65, 0.3, [
            ("谢谢", 0.85), ("感谢", 0.85), ("太好了", 0.7),
            ("帮大忙", 0.85), ("多亏", 0.8), ("感激", 0.9),
        ]),
        (Emotion.SATISFACTION, 0.55, 0.3, [
            ("不错", 0.7), ("满意", 0.85), ("搞定", 0.65),
            ("解决了", 0.8), ("perfect", 0.7), ("nice", 0.65),
            ("很好", 0.7), ("很棒", 0.8),
        ]),
        (Emotion.EXCITEMENT, 0.7, 0.8, [
            ("太棒", 0.85), ("激动", 0.9), ("期待", 0.75),
            ("兴奋", 0.9), ("开心", 0.9), ("高兴", 0.85),
            ("厉害", 0.7), ("!!", 0.6),
        ]),
    ]

    INTENSITY_BOOSTERS = [
        ("真的", 0.1), ("太", 0.1), ("非常", 0.15), ("极其", 0.2),
        ("简直", 0.15), ("好", 0.08), ("特别", 0.12),
    ]
    INTENSITY_DAMPERS = [
        ("有点", -0.1), ("稍微", -0.15), ("可能", -0.1), ("也许", -0.1),
        ("不太", -0.1), ("还算", -0.08),
    ]

    # 否定前缀：出现在关键词前 2 个字符内 → 反向
    NEGATION_PREFIXES = ["不", "没", "别", "并不", "并没有", "不是"]

    # 模糊匹配: 关键词与文本窗口的字符二元组相似度阈值
    FUZZY_THRESHOLD = 0.75
    FUZZY_WEIGHT_PENALTY = 0.7   # 模糊命中的权重折扣

    # dominance 估计标记（SELAgents PAD 第三维: 支配/主动 vs 顺从/被动）
    DOMINANCE_HIGH_MARKERS = [
        ("我要", 0.35), ("必须", 0.35), ("一定", 0.3), ("立刻", 0.25),
        ("马上", 0.25), ("给我", 0.3), ("凭什么", 0.3), ("别跟我", 0.35),
        ("命令", 0.4), ("要求", 0.3), ("赶紧", 0.3), ("别废话", 0.4),
    ]
    DOMINANCE_LOW_MARKERS = [
        ("能不能", -0.35), ("帮我", -0.25), ("拜托", -0.35), ("请问", -0.25),
        ("不好意思", -0.3), ("求求", -0.4), ("不知道怎么办", -0.35),
        ("没底", -0.3), ("无助", -0.35), ("没办法", -0.3), ("依赖", -0.25),
    ]
    # 情绪类型对 dominance 的先验偏移
    DOMINANCE_EMOTION_PRIOR = {
        Emotion.ANGER: 0.25,
        Emotion.FRUSTRATION: 0.05,
        Emotion.ANXIETY: -0.2,
        Emotion.SADNESS: -0.15,
        Emotion.CONFUSION: -0.15,
        Emotion.DISAPPOINTMENT: -0.1,
    }

    @staticmethod
    def _char_bigram_sim(kw: str, text: str) -> float:
        """字符二元组相似度: 处理错别字/口语改写（如 '烦燥' vs '烦躁'、'气死了' vs '气死'）"""
        def bigrams(s: str) -> set[str]:
            if len(s) <= 1:
                return {s}
            return {s[i:i + 2] for i in range(len(s) - 1)}

        kw_b = bigrams(kw)
        if not kw_b:
            return 0.0
        best = 0.0
        step = max(1, len(kw) - 1)
        for i in range(0, max(1, len(text) - len(kw) + 1), step):
            win = text[i:i + len(kw)]
            w_b = bigrams(win)
            inter = len(kw_b & w_b)
            best = max(best, 2 * inter / max(1, len(kw_b) + len(w_b)))
        return best

    def _is_negated(self, text: str, keyword: str, pos: int) -> bool:
        """检查关键词在文本中是否有否定前缀"""
        for neg in self.NEGATION_PREFIXES:
            neg_start = pos - len(neg)
            if neg_start >= 0 and text[neg_start:pos] == neg:
                # 再往前看是否还有否定（处理双重否定："不是不开心" → 不算否定）
                for neg2 in self.NEGATION_PREFIXES:
                    dbl_start = neg_start - len(neg2)
                    if dbl_start >= 0 and text[dbl_start:neg_start] == neg2:
                        return False  # 双重否定 → 不算否定
                return True
        return False

    def perceive(self, text: str) -> EmotionSignal:
        """
        输入文本 → 输出情绪信号
        如果无法识别任何情绪，返回低置信 NEUTRAL（上层按置信度门控决定是否走 LLM 慢通道）
        """
        text_lower = text.lower()
        scores: dict[Emotion, float] = {}
        total_hits = 0

        for emotion, valence, arousal, patterns in self.EMOTION_PATTERNS:
            score = 0.0
            hit_count = 0
            for keyword, weight in patterns:
                pos = text_lower.find(keyword)
                appears = pos != -1
                while pos != -1:
                    if not self._is_negated(text_lower, keyword, pos):
                        score += weight
                        hit_count += 1
                        break  # 同一个关键词只计一次
                    # 否定匹配 → 跳过，继续往后找
                    pos = text_lower.find(keyword, pos + len(keyword))
                # 模糊匹配: 仅在关键词完全不出现时启用（出现但被否定 ≠ 未命中，
                # 否则"我不难过"会被模糊匹配重新命中"难过"）
                if hit_count == 0 and not appears and len(keyword) >= 2:
                    sim = self._char_bigram_sim(keyword, text_lower)
                    if sim >= self.FUZZY_THRESHOLD:
                        score += weight * self.FUZZY_WEIGHT_PENALTY
                        hit_count += 1
            if score > 0:
                # 多次命中提权
                if hit_count >= 2:
                    score *= 1.15
                scores[emotion] = min(score, 1.0)
                total_hits += hit_count

        if not scores:
            return EmotionSignal(
                primary=Emotion.NEUTRAL,
                confidence=0.4,     # 证据不足 → 低置信，上层门控触发 LLM 慢通道
                intensity=0.0,
                valence=0.0,
                arousal=0.15,
                dominance=0.0,
                raw_text=text,
            )

        # 强度修正：增强词/减弱词
        raw_intensity = max(scores.values())
        for word, delta in self.INTENSITY_BOOSTERS:
            if word in text_lower:
                raw_intensity += delta
        for word, delta in self.INTENSITY_DAMPERS:
            if word in text_lower:
                raw_intensity += delta
        intensity = max(0.1, min(1.0, raw_intensity))

        # 主情绪：得分最高
        primary = max(scores, key=scores.get)

        # 置信度校准: 基于命中证据量（替代 v3 的"占比式"置信度，
        # 后者单关键词弱命中也会给出接近 1.0 的置信度）
        max_score = scores[primary]
        confidence = min(0.95, 0.35 + 0.5 * max_score + 0.05 * max(0, total_hits - 1))

        # 次情绪：得分 > 0.3 的其他情绪
        total_score = sum(scores.values())
        secondary = [
            (e, s / total_score)
            for e, s in sorted(scores.items(), key=lambda x: -x[1])
            if e != primary and s > 0.3
        ][:2]

        # 从模板获取 valence / arousal
        base_valence = -0.1
        base_arousal = 0.2
        for e, v, a, _ in self.EMOTION_PATTERNS:
            if e == primary:
                base_valence = v
                base_arousal = a
                break

        # dominance 估计: 语言标记 + 情绪类型先验
        dom = 0.0
        for marker, delta in self.DOMINANCE_HIGH_MARKERS:
            if marker in text_lower:
                dom += delta
        for marker, delta in self.DOMINANCE_LOW_MARKERS:
            if marker in text_lower:
                dom += delta
        dom += self.DOMINANCE_EMOTION_PRIOR.get(primary, 0.0)
        dominance = max(-1.0, min(1.0, round(dom, 3)))

        return EmotionSignal(
            primary=primary,
            confidence=round(confidence, 3),
            intensity=round(intensity, 3),
            valence=round(base_valence * intensity, 3),
            arousal=round(base_arousal * intensity, 3),
            dominance=dominance,
            secondary=secondary,
            raw_text=text,
        )


# ---------------------------------------------------------------------------
# Layer 2: 情绪推理层
# ---------------------------------------------------------------------------


@dataclass
class EmotionSnapshot:
    """单轮情绪快照"""
    turn: int
    signal: EmotionSignal
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    escalation_risk: Optional[float] = None      # 该轮升级风险（reason 后回填，供前端轨迹图）
    decayed_valence: Optional[float] = None      # 该轮时刻的衰减后情绪状态（泄漏积分器）


@dataclass
class EmotionTrajectory:
    """
    情绪推理层输出 — 基于轨迹的推理
    """
    current: EmotionSignal
    history: list[EmotionSnapshot]
    trend: str                # "improving" | "worsening" | "stable" | "volatile"
    valence_slope: float      # 近 N 轮的 valence 变化斜率（时间衰减加权）
    arousal_slope: float      # 近 N 轮的 arousal 变化斜率（辅助：valence 不变但 arousal 飙升 = 恶化）
    dominant_pattern: str     # "acute_negative" | "chronic_low" | "positive" | "mixed"
    inferred_need: str        # 推理出的用户需求
    escalation_risk: float    # 0~1 是否需要升级处理
    needs_escalation: bool    # 升级风险超过阈值 → True（由 Layer 2 用自身阈值判定）
    summary: str              # 一句话总结
    # 泄漏积分器衰减后的情绪状态（SELAgents: dE/dt = -λE + g(S)，按真实时间指数衰减）
    decayed_valence: float = 0.0
    decayed_arousal: float = 0.0
    decayed_dominance: float = 0.0
    # 心智理论: 用户需求信念分布（贝叶斯更新）
    tom_need: str = ""        # 最大后验需求
    tom_belief: float = 0.5   # 该需求的信念强度（最大后验概率）
    tom_top_needs: list = field(default_factory=list)  # [(需求, 概率)] top-2


class UserNeedBelief:
    """
    轻量心智理论（ToM）——用户需求信念分布（SELAgents 消融中贡献最大的组件: 31.2%）

    用贝叶斯更新维护 P(need | history)：先验 = 上一轮信念，似然 = 当前情绪证据
    与各需求的匹配度。带平滑项防止信念坍缩到 0/1（感知有噪声）。
    """

    NEEDS = [
        "understanding",   # 需要被理解 + 问题立刻解决
        "safety",          # 需要安全感和确定性
        "clarity",         # 需要清晰、分步的引导
        "companionship",   # 需要长期陪伴 + 小步正向反馈
        "escalation",      # 可能需要人工介入或安全资源
        "celebration",     # 需要巩固正向进展 + 适度推进
        "normal",          # 维持当前服务节奏
    ]

    NEED_NAMES = {
        "understanding": "被理解与即时解决",
        "safety": "安全感与确定性",
        "clarity": "清晰分步引导",
        "companionship": "长期陪伴",
        "escalation": "人工介入/安全资源",
        "celebration": "正向巩固",
        "normal": "标准服务",
    }

    def __init__(self, smoothing: float = 0.1):
        self.smoothing = smoothing
        self.belief: dict[str, float] = {n: 1.0 / len(self.NEEDS) for n in self.NEEDS}

    def _likelihood(self, signal: EmotionSignal, trend: str,
                    dominant: str, risk: float) -> dict[str, float]:
        """当前证据对各需求的匹配度（似然），底数 0.2 避免任何需求归零"""
        L = {n: 0.2 for n in self.NEEDS}
        e = signal.primary
        if e == Emotion.ANGER:
            L["understanding"] = 0.9
        elif e == Emotion.ANXIETY:
            L["safety"] = 0.9
        elif e == Emotion.CONFUSION:
            L["clarity"] = 0.9
        elif e in (Emotion.SADNESS, Emotion.DISAPPOINTMENT):
            L["companionship"] = 0.8
        elif e in (Emotion.GRATITUDE, Emotion.SATISFACTION, Emotion.EXCITEMENT):
            L["celebration"] = 0.8
        elif e == Emotion.FRUSTRATION:
            L["understanding"] = 0.6
            L["clarity"] = 0.5
        elif e == Emotion.NEUTRAL:
            L["normal"] = 0.7
        if risk >= 0.6:
            L["escalation"] = 0.95
        if dominant == "chronic_low":
            L["companionship"] = max(L["companionship"], 0.85)
        if trend == "worsening":
            L["safety"] = max(L["safety"], 0.6)
        if trend == "improving":
            L["celebration"] = max(L["celebration"], 0.6)
        return L

    def update(self, signal: EmotionSignal, trend: str,
               dominant: str, risk: float) -> dict[str, float]:
        L = self._likelihood(signal, trend, dominant, risk)
        new = {n: self.belief[n] * L[n] for n in self.NEEDS}
        total = sum(new.values())
        for n in self.NEEDS:
            new[n] /= total
            # 平滑: 防止信念坍缩到 0/1（感知噪声下保持开放）
            new[n] = (1 - self.smoothing) * new[n] + self.smoothing / len(self.NEEDS)
        self.belief = new
        return new

    def map_need(self) -> tuple[str, float]:
        need = max(self.belief, key=self.belief.get)
        return need, self.belief[need]

    def top(self, k: int = 2) -> list[tuple[str, float]]:
        return sorted(self.belief.items(), key=lambda x: -x[1])[:k]


class EmotionReasoningLayer:
    """
    情绪推理层
    追踪多轮情绪轨迹，推理用户深层需求

    v4 升级:
      - 泄漏积分器情绪动力学（SELAgents）: 按真实时间指数衰减的情绪状态
        E += (1-e^(-λΔt))·(signal − E)，半衰期默认 10 分钟（心理学校准）
      - 趋势回归权重改为时间指数衰减（替代 v3 的线性 0.55→1.0 权重）
      - 心智理论: 贝叶斯更新用户需求信念（ToM）
    """

    def __init__(self, window_size: int = 5, escalation_threshold: float = 0.7,
                 max_gap_seconds: float = 600.0,
                 decay_half_life_minutes: float = 10.0):
        self.window_size = window_size
        self.escalation_threshold = escalation_threshold
        self.max_gap_seconds = max_gap_seconds  # 超过此间隔的旧数据不参与回归
        self.trajectory: deque[EmotionSnapshot] = deque(maxlen=window_size)
        self.turn_counter = 0
        # 泄漏积分器（SELAgents）: 空闲期情绪按 λ = ln2/half_life 指数衰减到中性；
        # 每轮刺激以 EMA(α) 积分进衰减状态
        self.decay_half_life_minutes = decay_half_life_minutes
        self._decay_lambda = math.log(2) / (decay_half_life_minutes * 60.0)  # 每秒
        self.turn_alpha = 0.5   # 每轮刺激的 EMA 系数（平滑 2~3 轮噪声）
        self.decayed_state: dict[str, float] = {"valence": 0.0, "arousal": 0.0, "dominance": 0.0}
        self._last_ts: Optional[datetime] = None
        # 心智理论: 用户需求信念
        self.tom = UserNeedBelief()

    def reason(self, signal: EmotionSignal) -> EmotionTrajectory:
        self.turn_counter += 1
        snapshot = EmotionSnapshot(turn=self.turn_counter, signal=signal)
        self.trajectory.append(snapshot)

        full_history = list(self.trajectory)
        # 过滤过期数据供下游使用
        now = datetime.now()
        active_history = [
            s for s in full_history
            if (now - datetime.fromisoformat(s.timestamp)).total_seconds() < self.max_gap_seconds
        ]

        # 泄漏积分器状态更新（SELAgents）: 按真实时间衰减
        self._update_decayed_state(signal)

        # 计算 valence 趋势（时间指数衰减加权的回归）
        slope = self._compute_valence_slope(full_history)
        # 计算 arousal 趋势（同样的加权回归，用于捕获"越来激动但 valence 没变"的恶化）
        arousal_slope = self._compute_arousal_slope(full_history)

        # 判定趋势方向
        trend = self._classify_trend(slope, active_history)

        # 判定主导模式
        dominant = self._classify_dominant(active_history)

        # 推理用户需求
        need = self._infer_need(signal, trend, dominant)

        # 升级风险评估（含 arousal 辅助因子）
        risk = self._assess_escalation(active_history, slope, arousal_slope)

        # ToM: 贝叶斯更新用户需求信念
        self.tom.update(signal, trend, dominant, risk)
        tom_need, tom_belief = self.tom.map_need()

        # 回填快照（供前端轨迹图）
        snapshot.escalation_risk = round(risk, 3)
        snapshot.decayed_valence = round(self.decayed_state["valence"], 3)

        return EmotionTrajectory(
            current=signal,
            history=full_history,
            trend=trend,
            valence_slope=round(slope, 4),
            arousal_slope=round(arousal_slope, 4),
            dominant_pattern=dominant,
            inferred_need=need,
            escalation_risk=round(risk, 3),
            needs_escalation=risk >= self.escalation_threshold,
            summary=self._make_summary(signal, trend, dominant, need, risk),
            decayed_valence=round(self.decayed_state["valence"], 3),
            decayed_arousal=round(self.decayed_state["arousal"], 3),
            decayed_dominance=round(self.decayed_state["dominance"], 3),
            tom_need=tom_need,
            tom_belief=round(tom_belief, 3),
            tom_top_needs=[(n, round(p, 3)) for n, p in self.tom.top(2)],
        )

    def _update_decayed_state(self, signal: EmotionSignal) -> None:
        """泄漏积分器（SELAgents: dE/dt = -λE + g(S) 的离散化）:
        1) 空闲时间衰减: 情绪随时间向中性回归（半衰期 decay_half_life_minutes）
        2) 本轮刺激积分: 每轮一个时间步的指数平滑（EMA, α=0.5）
        """
        now = datetime.now()
        if self._last_ts is None:
            # 首轮: 直接用信号初始化衰减状态
            for dim in ("valence", "arousal", "dominance"):
                self.decayed_state[dim] = float(getattr(signal, dim))
            self._last_ts = now
            return
        dt_sec = max(0.0, (now - self._last_ts).total_seconds())
        idle = math.exp(-self._decay_lambda * dt_sec)
        self._last_ts = now
        for dim in ("valence", "arousal", "dominance"):
            cur = getattr(signal, dim)
            self.decayed_state[dim] = self.decayed_state[dim] * idle \
                + self.turn_alpha * (cur - self.decayed_state[dim] * idle)

    def _decay_weights(self, active: list[EmotionSnapshot]) -> list[float]:
        """时间指数衰减权重: w_i = e^(-λ·Δt_i)（SELAgents 泄漏积分器思想，
        替换 v3 的线性 0.55→1.0 权重——长间隔旧轮次自然降权）"""
        now = datetime.now()
        return [
            math.exp(-self._decay_lambda
                     * (now - datetime.fromisoformat(s.timestamp)).total_seconds())
            for s in active
        ]

    def _weighted_slope(self, values: list[float], weights: list[float]) -> float:
        n = len(values)
        w_sum = sum(weights)
        if w_sum == 0:
            return 0.0
        x_mean = sum(weights[i] * i for i in range(n)) / w_sum
        y_mean = sum(weights[i] * values[i] for i in range(n)) / w_sum
        num = sum(weights[i] * (i - x_mean) * (values[i] - y_mean) for i in range(n))
        den = sum(weights[i] * (i - x_mean) ** 2 for i in range(n))
        return num / den if den != 0 else 0.0

    def _compute_valence_slope(self, history: list[EmotionSnapshot]) -> float:
        if len(history) < 2:
            return 0.0

        # 过滤过期数据：距当前超过 max_gap_seconds 的 snapshot 不参与计算
        now = datetime.now()
        active = [s for s in history
                  if (now - datetime.fromisoformat(s.timestamp)).total_seconds() < self.max_gap_seconds]
        if len(active) < 2:
            return 0.0

        weights = self._decay_weights(active)
        return self._weighted_slope([s.signal.valence for s in active], weights)

    def _compute_arousal_slope(self, history: list[EmotionSnapshot]) -> float:
        """同 _compute_valence_slope，但对 arousal 做加权回归"""
        if len(history) < 2:
            return 0.0
        now = datetime.now()
        active = [s for s in history
                  if (now - datetime.fromisoformat(s.timestamp)).total_seconds() < self.max_gap_seconds]
        if len(active) < 2:
            return 0.0
        weights = self._decay_weights(active)
        return self._weighted_slope([s.signal.arousal for s in active], weights)

    def _classify_trend(self, slope: float, history: list[EmotionSnapshot]) -> str:
        if len(history) < 2:
            return "stable"
        if slope > 0.05:
            return "improving"
        if slope < -0.05:
            return "worsening"
        # 波动较大但无方向
        valences = [s.signal.valence for s in history]
        if max(valences) - min(valences) > 0.5:
            return "volatile"
        return "stable"

    def _classify_dominant(self, history: list[EmotionSnapshot]) -> str:
        if len(history) < 2:
            return "mixed"
        negatives = sum(1 for s in history if s.signal.is_negative)
        if negatives == 0:
            return "positive"
        ratio = negatives / len(history)
        if ratio >= 0.6:
            # 判断长期负面还是突发负面：近3轮 valence 滚动均值 < -0.25
            recent = list(history)[-3:] if len(history) >= 3 else list(history)
            avg_valence = sum(s.signal.valence for s in recent) / len(recent)
            if avg_valence < -0.25:
                return "chronic_low"
            return "acute_negative"
        return "mixed"

    def _infer_need(self, signal: EmotionSignal, trend: str, dominant: str) -> str:
        """推理用户核心需求"""
        if dominant == "acute_negative":
            if signal.primary == Emotion.ANGER:
                return "需要被理解 + 问题立刻解决"
            if signal.primary == Emotion.ANXIETY:
                return "需要安全感和确定性"
            if signal.primary == Emotion.CONFUSION:
                return "需要清晰、分步的引导"
            return "需要情绪被看见和承认"

        if dominant == "chronic_low":
            return "需要长期陪伴 + 小步正向反馈"

        if trend == "worsening":
            return "可能需要人工介入或安全资源"

        if trend == "improving":
            return "需要巩固正向进展 + 适度推进"

        return "维持当前服务节奏"

    def _assess_escalation(self, history: list[EmotionSnapshot], slope: float,
                           arousal_slope: float = 0.0) -> float:
        if len(history) < 2:
            return 0.0
        risk = 0.0
        # 因子1: valence 持续下行
        if slope < -0.08:
            risk += 0.25
        # 因子2: arousal 持续上行（越来越激动）+ valence 非正 → 恶化辅助信号
        if arousal_slope > 0.06 and history[-1].signal.is_negative:
            risk += 0.15
        # 因子3: 连续多轮负面
        recent_neg = sum(1 for s in history[-3:] if s.signal.is_negative)
        risk += recent_neg * 0.15
        # 因子4: 高强度负面
        if history[-1].signal.is_negative and history[-1].signal.intensity > 0.8:
            risk += 0.15
        return min(1.0, risk)

    def _make_summary(self, signal, trend, dominant, need, risk):
        labels = {
            "improving": "↗ 情绪向好",
            "worsening": "↘ 情绪走低",
            "stable": "→ 情绪平稳",
            "volatile": "↔ 情绪波动",
        }
        return (
            f"[{labels.get(trend, trend)} | {dominant}] "
            f"当前: {signal.primary.value}(I={signal.intensity:.2f}) | "
            f"需求: {need} | "
            f"升级风险: {risk:.2f}"
        )


# ---------------------------------------------------------------------------
# Layer 3: 策略适配层
# ---------------------------------------------------------------------------


@dataclass
class ResponseStrategy:
    """策略适配层输出"""
    name: str
    priority: str          # "deescalate" | "support" | "guide" | "maintain" | "celebrate"
    tone: str              # 语气
    pace: str              # 节奏: "slow" | "normal" | "fast"
    verbosity: str         # 信息密度: "minimal" | "normal" | "detailed"
    structure: str         # 结构: "stepwise" | "structured" | "freeform"
    do_first: str          # 开口第一件事
    avoid: list[str]       # 要避免的做法
    examples: list[str]    # few-shot 示例（可选）


class StrategyAdaptationLayer:
    """策略适配层"""

    STRATEGY_MAP: dict[str, ResponseStrategy] = {
        "needs_understanding": ResponseStrategy(
            name="降级共情",
            priority="deescalate",
            tone="calm_and_empathetic",
            pace="normal",
            verbosity="normal",
            structure="freeform",
            do_first="先共情、承认感受，再解决问题",
            avoid=["辩解", "推卸责任", "直接给解决方案而不先共情", "使用'但是'转折"],
            examples=[],
        ),
        "needs_unblocking": ResponseStrategy(
            name="排障共情",
            priority="deescalate",
            tone="steady_and_empathetic",
            pace="normal",
            verbosity="normal",
            structure="structured",
            do_first="承认挫败感是正常的 + 提供一条明确的排查路径",
            avoid=["空洞安慰（'会好的'）, 复杂解释, 跳过排查直接给结论"],
            examples=[],
        ),
        "needs_reassurance": ResponseStrategy(
            name="安抚锚定",
            priority="deescalate",
            tone="steady_and_warm",
            pace="slow",
            verbosity="normal",
            structure="structured",
            do_first="给出确定性——明确告知下一步会发生什么",
            avoid=["开放式问题", "不确定的措辞", "信息过载", "催促决定"],
            examples=[],
        ),
        "needs_clarity": ResponseStrategy(
            name="逐步引导",
            priority="guide",
            tone="patient_and_clear",
            pace="normal",
            verbosity="minimal",
            structure="stepwise",
            do_first="一次只给一个信息点，确认理解后再前进",
            avoid=["大段解释", "跳跃式回答", "假设用户已理解前提"],
            examples=[],
        ),
        "needs_companionship": ResponseStrategy(
            name="共情陪伴",
            priority="support",
            tone="warm_and_gentle",
            pace="slow",
            verbosity="normal",
            structure="freeform",
            do_first="验证感受——'这确实让人难过'——不急于给建议",
            avoid=["急于解决问题", "'想开点'类说教", "过度乐观", "转移话题"],
            examples=[],
        ),
        "needs_efficiency": ResponseStrategy(
            name="高效直达",
            priority="maintain",
            tone="professional_and_brisk",
            pace="fast",
            verbosity="minimal",
            structure="structured",
            do_first="跳过铺垫，直接给结论/方案",
            avoid=["冗长寒暄", "过度解释背景", "绕圈子"],
            examples=[],
        ),
        "needs_celebration": ResponseStrategy(
            name="正向巩固",
            priority="celebrate",
            tone="enthusiastic_and_warm",
            pace="normal",
            verbosity="normal",
            structure="freeform",
            do_first="肯定对方——具体说出对方做得好的地方",
            avoid=["冷水", "过度谦虚", "立即转移话题"],
            examples=[],
        ),
        "default": ResponseStrategy(
            name="标准服务",
            priority="maintain",
            tone="professional_and_friendly",
            pace="normal",
            verbosity="normal",
            structure="freeform",
            do_first="正常回应用户的问题",
            avoid=[],
            examples=[],
        ),
        "needs_clarification": ResponseStrategy(
            name="澄清式提问",
            priority="guide",
            tone="gentle_and_curious",
            pace="normal",
            verbosity="minimal",
            structure="freeform",
            do_first="先确认感受——用一句具体但开放的问题澄清，不预设情绪判断",
            avoid=["强势安抚", "直接给建议", "假设用户情绪", "大段解释"],
            examples=[],
        ),
    }

    # 策略兼容表: 规则默认策略 → 允许 Q 网络考虑的相邻策略（DialogXpert 式
    # 规则先验缩小动作空间；只允许语义兼容的替换，杜绝降级→庆祝之类的跳跃）
    COMPATIBLE: dict[str, list[str]] = {
        "needs_understanding": ["needs_unblocking", "needs_reassurance", "needs_companionship"],
        "needs_unblocking": ["needs_understanding", "needs_clarity"],
        "needs_reassurance": ["needs_understanding", "needs_companionship"],
        "needs_clarity": ["needs_unblocking", "needs_efficiency"],
        "needs_companionship": ["needs_reassurance", "needs_understanding"],
        "needs_efficiency": ["needs_clarity", "default"],
        "needs_celebration": ["needs_efficiency", "default"],
        "default": ["needs_efficiency", "needs_clarity"],
        "needs_clarification": ["default", "needs_companionship"],
    }

    # 策略中文名 → STRATEGY_MAP key 的反向索引
    _KEY_BY_NAME: dict[str, str] = {s.name: k for k, s in STRATEGY_MAP.items()}

    def candidates(self, signal: EmotionSignal, trajectory: EmotionTrajectory) -> list[str]:
        """
        规则先验候选集：规则默认策略 + 语义兼容的相邻策略（返回 STRATEGY_MAP 的 key）。
        Q 网络只在候选集内选择（缩小动作空间，保证冷启动与探索安全）。
        """
        rule = self.select(signal, trajectory)
        key = self._KEY_BY_NAME.get(rule.name, "default")
        allowed = {key}
        for name in self.COMPATIBLE.get(key, []):
            allowed.add(name)
        # 低置信感知 → 允许澄清式提问兜底
        if signal.confidence < 0.5 and rule.priority != "deescalate":
            allowed.add("needs_clarification")
        return sorted(allowed)

    def select(self, signal: EmotionSignal, trajectory: EmotionTrajectory) -> ResponseStrategy:
        # 升级风险高 → 安全策略（阈值由 Layer 2 内部判定，消费布尔值）
        if trajectory.needs_escalation:
            return ResponseStrategy(
                name="安全兜底",
                priority="deescalate",
                tone="calm_and_caring",
                pace="slow",
                verbosity="normal",
                structure="structured",
                do_first="表达关心 + 提供专业求助资源",
                avoid=["任何可能被解读为轻视的措辞"],
                examples=[],
            )

        # 感知置信度门控（SELAgents 部分可观测实验的教训: 推断误差与整体
        # 性能 r=-0.94）——情绪判断不可靠时不硬降级，改走澄清式提问
        if signal.confidence < 0.5 and signal.intensity >= 0.2:
            return self.STRATEGY_MAP["needs_clarification"]

        # 根据情绪 + 轨迹精准匹配
        dominant = trajectory.dominant_pattern

        if dominant == "acute_negative":
            if signal.primary == Emotion.ANGER:
                return self.STRATEGY_MAP["needs_understanding"]
            if signal.primary == Emotion.ANXIETY:
                # PAD 第三维: 低支配（无助感强）时陪伴优先于确定性安抚
                if signal.is_low_dominance:
                    return self.STRATEGY_MAP["needs_companionship"]
                return self.STRATEGY_MAP["needs_reassurance"]
            if signal.primary == Emotion.CONFUSION:
                return self.STRATEGY_MAP["needs_clarity"]
            if signal.primary == Emotion.FRUSTRATION:
                return self.STRATEGY_MAP["needs_unblocking"]
            if signal.primary == Emotion.DISAPPOINTMENT:
                return self.STRATEGY_MAP["needs_companionship"]
            return self.STRATEGY_MAP["needs_companionship"]

        if dominant == "chronic_low":
            # 连续低落 → 走陪伴策略而非解决问题
            return self.STRATEGY_MAP["needs_companionship"]

        if trajectory.trend == "worsening":
            return self.STRATEGY_MAP["needs_reassurance"]

        if trajectory.trend == "improving":
            return self.STRATEGY_MAP["needs_celebration"]

        # 即时信号判断 — 当前情绪优先于历史模式
        if signal.primary == Emotion.GRATITUDE or signal.primary == Emotion.SATISFACTION:
            return self.STRATEGY_MAP["needs_celebration"]

        if signal.primary == Emotion.EXCITEMENT:
            return self.STRATEGY_MAP["needs_celebration"]

        if signal.is_negative and signal.is_high_arousal:
            return self.STRATEGY_MAP["needs_understanding"]

        if signal.primary == Emotion.SADNESS:
            return self.STRATEGY_MAP["needs_companionship"]

        if signal.primary == Emotion.DISAPPOINTMENT:
            return self.STRATEGY_MAP["needs_companionship"]

        if signal.primary == Emotion.FRUSTRATION:
            return self.STRATEGY_MAP["needs_unblocking"]

        if signal.primary == Emotion.ANXIETY:
            return self.STRATEGY_MAP["needs_reassurance"]

        if signal.primary == Emotion.CONFUSION:
            return self.STRATEGY_MAP["needs_clarity"]

        return self.STRATEGY_MAP["default"]


# ---------------------------------------------------------------------------
# Layer 4: 生成优化层 — 情感条件解码
# ---------------------------------------------------------------------------


class EmotionConditionedGenerator:
    """
    生成优化层
    实现情感条件解码：将情绪信号注入生成过程，优化 token 选择和输出结构

    本 Demo 模拟 LLM 的生成过程，实际部署时替换为真实 LLM 调用
    """

    # 情感条件提示模板
    EMOTION_DECODER_TEMPLATES: dict[str, str] = {
        "deescalate": (
            "[情绪条件解码] 当前状态: 用户处于负面情绪中(高唤醒)。\n"
            "解码偏置:\n"
            "  - 优先选择共情、安抚类 token (理解/确实/辛苦/我听到)\n"
            "  - 抑制辩解、说教类 token (但是/你应该/实际上)\n"
            "  - 降低信息密度，每句话控制在 25 字以内\n"
            "  - 禁用反问句\n"
            "  - frustration(0.7)偏挫败→先承认感受再给排查路径; anger(0.85)偏愤怒→纯降级\n"
        ),
        "support": (
            "[情绪条件解码] 当前状态: 用户需要情绪支持。\n"
            "解码偏置:\n"
            "  - 优先选择验证感受的 token (这种感受很正常/换我也会)\n"
            "  - 抑制'解决问题'冲动的 token\n"
            "  - 保持语气温暖，节奏缓慢\n"
            "  - 给沉默留出空间（不要连续发问）\n"
        ),
        "guide": (
            "[情绪条件解码] 当前状态: 用户需要清晰引导。\n"
            "解码偏置:\n"
            "  - 强制结构化输出: 一次只输出一个步骤\n"
            "  - 每步结尾加确认: '这一步清楚吗？'\n"
            "  - 抑制跳步和多任务并行\n"
            "  - 使用编号或分点，视觉上更易读\n"
        ),
        "maintain": (
            "[情绪条件解码] 当前状态: 用户情绪正常。\n"
            "解码偏置: 标准模式，正常响应。\n"
        ),
        "celebrate": (
            "[情绪条件解码] 当前状态: 正向情绪，需要巩固。\n"
            "解码偏置:\n"
            "  - 优先选择积极反馈 token (太棒了/做得很好/坚持下来)\n"
            "  - 具体指出用户的努力/进步（而非泛泛夸奖）\n"
            "  - 适度推进下一步目标\n"
        ),
    }

    # 回复前缀（完整短句，不截断，确保与正文自然衔接）
    OPENING_PHRASES: dict[str, list[str]] = {
        "deescalate": [
            "我听到了。",
            "感谢你告诉我这些。",
            "我能理解你的感受。",
            "你说得对，我认真听了。",
            "先停下来，不急着往下走。",
        ],
        "support": [
            "这种感觉很正常。",
            "换作是我也会有同样的感受。",
            "没关系，慢慢来。",
            "我在这里陪着你。",
            "先不急，我们慢慢说。",
        ],
        "guide": [
            "我们从这里开始：",
            "一步一步来：",
            "很简单，就一件事：",
            "先做这一步：",
        ],
        "maintain": [
            "好的。",
            "明白了。",
            "收到。",
        ],
        "celebrate": [
            "太好了！",
            "真为你高兴。",
            "这个进步很不容易。",
            "干得漂亮。",
        ],
    }

    def __init__(self, model_name: str = "simulated-llm"):
        self.model_name = model_name
        self.generation_log: list[dict] = []  # 记录每次生成的解码偏置

    def generate(
        self,
        user_input: str,
        signal: EmotionSignal,
        trajectory: EmotionTrajectory,
        strategy: ResponseStrategy,
    ) -> str:
        """
        情感条件解码主流程

        实际部署时这里调用 LLM，将以下信息注入:
          - dynamic system prompt (情绪条件)
          - few-shot examples (匹配当前情绪状态)
          - logit_bias (特定 token 的加/减权)
        """
        priority = strategy.priority

        # 1. 构建情感条件 system prompt
        emotion_prompt = self._build_emotion_system_prompt(signal, trajectory, strategy)

        # 2. 选择开口方式
        opening = self._select_opening(priority, signal)

        # 3. 按策略构建输出结构
        body = self._build_structured_body(user_input, signal, trajectory, strategy)

        # 4. 情绪轨迹感知的收尾
        closing = self._build_closing(trajectory)

        # 5. (模拟) token-level 解码偏置
        logit_bias = self._compute_logit_bias(signal, strategy)

        # 6. 组装最终输出
        response = self._assemble(opening, body, closing, strategy)

        # 记录日志
        self.generation_log.append({
            "turn": trajectory.history[-1].turn if trajectory.history else 0,
            "emotion": signal.primary.value,
            "priority": priority,
            "valence": signal.valence,
            "logit_bias_applied": {k: f"{v:+.2f}" for k, v in list(logit_bias.items())[:5]},
        })

        return response

    def _build_emotion_system_prompt(
        self,
        signal: EmotionSignal,
        trajectory: EmotionTrajectory,
        strategy: ResponseStrategy,
    ) -> str:
        template = self.EMOTION_DECODER_TEMPLATES.get(strategy.priority, self.EMOTION_DECODER_TEMPLATES["maintain"])

        # 注入情绪轨迹上下文
        trajectory_context = (
            f"\n[情绪轨迹] 近{len(trajectory.history)}轮趋势: {trajectory.trend} "
            f"(slope={trajectory.valence_slope:.3f}) | "
            f"模式: {trajectory.dominant_pattern} | "
            f"升级风险: {trajectory.escalation_risk:.2f}\n"
        )

        # 注入当前情绪信号
        signal_context = (
            f"[当前情绪] {signal.primary.value} | "
            f"置信度={signal.confidence:.2f} | "
            f"强度={signal.intensity:.2f} | "
            f"valence={signal.valence:.2f} | "
            f"arousal={signal.arousal:.2f}\n"
        )

        # 策略指令
        strategy_context = (
            f"[输出策略] {strategy.name} | "
            f"语气={strategy.tone} | "
            f"节奏={strategy.pace} | "
            f"结构={strategy.structure}\n"
            f"第一优先级: {strategy.do_first}\n"
            + (f"禁止: {', '.join(strategy.avoid)}\n" if strategy.avoid else "")
        )

        return template + trajectory_context + signal_context + strategy_context

    def _select_opening(self, priority: str, signal: EmotionSignal) -> str:
        import random
        phrases = self.OPENING_PHRASES.get(priority, self.OPENING_PHRASES["maintain"])
        return random.choice(phrases)

    def _build_structured_body(
        self,
        user_input: str,
        signal: EmotionSignal,
        trajectory: EmotionTrajectory,
        strategy: ResponseStrategy,
    ) -> str:
        """根据策略构建不同结构的正文"""
        if strategy.name == "澄清式提问":
            return self._body_clarify(user_input, signal)
        if strategy.structure == "stepwise":
            return self._body_stepwise(user_input)
        elif strategy.structure == "structured":
            return self._body_structured(user_input, signal)
        else:
            return self._body_freeform(user_input, signal, trajectory)

    def _body_clarify(self, user_input: str, signal: EmotionSignal) -> str:
        """低置信感知下的澄清正文：不预设情绪，先确认感受"""
        if signal.primary == Emotion.NEUTRAL:
            return (
                "\n我有点拿不准你现在的感受——是遇到了麻烦事，还是单纯想聊聊？"
                "你可以随意说说，我来配合你的节奏。"
            )
        return (
            "\n听起来你似乎有点"
            + ("不舒服" if signal.is_negative else "不一样的情绪")
            + "，我想确认一下——你现在更接近哪种状态："
            "遇到了麻烦，还是心情上有些起伏？"
            "说出来，我才能更好地帮你。"
        )

    def _body_stepwise(self, user_input: str) -> str:
        return (
            "\n**第 1 步**：我们先确认一下你的问题——"
            + user_input[:30]
            + ("..." if len(user_input) > 30 else "")
            + "\n这一步清楚吗？"
        )

    def _body_structured(self, user_input: str, signal: EmotionSignal) -> str:
        if signal.primary == Emotion.FRUSTRATION:
            return (
                "\n我们先停下来，不急着继续踩坑。\n\n"
                "**现在**：告诉我具体卡在哪一步了——我会逐条排查\n"
                "**然后**：每解决一步，确认成功了再走下一步\n"
                "**最后**：确保整个流程跑通\n"
                "我们从「现在」这一步开始——你方便描述一下卡住时的画面吗？"
            )
        if signal.primary == Emotion.ANXIETY:
            return (
                "\n我先明确告诉你接下来的处理流程，这样你心里有数：\n"
                "  **现在**：我听你说完，不跳过任何细节\n"
                "  **然后**：我们一起梳理最关键的一步是什么\n"
                "  **最后**：给你一个明确可执行的方案\n"
                "我们先从「现在」这一步开始——你方便再多说一点吗？"
            )
        if signal.primary == Emotion.CONFUSION:
            return (
                "\n这个问题我帮你拆开来看：\n\n"
                "  1) 我们先确认目前你已理解的部分\n"
                "  2) 找到卡住的具体位置\n"
                "  3) 逐点解释，直到你完全清楚\n"
                "我们先从第 1 点开始——你现在对这个配置了解多少？不用怕说错。"
            )
        return "\n我帮你梳理一下：\n" + f"  核心问题：{user_input[:40]}...\n  建议方向：待进一步确认"

    def _body_freeform(self, user_input: str, signal: EmotionSignal, trajectory: EmotionTrajectory) -> str:
        if signal.primary == Emotion.ANGER:
            return (
                "\n我完全理解你为什么会有这样的感受。"
                "你遇到的情况确实让人不舒服。\n\n"
                "我想帮你把这件事解决好。"
                "你能告诉我，目前最困扰你的一点是什么吗？"
                "我先把这个问题优先处理。"
            )
        if signal.primary == Emotion.FRUSTRATION:
            return (
                "\n反复遇到同一个问题确实让人崩溃，这种挫败感我懂。\n\n"
                "我们换个思路——你刚才卡在哪一步了？"
                "我陪你一步步排查。"
            )
        if signal.primary == Emotion.SADNESS:
            return (
                "\n谢谢你愿意跟我说这些。这不是小事，你的感受很重要。\n\n"
                "我先不急着给你建议，就陪着你慢慢聊。"
                "如果你愿意，可以再多说一点现在的状态。"
            )
        if trajectory.dominant_pattern == "chronic_low":
            return (
                "\n我注意到这几次交流中，你似乎一直在承受挺大的压力。"
                "我想让你知道，我一直在这里。\n\n"
                "我们今天不一定要解决什么——随便聊聊也可以。"
                "你觉得呢？"
            )
        if signal.primary == Emotion.ANXIETY:
            return (
                "\n我听到了你的担心。让我们先停下来，把眼前的事情一件件理清楚。\n\n"
                "你不需要一个人面对这些，我们一起看看现在最让你不安的是什么？"
            )
        if signal.primary == Emotion.CONFUSION:
            return (
                "\n没关系，这个问题确实容易让人绕进去。\n"
                "我换个简单的方式说——"
            )
        if signal.primary == Emotion.GRATITUDE:
            return (
                "\n能帮到你我也很高兴！"
                "你愿意反馈给我，这份认可对我很重要。\n\n"
                "如果后面还有需要，随时找我。"
            )
        if signal.primary == Emotion.SATISFACTION:
            return (
                "\n看到你把问题解决了，我也替你开心。"
                "你做得很棒——能坚持一步步搞定不简单的。"
            )
        if signal.primary == Emotion.EXCITEMENT:
            return (
                "\n这个反应太有感染力了！看到你这么兴奋，我也跟着开心。\n\n"
                "这确实是个值得庆祝的时刻——好好享受！"
            )
        # 中性情绪：正常响应，不预设问答模式
        return (
            "\n好的，我来回答你的问题。"
            if any(kw in user_input for kw in ["怎么", "如何", "什么", "帮我", "请问", "?"])
            else "\n收到，有什么我可以帮你的吗？"
        )

    def _build_closing(self, trajectory: EmotionTrajectory) -> str:
        if trajectory.trend == "improving" and trajectory.current.valence > 0.3:
            return "\n\n看到情况在往好的方向发展——继续保持这个势头。我随时在这里。"
        if trajectory.trend == "worsening" and trajectory.escalation_risk >= 0.5:
            return (
                "\n\n另外，我注意到你最近几轮的状态有些波动。"
                "如果你觉得需要跟真人聊聊，我可以帮你找专业的支持资源——你不需要一个人扛。"
            )
        return ""

    def _compute_logit_bias(
        self, signal: EmotionSignal, strategy: ResponseStrategy
    ) -> dict[str, float]:
        """
        模拟 token-level 解码偏置
        实际部署: 通过 LLM API 的 logit_bias 参数实现
        例如 OpenAI: logit_bias={"2435": 0.5, "6432": -0.3}
        """
        bias: dict[str, float] = {}

        if strategy.priority == "deescalate":
            # 提升共情 token 概率
            bias.update({"我理解": 0.3, "确实": 0.2, "感谢": 0.25, "辛苦": 0.2})
            # 降低辩解 token 概率
            bias.update({"但是": -0.4, "不过": -0.3, "实际上": -0.35, "你应该": -0.5})

        elif strategy.priority == "support":
            bias.update({"正常": 0.2, "慢慢来": 0.3, "没关系": 0.25, "我在这里": 0.35})
            bias.update({"你可以": -0.2, "为什么": -0.3, "试试": -0.1})

        elif strategy.priority == "guide":
            bias.update({"第一步": 0.3, "先": 0.25, "接下来": 0.2, "然后": 0.15})
            bias.update({"同时": -0.3, "另外": -0.25})  # 抑制并行信息

        elif strategy.priority == "celebrate":
            bias.update({"太棒": 0.4, "厉害": 0.35, "坚持": 0.3, "进步": 0.3})

        return bias

    def _assemble(self, opening: str, body: str, closing: str, strategy: ResponseStrategy) -> str:
        # 确保开口句和正文之间用换行隔开，避免半句拼接
        sep = "\n\n" if strategy.priority in ("deescalate", "support", "celebrate") else "\n"
        return f"{opening}{sep}{body}{closing}"

    def get_generation_log(self) -> list[dict]:
        return self.generation_log


# ---------------------------------------------------------------------------
# 主流程: 四层编排
# ---------------------------------------------------------------------------


class EmotionAdaptiveAgent:
    """四层情绪优化 Agent"""

    def __init__(self):
        self.perception = EmotionPerceptionLayer()
        self.reasoning = EmotionReasoningLayer()
        self.strategy = StrategyAdaptationLayer()
        self.generator = EmotionConditionedGenerator()

    def process(self, user_input: str) -> dict:
        """一次完整的四层流水线"""
        # Layer 1
        signal = self.perception.perceive(user_input)

        # Layer 2
        trajectory = self.reasoning.reason(signal)

        # Layer 3
        strategy = self.strategy.select(signal, trajectory)

        # Layer 4
        response = self.generator.generate(user_input, signal, trajectory, strategy)

        return {
            "user_input": user_input,
            "layer1_perception": {
                "primary": signal.primary.value,
                "confidence": signal.confidence,
                "intensity": signal.intensity,
                "valence": signal.valence,
                "arousal": signal.arousal,
                "dominance": signal.dominance,
                "source": signal.source,
                "secondary": [(e.value, s) for e, s in signal.secondary],
            },
            "layer2_reasoning": {
                "trend": trajectory.trend,
                "valence_slope": trajectory.valence_slope,
                "dominant_pattern": trajectory.dominant_pattern,
                "inferred_need": trajectory.inferred_need,
                "escalation_risk": trajectory.escalation_risk,
                "summary": trajectory.summary,
                "decayed": {
                    "valence": trajectory.decayed_valence,
                    "arousal": trajectory.decayed_arousal,
                    "dominance": trajectory.decayed_dominance,
                },
                "tom_need": UserNeedBelief.NEED_NAMES.get(
                    trajectory.tom_need, trajectory.tom_need),
                "tom_belief": trajectory.tom_belief,
                "tom_top_needs": [
                    (UserNeedBelief.NEED_NAMES.get(n, n), p)
                    for n, p in trajectory.tom_top_needs
                ],
            },
            "layer3_strategy": {
                "name": strategy.name,
                "priority": strategy.priority,
                "tone": strategy.tone,
                "pace": strategy.pace,
                "do_first": strategy.do_first,
                "avoid": strategy.avoid,
            },
            "layer4_response": response,
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _short(val: float) -> str:
    """带颜色的数值条形图"""
    bar_len = int(abs(val) * 10)
    if val > 0:
        return f"+{val:.2f} {'█' * bar_len}"
    elif val < 0:
        return f"{val:.2f} {'█' * bar_len}"
    return " 0.00"


def _emotion_bar(valence: float) -> str:
    """valence 可视化"""
    if valence > 0.3:
        return f"😊 {valence:+.2f}"
    elif valence > 0:
        return f"🙂 {valence:+.2f}"
    elif valence > -0.3:
        return f"😐 {valence:+.2f}"
    elif valence > -0.6:
        return f"😟 {valence:+.2f}"
    return f"😡 {valence:+.2f}"


def print_result(result: dict):
    """格式化打印四层流水线输出"""
    p = result["layer1_perception"]
    r = result["layer2_reasoning"]
    s = result["layer3_strategy"]
    resp = result["layer4_response"]

    print()
    print("━" * 72)
    print(f"  输入: \"{result['user_input']}\"")
    print("━" * 72)

    # Layer 1
    print(f"\n  [Layer 1] 情绪感知 ({p.get('source', 'local')})")
    print(f"    主情绪: {p['primary']}  |  置信度: {p['confidence']:.0%}  |  强度: {p['intensity']:.2f}")
    print(f"    valence: {_emotion_bar(p['valence'])}  |  arousal: {p['arousal']:.2f}  |  dominance: {p.get('dominance', 0.0):+.2f}")
    if p["secondary"]:
        secs = ", ".join(f"{e}({s:.0%})" for e, s in p["secondary"])
        print(f"    次情绪: {secs}")

    # Layer 2
    print(f"\n  [Layer 2] 情绪推理 + 轨迹动力学 + 心智理论")
    print(f"    趋势: {r['trend']}  (slope={r['valence_slope']:+.3f})")
    d = r.get("decayed", {})
    print(f"    衰减状态: valence={d.get('valence', 0):+.2f}  arousal={d.get('arousal', 0):.2f}  dominance={d.get('dominance', 0):+.2f}")
    print(f"    模式: {r['dominant_pattern']}  |  升级风险: {r['escalation_risk']:.0%}")
    print(f"    需求: {r['inferred_need']}")
    print(f"    ToM: {r.get('tom_need', '-')} (信念 {r.get('tom_belief', 0):.0%})")
    print(f"    总结: {r['summary']}")

    # Layer 3
    print(f"\n  [Layer 3] 策略适配")
    print(f"    策略: {s['name']}  |  优先级: {s['priority']}  |  语气: {s['tone']}")
    print(f"    节奏: {s['pace']}  |  首要行动: {s['do_first']}")
    if s["avoid"]:
        print(f"    禁止: {', '.join(s['avoid'])}")

    # Layer 4
    print(f"\n  [Layer 4] Agent 输出 (情感条件解码)")
    print(f"    {'─' * 60}")
    for line in resp.strip().split("\n"):
        print(f"    │ {line}")
    print(f"    {'─' * 60}")

    print()


def run_interactive():
    """交互模式"""
    agent = EmotionAdaptiveAgent()
    print("\n  === 情绪自适应 Agent ===\n")
    print("  输入消息与 Agent 对话。Agent 会实时感知你的情绪并调整回复策略。")
    print("  输入 /trajectory 查看情绪轨迹    /log 查看解码日志    /quit 退出\n")

    while True:
        try:
            user_input = input("  👤 你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  再见！")
            break

        if not user_input:
            continue
        if user_input == "/quit":
            break
        if user_input == "/trajectory":
            _show_trajectory(agent)
            continue
        if user_input == "/log":
            _show_log(agent)
            continue

        result = agent.process(user_input)
        print_result(result)
        print(f"  🤖 Agent: {result['layer4_response']}")


def _show_trajectory(agent: EmotionAdaptiveAgent):
    history = list(agent.reasoning.trajectory)
    if not history:
        print("\n  暂无历史\n")
        return
    print(f"\n  === 情绪轨迹 (近{len(history)}轮) ===\n")
    for snap in history:
        v = snap.signal.valence
        marker = _emotion_bar(v)
        decayed = snap.decayed_valence
        dec_txt = f"  衰减后={decayed:+.2f}" if decayed is not None else ""
        print(
            f"  T{snap.turn:02d}  {snap.signal.primary.value:<14s} "
            f"{marker:<14s}  I={snap.signal.intensity:.2f}"
            f"  dom={snap.signal.dominance:+.2f}{dec_txt}"
        )
    print()


def _show_log(agent: EmotionAdaptiveAgent):
    log = agent.generator.get_generation_log()
    if not log:
        print("\n  暂无生成日志\n")
        return
    print(f"\n  === 解码偏置日志 ===\n")
    for entry in log:
        print(
            f"  T{entry['turn']:02d}  {entry['emotion']:<14s}  "
            f"{entry['priority']:<12s}  bias: {entry['logit_bias_applied']}"
        )
    print()


def run_demo():
    """演示模式：预设对话，展示完整的情绪轨迹追踪"""
    script = [
        "你好，我想问一下怎么导出数据",
        "谢谢！搞定了，导出成功了",
        "怎么又不行了？？刚才明明好了现在又出问题了，烦死了",
        "真的很崩溃，试了好多次了老是卡在这一步",
        "能不能帮我看看，我实在搞不懂这个配置是怎么回事",
        "好的，我试一下你说的第一步",
        "嗯...确实可以了。不好意思刚才态度不太好",
        "太棒了！！全部搞定了，感谢感谢！",
        "我觉得自己最近状态不太好，总是莫名想哭",
        "就是感觉做什么都没意思，也不想跟人说话",
        "谢谢你愿意听我说这些。就有人陪着说说话感觉好一点了",
        "今天心情好多了，谢谢你之前的陪伴",
    ]

    agent = EmotionAdaptiveAgent()
    print("\n  === 情绪自适应 Agent Demo ===\n")
    print("  演示完整的四层流水线 + 情绪轨迹追踪\n")
    print("─" * 72)

    for i, text in enumerate(script, 1):
        user_input = text
        print(f"\n\n  [{i}/{len(script)}] 👤 用户: \"{user_input}\"")

        result = agent.process(user_input)
        print_result(result)

    # 最终轨迹
    print("\n\n  " + "=" * 72)
    _show_trajectory(agent)
    print("  " + "=" * 72)
    _show_log(agent)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="情绪自适应 Agent Demo")
    parser.add_argument("--demo", action="store_true", help="运行预设演示")
    args = parser.parse_args()

    if args.demo:
        run_demo()
    else:
        run_interactive()
