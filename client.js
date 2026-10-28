// Browser bundle for the original EmoFlow v4 interface inside a Harness session.
window.__ModuleLoader__.load({
  id: "emoflow-harness-ui",
  factory: (require) => {
    const module = { exports: {} };
    const React = require("react");
    const h = React.createElement;
    const EMPTY = Object.freeze({ entries: [], hasMore: false, revision: 0 });
    const COLOR = {
      bg: "#09090b", card: "#0e0e11", border: "#27272a", text: "#fafafa",
      muted: "#a1a1aa", green: "#34d399", red: "#f87171",
      amber: "#fbbf24", violet: "#a78bfa", sky: "#7dd3fc"
    };
    const NEED_NAMES = {
      understanding: "被理解与即时解决", safety: "安全感与确定性",
      clarity: "清晰分步引导", companionship: "长期陪伴",
      escalation: "人工介入/安全资源", celebration: "正向巩固", normal: "标准服务"
    };
    const STYLES = `
      .efh-root,.efh-root *{box-sizing:border-box}
      .efh-root{height:100%;min-height:0;display:flex;background:#09090b;color:#fafafa;
        font:13px/1.6 ui-sans-serif,system-ui,-apple-system,"Segoe UI","Microsoft YaHei",sans-serif}
      .efh-side{width:292px;flex:none;border-right:1px solid #27272a;overflow-y:auto;background:#09090b}
      .efh-brand{display:flex;align-items:center;gap:10px;padding:14px 14px 12px;border-bottom:1px solid #27272a}
      .efh-logo{width:29px;height:29px;display:grid;place-items:center;border-radius:8px;
        background:linear-gradient(135deg,#8b5cf6,#3b82f6);box-shadow:0 2px 8px rgba(139,92,246,.35)}
      .efh-name{font-size:14px;font-weight:650}.efh-tag{font-size:10px;color:#a1a1aa;border:1px solid #27272a;
        border-radius:999px;padding:0 7px}.efh-sidebody{padding:6px 13px 24px}
      .efh-section{display:flex;align-items:center;gap:6px;color:#a1a1aa;font-size:10.5px;font-weight:600;
        letter-spacing:.04em;margin:14px 2px 7px}.efh-section:after{content:"";height:1px;background:#27272a;flex:1}
      .efh-card{border:1px solid #27272a;background:#0e0e11;border-radius:10px;padding:11px 12px}
      .efh-row{display:flex;justify-content:space-between;align-items:center;gap:8px;padding:3px 0;min-height:23px}
      .efh-row-label{color:#a1a1aa}.efh-row-value{text-align:right;font:500 11.5px ui-monospace,SFMono-Regular,Consolas,monospace;
        font-variant-numeric:tabular-nums;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:150px}
      .efh-badge{display:inline-flex;align-items:center;border-radius:999px;padding:1px 8px;font:600 11px ui-sans-serif,system-ui,sans-serif}
      .efh-negative{color:#fca5a5;background:rgba(248,113,113,.12);border:1px solid rgba(248,113,113,.22)}
      .efh-positive{color:#6ee7b7;background:rgba(52,211,153,.12);border:1px solid rgba(52,211,153,.22)}
      .efh-neutral{color:#93c5fd;background:rgba(125,211,252,.12);border:1px solid rgba(125,211,252,.22)}
      .efh-source{color:#6ee7b7;border:1px solid #27272a;border-radius:999px;padding:1px 6px;font-size:10px}
      .efh-progress{height:5px;background:#27272a;border-radius:99px;overflow:hidden;margin:4px 0 7px}
      .efh-progress-fill{height:100%;border-radius:99px;transition:width .35s}
      .efh-chart{display:block;width:100%;height:152px;border:1px solid #27272a;background:#0e0e11;border-radius:10px;padding:5px}
      .efh-trail{display:flex;align-items:center;gap:8px;padding:6px 7px;border-radius:7px;font-size:11px}
      .efh-trail:hover{background:#18181b}.efh-dot{width:7px;height:7px;border-radius:50%;flex:none}
      .efh-trail-value{margin-left:auto;color:#a1a1aa;font:11px ui-monospace,Consolas,monospace}
      .efh-empty{padding:11px;text-align:center;color:#71717a;font-size:11px}
      .efh-main{flex:1;min-width:0;display:flex;flex-direction:column;background:#09090b;overflow:hidden}
      .efh-topbar{height:47px;flex:none;display:flex;align-items:center;justify-content:space-between;
        gap:10px;padding:0 16px;border-bottom:1px solid #27272a;font-size:12px;font-weight:550}
      .efh-session{display:flex;align-items:center;gap:6px;padding:3px 9px;background:#18181b;
        border-radius:99px;color:#a1a1aa;font:10px ui-monospace,Consolas,monospace}
      .efh-session-dot{height:6px;width:6px;border-radius:50%;background:#34d399;box-shadow:0 0 6px #34d399}
      .efh-messages{flex:1;min-height:0;overflow-y:auto;padding:18px 18px 155px}
      .efh-message-list{max-width:800px;margin:0 auto;display:flex;flex-direction:column;gap:16px}
      .efh-message{max-width:min(82%,560px);padding:11px 14px;white-space:pre-wrap;line-height:1.65;font-size:12.5px}
      .efh-user{align-self:flex-end;color:#09090b;background:#fafafa;border-radius:14px 14px 4px 14px}
      .efh-agent{align-self:flex-start;background:#0e0e11;border:1px solid #27272a;border-radius:14px 14px 14px 4px}
      .efh-meta{font-size:10px;color:#a1a1aa;margin-bottom:5px}.efh-help{color:#71717a;font-size:11px;text-align:center;padding:12px}
      .efh-export{width:100%;margin-top:10px;background:#18181b;color:#a1a1aa;border:1px solid #27272a;
        border-radius:8px;padding:6px;cursor:pointer;font-size:11px}.efh-export:hover{color:#fafafa;border-color:#52525b}
      @media(max-width:760px){.efh-side{width:230px}.efh-row-value{max-width:100px}.efh-message{max-width:95%}}
    `;

    function textOf(content) {
      return Array.isArray(content) ? content.filter((part) => part?.type === "text" && typeof part.text === "string")
        .map((part) => part.text).join("\n").trim() : "";
    }
    function resultObject(content) {
      const text = textOf(content);
      if (!text) return null;
      try {
        const parsed = JSON.parse(text);
        if (parsed?.emotion || parsed?.sensor_status) return parsed;
        if (parsed?.structuredContent?.emotion || parsed?.structuredContent?.sensor_status) return parsed.structuredContent;
        if (parsed?.result?.emotion) return parsed.result;
      } catch { /* Ignore non-EmoFlow tool output. */ }
      return null;
    }
    function number(value, fallback = 0) {
      if (value === null || value === undefined || value === "") return fallback;
      const result = Number(value);
      return Number.isFinite(result) ? result : fallback;
    }
    function project(entries) {
      const calls = new Map(), points = new Map(), sensors = new Map();
      function addPoint(data, seq, time, turn, origin) {
        const valence = number(data.emotion?.valence, NaN);
        if (!Number.isFinite(valence) || valence < -1 || valence > 1) return;
        points.set(turn ?? seq, {
          seq, time, turn, origin,
          emotion: String(data.emotion.primary || "neutral"),
          source: String(data.emotion.source || "local"),
          confidence: number(data.emotion.confidence), intensity: number(data.emotion.intensity),
          valence, arousal: number(data.emotion.arousal), dominance: number(data.emotion.dominance),
          trend: String(data.trajectory?.trend || "stable"),
          pattern: String(data.trajectory?.dominant_pattern || "mixed"),
          risk: number(data.trajectory?.escalation_risk),
          decayedValence: number(data.reasoning_details?.decayed_valence, valence),
          tomNeed: String(data.reasoning_details?.tom_need || "—"),
          tomBelief: number(data.reasoning_details?.tom_belief),
          strategy: String(data.strategy?.name || "—"),
          selectionSource: String(data.decision_source || "local_rule"),
          learningEnabled: data.learning?.enabled === true,
          mode: String(data.service_decision?.mode || "companion")
        });
      }
      for (const item of entries || []) {
        if (item?.type !== "event") continue;
        const event = item.event;
        if (event?.type === "user/message" && event.data?.source?.kind === "emoflow-auto") {
          const data = event.data.source.analysis;
          if (data?.emotion) addPoint(data, event.seq, event.time, event.data.source.turn, "automatic");
          continue;
        }
        if (event?.type === "tool/call") {
          calls.set(String(event.data.callId), event.data.name);
          continue;
        }
        if (event?.type !== "tool/result" || event.data?.message?.isError) continue;
        const toolName = calls.get(String(event.data.message?.source?.callId ?? "")) || "";
        const data = resultObject(event.data.message?.content);
        if (!data) continue;
        const turn = event.data.turn;
        if ((toolName === "mcp__emoflow__assess_emotion" || data.engine === "EmoFlow local four-layer engine") && data.emotion) {
          addPoint(data, event.seq, event.time, turn, "tool");
        } else if (toolName === "mcp__emoflow__assess_simulated_sensor" && data.source === "simulated") {
          sensors.set(turn, String(data.sensor_status || "missing"));
        }
      }
      return [...points.values()].sort((a, b) => a.seq - b.seq).map((point) => ({ ...point, sensorStatus: sensors.get(point.turn) || "" }));
    }
    function projectMessages(entries) {
      const messages = [];
      for (const item of entries || []) {
        if (item?.type !== "event") continue;
        const event = item.event;
        if (event?.type === "user/message" && event.data?.source?.kind === "user") {
          const text = textOf(event.data.content);
          if (text) messages.push({ seq: event.seq, role: "user", text });
        } else if (event?.type === "assistant/message") {
          const text = textOf(event.data?.message?.content);
          if (text) messages.push({ seq: event.seq, role: "agent", text, turn: event.data.turn });
        }
      }
      return messages.sort((a, b) => a.seq - b.seq);
    }
    function percent(value) { return `${Math.round(Math.max(0, Math.min(1, value || 0)) * 100)}%`; }
    function fixed(value) { return Number.isFinite(value) ? value.toFixed(2) : "—"; }
    function row(label, value, key) {
      return h("div", { className: "efh-row", key: key || label },
        h("span", { className: "efh-row-label" }, label), h("span", { className: "efh-row-value" }, value));
    }
    function section(label, child) {
      return h(React.Fragment, { key: label }, h("div", { className: "efh-section" }, label), child);
    }
    function progress(value, color) {
      return h("div", { className: "efh-progress" },
        h("div", { className: "efh-progress-fill", style: { width: percent(value), background: color } }));
    }
    function riskColor(value) { return value >= .7 ? COLOR.red : value >= .4 ? COLOR.amber : COLOR.green; }
    function valenceColor(value) { return value > 0 ? COLOR.green : value > -.3 ? COLOR.amber : value > -.6 ? "#fb923c" : COLOR.red; }
    function trendName(value) {
      return ({ improving: "↗ 向好", worsening: "↘ 走低", declining: "↘ 走低", stable: "→ 平稳", volatile: "↔ 波动" })[value] || value;
    }
    function MiniChart({ points }) {
      const width = 264, height = 148, left = 24, right = 27, top = 17, bottom = 19;
      const x = (i) => left + (points.length === 1 ? .5 : i / (points.length - 1)) * (width - left - right);
      const yVal = (v) => top + (1 - (v + 1) / 2) * (height - top - bottom);
      const yRisk = (v) => top + (1 - v) * (height - top - bottom);
      const path = (pick, y) => points.map((p, i) => `${i ? "L" : "M"}${x(i).toFixed(1)} ${y(pick(p)).toFixed(1)}`).join(" ");
      return h("svg", { className: "efh-chart", viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": "情绪效价与风险联动图" },
        [0, .5, 1].map((v) => h("line", { key: v, x1: left, x2: width - right, y1: yRisk(v), y2: yRisk(v), stroke: COLOR.border })),
        h("text", { x: 2, y: 17, fill: COLOR.muted, fontSize: 10 }, "v"),
        h("text", { x: width - 13, y: 17, fill: COLOR.muted, fontSize: 10 }, "r"),
        h("path", { d: path((p) => p.valence, yVal), fill: "none", stroke: COLOR.violet, strokeWidth: 2, strokeLinecap: "round" }),
        h("path", { d: path((p) => p.risk, yRisk), fill: "none", stroke: COLOR.amber, strokeWidth: 2, strokeLinecap: "round" }),
        points.map((p, i) => h("circle", { key: p.seq, cx: x(i), cy: yVal(p.valence), r: 3, fill: COLOR.violet },
          h("title", null, `T${p.turn}: valence ${fixed(p.valence)}, risk ${percent(p.risk)}`)))
      );
    }
    function exportTrajectory(points) {
      const blob = new Blob([JSON.stringify({ source: "EmoFlow Harness", exportedAt: new Date().toISOString(), points }, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob), link = document.createElement("a");
      link.href = url; link.download = "emoflow-emotion-trajectory.json"; link.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    }
    function Dashboard({ eventSource, loadOlder, sessionId }) {
      const snapshot = React.useSyncExternalStore(
        eventSource ? (listener) => eventSource.subscribe(listener) : () => () => {},
        eventSource ? () => eventSource.getSnapshot() : () => EMPTY
      );
      const points = React.useMemo(() => project(snapshot.entries), [snapshot]);
      const messages = React.useMemo(() => projectMessages(snapshot.entries), [snapshot]);
      const current = points.at(-1);
      const related = new Map(points.map((point) => [point.turn, point]));
      const emotionBadge = h("span", { className: `efh-badge ${!current ? "efh-neutral" : current.valence < -.15 ? "efh-negative" : current.valence > .15 ? "efh-positive" : "efh-neutral"}` }, current?.emotion || "—");
      return h("div", { className: "efh-root" },
        h("style", null, STYLES),
        h("aside", { className: "efh-side" },
          h("div", { className: "efh-brand" }, h("div", { className: "efh-logo" }, "💜"), h("span", { className: "efh-name" }, "EmoFlow"), h("span", { className: "efh-tag" }, "v4")),
          h("div", { className: "efh-sidebody" },
            section("情绪信号", h("div", { className: "efh-card" },
              row("主情绪", emotionBadge),
              row("感知来源", h("span", { className: "efh-source" }, current ? current.source === "local" ? "本地快通道" : current.source : "—")),
              row("强度", current ? percent(current.intensity) : "—"),
              row("valence", current ? fixed(current.valence) : "—"),
              row("arousal", current ? fixed(current.arousal) : "—"),
              row("dominance", current ? fixed(current.dominance) : "—")
            )),
            section("推理 · 策略", h("div", { className: "efh-card" },
              row("趋势", current ? trendName(current.trend) : "—"),
              row("模式", current?.pattern || "—"),
              row("升级风险", current ? percent(current.risk) : "—"),
              progress(current?.risk, riskColor(current?.risk || 0)),
              row("衰减后 valence", current ? fixed(current.decayedValence) : "—"),
              row("ToM 需求", current ? NEED_NAMES[current.tomNeed] || current.tomNeed : "—"),
              row("信念强度", current ? percent(current.tomBelief) : "—"),
              progress(current?.tomBelief, COLOR.violet),
              row("策略", current?.strategy || "—"),
              row("选择来源", current ? current.selectionSource === "local_rule" ? "冷启动 · 规则" : current.selectionSource : "—")
            )),
            section("Q 学习状态", h("div", { className: "efh-card" },
              row("学习介入率", current?.learningEnabled ? "已启用" : "未启用"),
              row("经验数", "—"), row("最近结算奖励", "—"), row("学习率调制", "—")
            )),
            section("趋势 × 风险联动图", h(MiniChart, { points })),
            section("情绪轨迹", points.length ? h("div", null,
              points.map((point, index) => h("div", { className: "efh-trail", key: point.seq },
                h("span", { className: "efh-dot", style: { background: valenceColor(point.valence) } }),
                h("span", null, `T${point.turn || index + 1} ${point.emotion}`),
                h("span", { className: "efh-trail-value" }, fixed(point.valence))
              )), snapshot.hasMore && h("button", { className: "efh-export", onClick: loadOlder }, "加载更早记录")) : h("div", { className: "efh-empty" }, "暂无轨迹")),
            h("button", { className: "efh-export", disabled: !points.length, onClick: () => exportTrajectory(points) }, "导出轨迹 JSON")
          )
        ),
        h("div", { className: "efh-main" },
          h("div", { className: "efh-topbar" },
            h("span", null, "EmoFlow · 情绪自适应 Agent"),
            h("span", { className: "efh-session" }, h("span", { className: "efh-session-dot" }), String(sessionId || "未连接").slice(0, 12))
          ),
          h("div", { className: "efh-messages" }, h("div", { className: "efh-message-list" },
            h("div", { className: "efh-message efh-agent" },
              h("div", { className: "efh-meta" }, "EmoFlow 界面提示"),
              "你好！这里会同步显示当前 Harness 会话与 EmoFlow 情绪分析。请在下方输入框自然对话；每条新消息都会更新左侧诊断。"
            ),
            messages.map((message) => h("div", { key: message.seq, className: `efh-message ${message.role === "user" ? "efh-user" : "efh-agent"}` },
              message.role === "agent" && related.get(message.turn) && h("div", { className: "efh-meta" },
                `[${related.get(message.turn).origin === "tool" ? "本轮已调用 EmoFlow" : "EmoFlow 自动分析"} · ${related.get(message.turn).strategy}]`),
              message.text
            )),
            !messages.length && h("div", { className: "efh-help" }, "当前会话尚无消息")
          ))
        )
      );
    }

    const inject = ["slots", "sessions"];
    function apply(ctx) {
      ctx.slots.inject("conversation.view", () => ctx.slots.register({
        name: "conversation.view", id: "emoflow-emotions", order: 20,
        label: "EmoFlow",
        inject: (sessionId) => {
          const binding = ctx.sessions.binding(sessionId);
          return { sessionId, eventSource: binding?.eventSource, loadOlder: () => binding?.session.loadOlder() };
        }
      }, Dashboard));
    }
    module.exports = { apply, inject, project, projectMessages };
    return module.exports;
  }
});
