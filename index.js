/** Run the existing local EmoFlow engine before each user turn is sent to Harness. */
import { spawn } from "node:child_process";
import { randomUUID } from "node:crypto";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const root = dirname(fileURLToPath(import.meta.url));
const script = join(root, "python", "analyze.py");
const textOf = (content) => Array.isArray(content)
  ? content.filter((part) => part?.type === "text" && typeof part.text === "string")
    .map((part) => part.text).join("\n").trim()
  : "";

async function analyze(text, recent, signal) {
  const python = process.env.EMOFLOW_PYTHON || (process.platform === "win32" ? "python" : "python3");
  return new Promise((resolveResult, reject) => {
    const child = spawn(python, ["-B", "-X", "utf8", script], {
      cwd: root, windowsHide: true, stdio: ["pipe", "pipe", "pipe"], signal,
      env: { ...process.env, DEEPSEEK_API_KEY: "", OPENAI_API_KEY: "" }
    });
    let stdout = "", stderr = "";
    const timeout = setTimeout(() => child.kill(), 8000);
    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (chunk) => { stdout += chunk; if (stdout.length > 32768) child.kill(); });
    child.stderr.on("data", (chunk) => { stderr += chunk; if (stderr.length > 4096) child.kill(); });
    child.on("error", reject);
    child.on("close", (code) => {
      clearTimeout(timeout);
      if (code !== 0) return reject(new Error(`EmoFlow analysis exited ${code}: ${stderr.slice(0, 300)}`));
      try { resolveResult(JSON.parse(stdout)); } catch (error) { reject(error); }
    });
    child.stdin.end(JSON.stringify({ text, recent_user_messages: recent }));
  });
}

export function apply(ctx) {
  ctx.on("agent/pre-step", async ({ agent, messages, turn, step, signal }, next) => {
    const decision = await next();
    if (decision.kind !== "enter" || step !== 1 || signal.aborted) return decision;
    const current = messages.findLast((message) => message.role === "user" && message.source?.kind === "user");
    const text = textOf(current?.content);
    if (!text || text.length > 2000) return decision;
    const recent = agent.session.deriveMessages()
      .filter((message) => message.role === "user" && message.source?.kind === "user")
      .map((message) => textOf(message.content)).filter(Boolean).slice(-4);
    try {
      const analysis = await analyze(text, recent, signal);
      signal.throwIfAborted();
      const confidence = Number(analysis.emotion?.confidence) || 0;
      const guidance = confidence >= 0.6
        ? `情绪线索：${analysis.emotion.primary}，强度 ${analysis.emotion.intensity}，策略：${analysis.strategy.name}。${analysis.reply_guidance}`
        : "本地情绪识别信心不足；请以用户原话为准，自行判断语气，不要宣称用户情绪为中性。";
      const note = {
        id: randomUUID(), role: "user",
        source: { kind: "emoflow-auto", turn, analysis },
        content: [{ type: "text", text: `[EmoFlow 情绪辅助信息，仅用于调整表达，不要向用户复述]\n${guidance}` }]
      };
      return { ...decision, messages: [...decision.messages, note] };
    } catch (error) {
      if (!signal.aborted) ctx.logger.warn(`EmoFlow automatic analysis skipped: ${error.message}`);
      return decision;
    }
  });
}
