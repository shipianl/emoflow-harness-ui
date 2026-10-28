# EmoFlow for DeepSeek Harness

这是可独立安装的 DeepSeek Harness bundle。它在每轮用户消息进入模型前运行本地 EmoFlow 四层引擎，并在会话的 **EmoFlow** 标签页展示 PAD、趋势、风险和轨迹。模型收到简短的语气与策略建议，最终回复仍由 Harness 当前模型生成。

作者：[@shipianl](https://github.com/shipianl)。本项目允许下载、安装和运行；再发布及发布改版需取得作者书面许可，详见 [LICENSE.md](LICENSE.md)。

## 要求

- DeepSeek Harness `>=0.2.1-rc.2 <0.3.0`。
- Node.js 20+。
- Python 3.11+，在系统 `PATH` 中可用；也可设置环境变量 `EMOFLOW_PYTHON` 为 Python 可执行文件的绝对路径。

运行中的每条用户消息会在本机启动一个短暂的 Python 子进程分析，耗时主要由 Python 启动决定。插件不另行调用 DeepSeek API，也不读取模型密钥。分析结果和辅助信息保存在 Harness 当前会话日志中；导出会话或轨迹时会包含情绪数据。停用 bundle 可停止后续自动分析，但不会删除已有会话记录。

## 本地安装

可直接安装发行压缩包（以下路径请换成实际保存位置）：

```powershell
dsh plugin --profile web add 'D:\path\to\emoflow-harness-ui-0.1.0.tgz'
dsh --profile web
```

开发时也可把安装目标改为本目录 `market_plugin`。若 Python 不在系统 `PATH`，先设置 `EMOFLOW_PYTHON` 为其可执行文件的绝对路径，再启动 Harness。安装后新建会话，在顶部打开 **EmoFlow** 标签。旧会话不会回算。

发布到 npm 或 Git 仓库后，市场索引可指向固定版本或固定提交。当前 Harness 的自定义市场是实验功能：在 **插件 → 更多 → 实验性插件 → 自定义插件市场** 启用，再在“发现”页添加市场索引地址。没有默认的官方市场或统一上架入口。

## 能力边界

- 情绪感知是本地关键词快通道；复杂语义和反讽可能识别不准。
- Q 学习、真实手表、模拟传感器 MCP 工具不包含在此独立 bundle 中。
- 情绪标签、风险分数和建议不构成医疗判断。
- 每轮的辅助信息会影响模型可见上下文，但不能保证模型逐字执行策略。

`python/analyze.py` 只依赖 Python 标准库和包内的两个 EmoFlow 源文件，不需要 pip 安装额外依赖。
