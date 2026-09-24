# 本地修改说明（2026-09-24，Yuchen）

基于 `main` 分支提交 `a7e0ba4`。改动集中在三处，均已在本地跑通（6 个询价案例 + 5 轮谈判对话 + 中文询价）。

## 1. 修复中文输入崩溃（编码问题）

**现象：** 客户消息含中文时接口返回 `400 'charmap' codec can't encode`，商机文件写不进去；Windows 上保存的文件在 Linux 服务器上也可能读不出来。

**原因：** 所有 `Path.read_text()` / `Path.write_text()` 都没有指定编码，Python 会使用系统默认编码（Windows 中文/英文系统是 cp1252 或 GBK，Linux 是 UTF-8），跨机器不一致。

**改动：** 27 处文件读写全部加上 `encoding="utf-8"`：

| 文件 | 处数 |
|---|---|
| `agent.py` | 11 |
| `infrastructure/knowledge_store.py` | 14 |
| `infrastructure/state_store.py` | 2 |

另外 `agent.py` 的 `_load_opportunities()` 在读取商机文件时补充捕获 `UnicodeDecodeError`，遇到旧的坏文件跳过而不是启动崩溃。

## 2. 大模型调用改为可切换（新增 Anthropic 分支）

**背景：** 原代码只支持 DeepSeek（OpenAI 兼容格式）。比赛只允许 AWS Bedrock 上的 Claude Sonnet 4.5，所以需要换成 Claude 系列的调用方式。这次先接了 Anthropic 官方 API 做验证，**Bedrock 用的是同一个 SDK，之后只需换客户端（见第 3 节）**。

**改动位置（全部在 `agent.py`）：**

- `CorporateEventAgent.__init__()`（约第 105–113 行）：新增 `self.anthropic_key` / `self.anthropic_model`，读取环境变量 `ANTHROPIC_API_KEY` 和 `ANTHROPIC_MODEL`（默认 `claude-opus-5`）。优先级：有 `ANTHROPIC_API_KEY` 用 Claude → 否则有 `DEEPSEEK_API_KEY` 用 DeepSeek → 都没有则退回纯规则层。`.env.example` 里的占位值 `replace_with_a_new_key` 现在视为"未配置"。
- `_llm()`（约第 255 行）：开头加一行，有 Anthropic key 就转给 `_llm_anthropic()`；DeepSeek 路径原样保留。
- 新增 `_llm_anthropic(system, user, json_mode)`（约第 270–285 行）：用官方 `anthropic` SDK 调 `client.messages.create()`，返回纯文本；`json_mode=True` 时在 system prompt 末尾追加"只返回一个 JSON 对象"，并去掉可能出现的 ```` ```json ```` 围栏。调用失败时打印日志并返回 `None`，和原来 DeepSeek 路径的行为一致。
- 启动横幅（`main()`）：现在显示实际使用的模型（`anthropic:claude-opus-5` / `deepseek:deepseek-chat`），key 无效或为占位值时显示 DISABLED。原来只要 key 非空就显示 enabled，会误导。

**调用点没有改动：** 理解消息（`_llm_interpretation`）、写客户回复（`_customer_reply`）、总结人工反馈（`reflect`）、导入历史邮件（`ingestion`）四处都只调用 `_llm()`，不关心背后是哪家模型，所以提示词和流程全部不用动。

**依赖：** `requirements.txt` 新增 `anthropic>=1.8`。

**本地运行方式：** 把 `ANTHROPIC_API_KEY` 设为环境变量（或写进 `.env`，该文件已被 gitignore），然后 `python agent.py serve --port 8080`。**不要把 key 写进任何会提交的文件。**

## 3. 之后接比赛官方 API（AWS Bedrock）要改哪里

只需要改 `_llm_anthropic()` 里创建客户端的那一行，其余不动。具体格式以组委会在 Slack 上发的 JSON 说明为准，两种可能：

**情况 A：组委会给的是 Bedrock API key（Bearer token）+ HTTPS 端点**

```python
# 替换 _llm_anthropic() 中的 client = anthropic.Anthropic(...) 一行：
client = anthropic.AnthropicBedrockMantle(aws_region="ap-southeast-1")   # 区域按组委会说明
# 模型名改为 Bedrock 格式，通过 ANTHROPIC_MODEL 环境变量传入，例如：
#   ANTHROPIC_MODEL=anthropic.claude-sonnet-4-5
```

`AnthropicBedrockMantle` 是 anthropic SDK 自带的 Bedrock 客户端，`messages.create()` 的用法和现在完全一样。认证方式（Bearer key 还是 AWS 凭证）按组委会文档配置环境变量。

**情况 B：组委会要求直接按他们给的 JSON 格式发 HTTP 请求**

那就在 `_llm_anthropic()` 旁边再写一个 `_llm_bedrock()`，用 `urllib`（原 DeepSeek 路径的写法）把 `system` + `user` 拼成组委会规定的 JSON 发到指定 URL，取回文本。`_llm()` 开头加一个判断（有 `BEDROCK_API_KEY` 就走这条）即可。

**换完后要做的验证：** 重跑同样的 6 个询价案例 + 5 轮谈判对话（测试脚本见测试报告），对比提取结果和回复质量；Sonnet 4.5 比 Opus 5 快，但输出风格可能略有差异，提示词可能要微调。

## 4. 测试时发现、尚未改的问题

1. 回复偶尔以 `# Draft Email` / `**Subject:**` 等 Markdown 开头，直接发邮件不合适 → `_customer_reply()` 的 system prompt 加一句"纯文本，不要 Markdown"。
2. `_merge_interpretation()` 会把 AI 提取的 `commercial_updates`（客户预算、议价备注）写进 `commercial` 状态。README 说 commercial 只能由人工批准修改。内容只是备注不是改价，但严格来说违反设计原则，请 Kevin 确认是否有意为之。
3. README 的命令是 Mac/Linux 写法（`python3`、`cp`、`source .venv/bin/activate`），Windows 上跑不了；`http://localhost:8080/` 首页实际返回 404。
4. 没有 key 时的回退回复会把内部指令原文（"Thank the customer, confirm what is already known…"）直接发给客户，且追问固定。上线后不会遇到，但 API 出错时会暴露。

## 5. 未包含在本次改动中的文件

- `.gitignore` 加了 `.venv/`（本地虚拟环境）。
- `data/` 是运行时数据，本来就不提交。
