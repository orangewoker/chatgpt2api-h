# GPT-6 Sol / Luna 账号路由

从 v1.8.4 起，混合 Plus 与 Free 账号池不再需要都选择通用 `gpt-6`。

| API 模型名 | 默认账号路由 | 说明 |
| --- | --- | --- |
| `gpt-6-sol` | Plus / Pro / ProLite / Team / Enterprise / Edu | 只在对应账号目录包含 GPT-6 时提供 |
| `gpt-6-luna` | Free / Go | 只在对应账号目录包含 GPT-6 时提供 |
| `gpt-6` | 上游宣告支持该 ID 的账号 | 保留原行为，可能在不同套餐间轮询 |
| `auto` | 原有自动选择逻辑 | 不修改 |

## 模型名的含义

- 如果上游已经宣告 `gpt-6-sol` 或 `gpt-6-luna`，直接使用原生 ID。
- 如果上游返回 `gpt-6` 但模型标题明确写有 Sol 或 Luna，优先使用标题区分，不强制按套餐猜测。
- 如果上游只提供通用 `gpt-6`，使用上表的套餐映射生成**本服务的路由别名**。别名用于选账号，请求上游时仍发送 `gpt-6`，不会把本地别名当成新的上游 ID。
- 套餐映射不等于实际底层模型检测；上游可能调整套餐对应的模型。`/v1/models` 中的 `routing_basis` 会标明 `account_plan`（套餐映射）或 `upstream_title`（上游标题），而不是宣称已经验证底层模型。
- 匿名目录中的通用 `gpt-6` 不会凭空生成 Sol/Luna 别名。
- 没有匹配账号时，指定变体的请求会明确失败，不自动降级到其他套餐或匿名请求。
- GPT Image 2 / 2.5 和 Codex 图片模型的来源识别与生成逻辑不修改。

## 调用示例

```json
{
  "model": "gpt-6-sol",
  "messages": [{"role": "user", "content": "你好"}]
}
```

Free 路由使用同样的请求，将 `model` 改成 `gpt-6-luna`。

适用于 `/v1/chat/completions`、`/v1/responses`、`/v1/messages`，包括流式调用。响应、日志、请求缓存保留客户端指定的公开模型名；每次 Token 重试都会重新解析该账号的上游模型 ID。

模型列表额外提供 `display_name` 和 `account_types`，账号页面展示名称与套餐标签，点击仍复制可直接用于 API 的模型 ID。
