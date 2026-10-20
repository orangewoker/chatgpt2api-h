# Gemini Web 文本与视觉对话

使用号池中已配置的 Gemini Web Cookie，直接通过 HTTP 调用网页后端；不依赖浏览器模拟点击，不使用 Gemini 官方 API Key。原有 Gemini 生图、参考图编辑和全尺寸下载链路保持独立。

## 模型

| 网页模型 | 低档 | 中档 | 高档 |
| --- | --- | --- | --- |
| 3.8 Flash | `gemini-3.8-flash-low` | `gemini-3.8-flash-medium` | `gemini-3.8-flash-high` |
| 4 Argon | `gemini-4.0-argon-low` | `gemini-4.0-argon-medium` | `gemini-4.0-argon-high` |

`GET /v1/models` 从有效 Gemini 账号的网页模型目录发现可用版本，目录缓存 5 分钟，不发布猜测的模型。仅识别 3.8 Flash 和 4/4.0 Argon，不自动替换成其他版本。模型 ID 固定思考档位；同时传入 `reasoning_effort`、`thinking_effort` 或 `reasoning.effort` 时必须与 ID 一致，否则返回 400。

账号选择与 GPT 隔离，不占用图片并发槽或扣减本地图片配额。鉴权/限流失败会短暂冷却该账号；切换账号时仍保持同一模型与档位，已经输出文本后不重试、不拼接其他账号的回答。

## Chat Completions

`POST /v1/chat/completions`，使用现有服务的 Bearer API 密钥：

```json
{
  "model": "gemini-3.8-flash-medium",
  "stream": true,
  "messages": [
    {"role": "system", "content": "请简洁地用中文回答。"},
    {"role": "user", "content": [
      {"type": "text", "text": "这张图片是什么颜色、什么形状？"},
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
    ]}
  ]
}
```

## Responses

`POST /v1/responses`：

```json
{
  "model": "gemini-4.0-argon-high",
  "stream": false,
  "instructions": "请简洁地用中文回答。",
  "input": [{"role": "user", "content": [
    {"type": "input_text", "text": "描述这张图片。"},
    {"type": "input_image", "image_url": "data:image/png;base64,..."}
  ]}]
}
```

两个接口均支持纯文本、图片 URL/data URL、多个图片附件、完整历史消息，以及 `stream: true/false`。图片作为真实附件上传，不会因为有图片输入而切到生图。流式读取上游增量，不在本地人为拆字；很短的回答可能只有一个文本增量。

## 视频输入与音频边界

新增 MP4/WebM 视频附件：单个文件最大 20MB，所有附件合计不超过 50MB、最多 10 个。当前仅支持内联 Base64，不支持远程音视频 URL 或 `file_id`；这属于本适配器限制，不代表模型官方限制。

Chat Completions 的 content 可使用：

```json
{"type": "video_url", "video_url": {"url": "data:video/mp4;base64,..."}}
```

Responses 的 content 可使用：

```json
{"type": "input_file", "filename": "clip.mp4", "file_data": "data:video/mp4;base64,..."}
```

原生音频暂未开放：WAV/MP3 上传虽能成功，但当前 Cookie 实测没有可靠识别音轨，出现无音频或错误转写。接口明确返回 `invalid_media`，不会静默忽略音频并输出貌似成功的回答。可暂时发送转写文本；不能据此断言 Argon 模型本身不支持音频。

## 兼容边界

- 普通单轮直接发送用户原文，不再包装成 JSON。服务在同账号、同模型、完整历史精确匹配时接续网页会话，只发送最新用户消息及附件。会话标识仅存内存，最长 30 分钟、最多 256 条；重启、过期、切换账号或导入历史时回退为可读的角色文本，不再使用 JSON。System 指令仍不是独立的网页优先级通道。多轮仍需每次提交完整历史。
- 不执行函数/工具调用。客户端附带可选工具且 `tool_choice` 为 `auto`/`none` 时，选择普通文本回答，忽略工具定义，不转交 GPT；强制工具调用、`image_generation`、`previous_response_id` 明确返回 400。
- `modalities: ["image"]` 表示图片**输出**，不是识图能力开关，仍返回明确 400。识图使用 `image_url` / `input_image`，省略 modalities 或设为 `["text"]`。生图请继续使用原有图片模型。
- 不返回隐藏思考内容。`usage` 延用兼容层的估算，不是 Google 官方计费 token 数。
- `temperature`、`top_p`、`max_tokens` 等采样控制不映射到网页参数，不能视为上游保证；结构化 JSON 输出同样不保证。
- 如果启用了服务原有的对话缓存/历史去重设置，其行为仍然适用；需要保留所有 assistant 历史时不要启用删除 assistant 历史的选项。

## 验证（2026-10-20）

- 用登录 Cookie 直接调用上游，六个模型 ID 均获得真实回答，没有模拟点击发送。
- Flash High 经 Responses 流式处理、Argon Low 经 Chat Completions 非流式处理，均正确识别合成参考图中的红色方形。
- 真实测试经过本地协议 handler；HTTP 路由与 SSE 封装另用 FastAPI TestClient 配合模拟上游回归测试。测试 Cookie 仅驻内存，不写入号池或日志。
- 修复后 Argon Low 两轮实测正确记住测试口令，第二轮确认使用了原生会话标识；携带可选 web_search 工具定义的图片请求正确识别红色图片；MP4 视频正确识别先红后蓝。音频实测失败，因此未宣称原生音频支持。
