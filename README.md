# cc-stt

Claude Code 语音输入工具，基于 FunASR。

## 安装

```bash
uv sync
```

## 配置 Claude Code

编辑 `~/.claude/settings.json`：

```json
{
  "mcpServers": {
    "cc-stt": {
      "command": "uv",
      "args": ["--directory", "/path/to/cc-stt", "run", "cc-stt"]
    }
  }
}
```

## MCP 工具

### transcribe

录音并转文字。

- `max_duration`: 最大录音时长（秒），默认 30

### configure_hotwords

更新热词配置。

- `hotwords`: 热词数组
- `mode`: `replace` 或 `append`（默认）

## 配置文件

- `~/.config/cc-stt/config.json` - 主配置
- `~/.config/cc-stt/hotwords.txt` - 热词列表

## 测试

```bash
uv run pytest tests/ -v
```
