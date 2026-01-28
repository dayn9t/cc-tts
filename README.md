# cc-stt

Claude Code 语音输入工具，基于 FunASR。

## 安装

```bash
uv sync
uv pip install openwakeword==0.4.0
```

## 使用方式

### 方式 1: MCP 服务器（推荐用于 Claude Code）

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

然后在 Claude Code 中使用 MCP 工具：

#### transcribe

录音并转文字。

- `max_duration`: 最大录音时长（秒），默认 30

#### configure_hotwords

更新热词配置。

- `hotwords`: 热词数组
- `mode`: `replace` 或 `append`（默认）

### 方式 2: 语音助手守护进程

启动后台语音助手，通过唤醒词控制：

```bash
# 直接运行
uv run cc-stt-daemon

# 或使用 systemd 服务（推荐）
./scripts/install-service.sh
systemctl --user start cc-stt-daemon
```

**工作流程：**

1. 说唤醒词（默认 "alexa"）
2. 听到提示后开始说话
3. 转写完成后弹出编辑窗口
4. 确认后自动发送到剪贴板，粘贴到当前应用

**systemd 服务命令：**

```bash
systemctl --user start cc-stt-daemon    # 启动
systemctl --user stop cc-stt-daemon     # 停止
systemctl --user status cc-stt-daemon   # 状态
journalctl --user -u cc-stt-daemon -f   # 日志
```

## 配置文件

- `~/.config/cc-stt/config.json` - 主配置
- `~/.config/cc-stt/hotwords.txt` - 热词列表

## 测试

```bash
uv run pytest tests/ -v
```
