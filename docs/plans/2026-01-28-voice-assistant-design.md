# Claude Code 语音助手设计

## 概述

将 cc-stt 从简单的 MCP 语音输入工具扩展为语音助手，支持唤醒词触发、语音编辑、自动发送到 Claude Code。

## 功能流程

```
┌─────────────────────────────────────────────────┐
│           后台守护进程 (daemon)                   │
│   OpenWakeWord 持续监听 → 检测 "alexa"           │
└─────────────────────┬───────────────────────────┘
                      ↓ 唤醒
┌─────────────────────────────────────────────────┐
│              录音 + 转写                          │
│   AudioRecorder → FunASR → 文本                  │
└─────────────────────┬───────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│           编辑窗口 (Tkinter)                      │
│   - 浮动窗口，弹出时抢焦点                         │
│   - 显示转录结果                                  │
│   - 语音指令编辑（Ollama + Qwen2.5:3b）           │
│   - 重新录音替换                                  │
│   - 键盘直接编辑                                  │
│   - 确认：Enter 或语音 "确认"                     │
│   - 取消：Esc 或语音 "取消"                       │
└─────────────────────┬───────────────────────────┘
                      ↓ 确认
┌─────────────────────────────────────────────────┐
│              自动发送                             │
│   写入剪贴板 → xdotool 聚焦终端                   │
│   → Ctrl+Shift+V 粘贴 → Enter 提交               │
└─────────────────────────────────────────────────┘
```

## 技术选型

| 组件 | 选择 | 说明 |
|------|------|------|
| 唤醒词引擎 | OpenWakeWord | 开源，预训练 alexa 模型 |
| 录音 | sounddevice | 复用现有 recorder |
| ASR | FunASR | 复用现有 transcriber |
| GUI | Tkinter | 轻量，awesome WM 兼容 |
| 语音编辑 LLM | Ollama + Qwen2.5:3b | 本地部署，6GB 显存 |
| 自动发送 | xdotool + pyperclip | X11 环境，xfce4-terminal |
| 进程管理 | systemd user service | 开机自启 |

## 模块设计

### 新增模块

```
src/cc_stt/
├── daemon.py       # 守护进程主循环
├── wakeword.py     # OpenWakeWord 封装
├── editor.py       # Tkinter 编辑窗口
├── voice_edit.py   # 语音指令处理（Ollama）
└── sender.py       # xdotool 自动发送
```

### daemon.py

守护进程主循环：

```python
class Daemon:
    def __init__(self):
        self.wakeword = WakewordDetector("alexa")
        self.recorder = AudioRecorder()
        self.transcriber = SpeechTranscriber()
        self.editor = None

    def run(self):
        while True:
            if self.wakeword.detected():
                audio = self.recorder.record()
                text = self.transcriber.transcribe(audio)
                self.show_editor(text)

    def show_editor(self, text):
        # 启动 Tkinter 窗口（需在主线程）
        ...
```

### wakeword.py

OpenWakeWord 封装：

```python
from openwakeword.model import Model

class WakewordDetector:
    def __init__(self, wakeword: str = "alexa"):
        self.model = Model(wakeword_models=[wakeword])
        self.stream = None  # sounddevice InputStream

    def start(self):
        # 开始持续监听
        ...

    def detected(self) -> bool:
        # 检测到唤醒词返回 True
        ...
```

### editor.py

Tkinter 编辑窗口：

```python
import tkinter as tk

class EditorWindow:
    def __init__(self, text: str, on_confirm, on_cancel):
        self.root = tk.Tk()
        self.root.title("语音输入")
        self.root.attributes("-topmost", True)  # 置顶
        self.root.focus_force()  # 抢焦点

        # 文本编辑区
        self.text_area = tk.Text(self.root)
        self.text_area.insert("1.0", text)

        # 绑定快捷键
        self.root.bind("<Return>", self.confirm)
        self.root.bind("<Escape>", self.cancel)

    def confirm(self, event=None):
        text = self.text_area.get("1.0", "end-1c")
        self.on_confirm(text)
        self.root.destroy()

    def cancel(self, event=None):
        self.on_cancel()
        self.root.destroy()

    def run(self):
        self.root.mainloop()
```

### voice_edit.py

语音指令处理：

```python
import ollama

class VoiceEditor:
    def __init__(self, model: str = "qwen2.5:3b"):
        self.model = model

    def apply_edit(self, original: str, instruction: str) -> str:
        prompt = f"""原文：{original}

用户指令：{instruction}

请根据指令修改原文，只返回修改后的文本，不要解释。"""

        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"]
```

### sender.py

自动发送到 Claude Code：

```python
import subprocess
import pyperclip

class Sender:
    def __init__(self, terminal_class: str = "xfce4-terminal"):
        self.terminal_class = terminal_class

    def send(self, text: str):
        # 写入剪贴板
        pyperclip.copy(text)

        # 聚焦终端窗口
        subprocess.run([
            "xdotool", "search", "--class", self.terminal_class,
            "windowactivate"
        ])

        # 粘贴
        subprocess.run(["xdotool", "key", "ctrl+shift+v"])

        # 回车提交
        subprocess.run(["xdotool", "key", "Return"])
```

## systemd 配置

`~/.config/systemd/user/cc-stt-daemon.service`:

```ini
[Unit]
Description=CC-STT Voice Assistant Daemon
After=graphical-session.target

[Service]
Type=simple
ExecStart=/home/jiang/cc/cc-stt/.venv/bin/python -m cc_stt.daemon
Restart=on-failure
Environment=DISPLAY=:0

[Install]
WantedBy=default.target
```

启用：

```bash
systemctl --user enable cc-stt-daemon
systemctl --user start cc-stt-daemon
```

## 依赖

新增依赖：

```toml
[project.dependencies]
openwakeword = "^0.6"
pyperclip = "^1.8"
ollama = "^0.4"
```

系统依赖：

```bash
sudo apt install xdotool
```

## awesome WM 配置

`~/.config/awesome/rc.lua` 添加浮动规则：

```lua
ruled.client.append_rule {
    rule = { class = "Tk" },
    properties = { floating = true, placement = awful.placement.centered }
}
```

## 扩展预留

架构设计为可插拔，未来扩展方向：

1. **TTS 语音回复** - 添加 output 模块
2. **多轮对话** - 添加 session 状态管理
3. **系统控制** - 添加 action 模块
4. **其他 LLM 后端** - 抽象 LLM 接口

## 实现顺序

1. wakeword.py - OpenWakeWord 集成
2. daemon.py - 守护进程框架
3. editor.py - Tkinter 编辑窗口
4. sender.py - xdotool 自动发送
5. voice_edit.py - 语音指令编辑
6. systemd 配置 - 服务化部署
7. 集成测试
