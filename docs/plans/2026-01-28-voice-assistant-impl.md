# Voice Assistant Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将 cc-stt 扩展为语音助手，支持唤醒词触发、语音编辑、自动发送到 Claude Code。

**Architecture:** 后台守护进程持续监听唤醒词 "alexa"，检测到后录音转写，弹出 Tkinter 编辑窗口供用户确认/编辑，确认后通过 xdotool 自动发送到 Claude Code。

**Tech Stack:** OpenWakeWord, Tkinter, Ollama (qwen2.5:3b), xdotool, pyperclip, systemd

---

## Task 1: 添加新依赖

**Files:**
- Modify: `pyproject.toml:10-16`

**Step 1: 更新 pyproject.toml 添加依赖**

```toml
dependencies = [
    "mcp[cli]>=1.0.0",
    "funasr>=1.0.0",
    "sounddevice>=0.4.6",
    "numpy>=1.24.0",
    "modelscope>=1.9.0",
    "openwakeword>=0.6.0",
    "pyperclip>=1.8.0",
    "ollama>=0.4.0",
]
```

**Step 2: 同步依赖**

Run: `uv sync`
Expected: 依赖安装成功

**Step 3: 验证依赖可导入**

Run: `uv run python -c "import openwakeword; import pyperclip; import ollama; print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add openwakeword, pyperclip, ollama dependencies"
```

---

## Task 2: 实现 WakewordDetector

**Files:**
- Create: `src/cc_stt/wakeword.py`
- Create: `tests/test_wakeword.py`

**Step 1: 写失败测试**

`tests/test_wakeword.py`:
```python
import numpy as np
from cc_stt.wakeword import WakewordDetector


def test_wakeword_detector_init():
    detector = WakewordDetector(wakeword="alexa")
    assert detector.wakeword == "alexa"
    assert detector.model is not None


def test_wakeword_detector_process_audio():
    detector = WakewordDetector(wakeword="alexa")
    # 静音音频，不应触发
    silence = np.zeros(1280, dtype=np.float32)
    result = detector.process_audio(silence)
    assert result is False
```

**Step 2: 运行测试验证失败**

Run: `uv run pytest tests/test_wakeword.py -v`
Expected: FAIL with "No module named 'cc_stt.wakeword'"

**Step 3: 实现 wakeword.py**

`src/cc_stt/wakeword.py`:
```python
import numpy as np
from openwakeword.model import Model


class WakewordDetector:
    def __init__(self, wakeword: str = "alexa", threshold: float = 0.5):
        self.wakeword = wakeword
        self.threshold = threshold
        self.model = Model(wakeword_models=[wakeword])

    def process_audio(self, audio: np.ndarray) -> bool:
        """处理音频帧，返回是否检测到唤醒词"""
        prediction = self.model.predict(audio)
        score = prediction.get(self.wakeword, 0)
        if score > self.threshold:
            self.model.reset()
            return True
        return False

    def reset(self):
        """重置检测状态"""
        self.model.reset()
```

**Step 4: 运行测试验证通过**

Run: `uv run pytest tests/test_wakeword.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/cc_stt/wakeword.py tests/test_wakeword.py
git commit -m "feat: add WakewordDetector with OpenWakeWord"
```

---

## Task 3: 实现 Sender

**Files:**
- Create: `src/cc_stt/sender.py`
- Create: `tests/test_sender.py`

**Step 1: 写失败测试**

`tests/test_sender.py`:
```python
from unittest.mock import patch, call
from cc_stt.sender import Sender


def test_sender_init():
    sender = Sender(terminal_class="xfce4-terminal")
    assert sender.terminal_class == "xfce4-terminal"


@patch("cc_stt.sender.subprocess.run")
@patch("cc_stt.sender.pyperclip.copy")
def test_sender_send(mock_copy, mock_run):
    sender = Sender(terminal_class="xfce4-terminal")
    sender.send("hello world")

    mock_copy.assert_called_once_with("hello world")
    assert mock_run.call_count == 3  # windowactivate, paste, enter
```

**Step 2: 运行测试验证失败**

Run: `uv run pytest tests/test_sender.py -v`
Expected: FAIL with "No module named 'cc_stt.sender'"

**Step 3: 实现 sender.py**

`src/cc_stt/sender.py`:
```python
import subprocess
import pyperclip


class Sender:
    def __init__(self, terminal_class: str = "xfce4-terminal"):
        self.terminal_class = terminal_class

    def send(self, text: str):
        """发送文本到终端"""
        pyperclip.copy(text)

        # 聚焦终端窗口
        subprocess.run(
            ["xdotool", "search", "--class", self.terminal_class, "windowactivate"],
            check=False
        )

        # 粘贴
        subprocess.run(["xdotool", "key", "ctrl+shift+v"], check=False)

        # 回车提交
        subprocess.run(["xdotool", "key", "Return"], check=False)
```

**Step 4: 运行测试验证通过**

Run: `uv run pytest tests/test_sender.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/cc_stt/sender.py tests/test_sender.py
git commit -m "feat: add Sender for xdotool auto-input"
```

---

## Task 4: 实现 VoiceEditor

**Files:**
- Create: `src/cc_stt/voice_edit.py`
- Create: `tests/test_voice_edit.py`

**Step 1: 写失败测试**

`tests/test_voice_edit.py`:
```python
from unittest.mock import patch, MagicMock
from cc_stt.voice_edit import VoiceEditor


def test_voice_editor_init():
    editor = VoiceEditor(model="qwen2.5:3b")
    assert editor.model == "qwen2.5:3b"


@patch("cc_stt.voice_edit.ollama.chat")
def test_voice_editor_apply_edit(mock_chat):
    mock_chat.return_value = {"message": {"content": "Hello World"}}

    editor = VoiceEditor()
    result = editor.apply_edit("hello world", "把 hello 改成 Hello")

    assert result == "Hello World"
    mock_chat.assert_called_once()
```

**Step 2: 运行测试验证失败**

Run: `uv run pytest tests/test_voice_edit.py -v`
Expected: FAIL with "No module named 'cc_stt.voice_edit'"

**Step 3: 实现 voice_edit.py**

`src/cc_stt/voice_edit.py`:
```python
import ollama


class VoiceEditor:
    def __init__(self, model: str = "qwen2.5:3b"):
        self.model = model

    def apply_edit(self, original: str, instruction: str) -> str:
        """根据语音指令编辑文本"""
        prompt = f"""原文：{original}

用户指令：{instruction}

请根据指令修改原文，只返回修改后的文本，不要解释。"""

        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"].strip()
```

**Step 4: 运行测试验证通过**

Run: `uv run pytest tests/test_voice_edit.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/cc_stt/voice_edit.py tests/test_voice_edit.py
git commit -m "feat: add VoiceEditor with Ollama integration"
```

---

## Task 5: 实现 EditorWindow

**Files:**
- Create: `src/cc_stt/editor.py`
- Create: `tests/test_editor.py`

**Step 1: 写失败测试**

`tests/test_editor.py`:
```python
from unittest.mock import MagicMock, patch


def test_editor_window_init():
    with patch("cc_stt.editor.tk.Tk") as mock_tk:
        mock_root = MagicMock()
        mock_tk.return_value = mock_root

        from cc_stt.editor import EditorWindow

        on_confirm = MagicMock()
        on_cancel = MagicMock()
        editor = EditorWindow("test text", on_confirm, on_cancel)

        assert editor.on_confirm == on_confirm
        assert editor.on_cancel == on_cancel
        mock_root.title.assert_called_once()
        mock_root.attributes.assert_called()
```

**Step 2: 运行测试验证失败**

Run: `uv run pytest tests/test_editor.py -v`
Expected: FAIL with "No module named 'cc_stt.editor'"

**Step 3: 实现 editor.py**

`src/cc_stt/editor.py`:
```python
import tkinter as tk
from tkinter import scrolledtext
from typing import Callable


class EditorWindow:
    def __init__(
        self,
        text: str,
        on_confirm: Callable[[str], None],
        on_cancel: Callable[[], None]
    ):
        self.on_confirm = on_confirm
        self.on_cancel = on_cancel

        self.root = tk.Tk()
        self.root.title("语音输入")
        self.root.geometry("500x300")
        self.root.attributes("-topmost", True)
        self.root.focus_force()

        # 文本编辑区
        self.text_area = scrolledtext.ScrolledText(
            self.root, wrap=tk.WORD, font=("monospace", 12)
        )
        self.text_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.text_area.insert("1.0", text)
        self.text_area.focus_set()

        # 按钮区
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        tk.Button(btn_frame, text="确认 (Enter)", command=self._confirm).pack(
            side=tk.LEFT, padx=5
        )
        tk.Button(btn_frame, text="取消 (Esc)", command=self._cancel).pack(
            side=tk.LEFT, padx=5
        )

        # 绑定快捷键
        self.root.bind("<Return>", lambda e: self._confirm())
        self.root.bind("<Escape>", lambda e: self._cancel())

    def _confirm(self):
        text = self.text_area.get("1.0", "end-1c")
        self.root.destroy()
        self.on_confirm(text)

    def _cancel(self):
        self.root.destroy()
        self.on_cancel()

    def update_text(self, text: str):
        """更新文本内容"""
        self.text_area.delete("1.0", tk.END)
        self.text_area.insert("1.0", text)

    def run(self):
        self.root.mainloop()
```

**Step 4: 运行测试验证通过**

Run: `uv run pytest tests/test_editor.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/cc_stt/editor.py tests/test_editor.py
git commit -m "feat: add EditorWindow with Tkinter"
```

---

## Task 6: 实现 Daemon

**Files:**
- Create: `src/cc_stt/daemon.py`
- Create: `tests/test_daemon.py`

**Step 1: 写失败测试**

`tests/test_daemon.py`:
```python
from unittest.mock import MagicMock, patch


def test_daemon_init():
    with patch("cc_stt.daemon.WakewordDetector") as mock_wakeword, \
         patch("cc_stt.daemon.AudioRecorder") as mock_recorder, \
         patch("cc_stt.daemon.SpeechTranscriber") as mock_transcriber, \
         patch("cc_stt.daemon.Config") as mock_config:

        mock_config.load.return_value = MagicMock()

        from cc_stt.daemon import Daemon
        daemon = Daemon()

        assert daemon.wakeword is not None
        assert daemon.recorder is not None
        assert daemon.transcriber is not None
```

**Step 2: 运行测试验证失败**

Run: `uv run pytest tests/test_daemon.py -v`
Expected: FAIL with "No module named 'cc_stt.daemon'"

**Step 3: 实现 daemon.py**

`src/cc_stt/daemon.py`:
```python
import numpy as np
import sounddevice as sd
from cc_stt.config import Config
from cc_stt.wakeword import WakewordDetector
from cc_stt.recorder import AudioRecorder
from cc_stt.transcriber import SpeechTranscriber
from cc_stt.editor import EditorWindow
from cc_stt.voice_edit import VoiceEditor
from cc_stt.sender import Sender


class Daemon:
    def __init__(self, wakeword: str = "alexa"):
        self.config = Config.load()
        self.wakeword = WakewordDetector(wakeword=wakeword)
        self.recorder = AudioRecorder(
            sample_rate=self.config.audio.sample_rate,
            channels=self.config.audio.channels
        )
        self.transcriber = SpeechTranscriber(self.config.model.name)
        self.voice_editor = VoiceEditor()
        self.sender = Sender()
        self.running = False
        self.current_text = ""

    def _on_confirm(self, text: str):
        """确认回调"""
        if text.strip():
            self.sender.send(text)

    def _on_cancel(self):
        """取消回调"""
        pass

    def _show_editor(self, text: str):
        """显示编辑窗口"""
        self.current_text = text
        editor = EditorWindow(text, self._on_confirm, self._on_cancel)
        editor.run()

    def _audio_callback(self, indata, frames, time, status):
        """音频流回调"""
        audio = indata[:, 0].astype(np.float32)
        if self.wakeword.process_audio(audio):
            raise sd.CallbackStop()

    def run(self):
        """主循环"""
        self.running = True
        print(f"语音助手已启动，唤醒词: {self.wakeword.wakeword}")

        while self.running:
            try:
                # 监听唤醒词
                with sd.InputStream(
                    samplerate=self.config.audio.sample_rate,
                    channels=self.config.audio.channels,
                    callback=self._audio_callback,
                    blocksize=1280
                ):
                    sd.sleep(int(3600 * 1000))  # 持续监听
            except sd.CallbackStop:
                print("检测到唤醒词，开始录音...")

                # 录音
                audio = self.recorder.record(
                    max_duration=self.config.audio.max_duration,
                    silence_threshold=self.config.audio.silence_threshold,
                    silence_duration=self.config.audio.silence_duration
                )

                if len(audio) > 0:
                    # 转写
                    text = self.transcriber.transcribe(
                        audio, self.config.audio.sample_rate
                    )
                    print(f"转写结果: {text}")

                    if text.strip():
                        self._show_editor(text)

                self.wakeword.reset()

    def stop(self):
        """停止守护进程"""
        self.running = False


def main():
    daemon = Daemon()
    try:
        daemon.run()
    except KeyboardInterrupt:
        daemon.stop()
        print("\n语音助手已停止")


if __name__ == "__main__":
    main()
```

**Step 4: 运行测试验证通过**

Run: `uv run pytest tests/test_daemon.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/cc_stt/daemon.py tests/test_daemon.py
git commit -m "feat: add Daemon for wakeword listening loop"
```

---

## Task 7: 添加 CLI 入口

**Files:**
- Modify: `src/cc_stt/__init__.py`
- Modify: `pyproject.toml:18-19`

**Step 1: 更新 __init__.py**

`src/cc_stt/__init__.py`:
```python
import asyncio
from .server import main as server_main
from .daemon import main as daemon_main


def cli():
    """MCP server CLI entry point"""
    asyncio.run(server_main())


def daemon_cli():
    """Voice assistant daemon CLI entry point"""
    daemon_main()


__all__ = ["cli", "daemon_cli", "server_main", "daemon_main"]
```

**Step 2: 更新 pyproject.toml 添加新入口**

```toml
[project.scripts]
cc-stt = "cc_stt:cli"
cc-stt-daemon = "cc_stt:daemon_cli"
```

**Step 3: 验证入口可用**

Run: `uv run cc-stt-daemon --help || echo "Entry point registered"`
Expected: 入口已注册（会因缺少参数报错或直接运行）

**Step 4: Commit**

```bash
git add src/cc_stt/__init__.py pyproject.toml
git commit -m "feat: add cc-stt-daemon CLI entry point"
```

---

## Task 8: 创建 systemd 服务

**Files:**
- Create: `systemd/cc-stt-daemon.service`

**Step 1: 创建 systemd 目录和服务文件**

`systemd/cc-stt-daemon.service`:
```ini
[Unit]
Description=CC-STT Voice Assistant Daemon
After=graphical-session.target

[Service]
Type=simple
ExecStart=%h/cc/cc-stt/.venv/bin/cc-stt-daemon
Restart=on-failure
RestartSec=5
Environment=DISPLAY=:0

[Install]
WantedBy=default.target
```

**Step 2: 创建安装脚本**

`scripts/install-service.sh`:
```bash
#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

mkdir -p ~/.config/systemd/user
cp "$PROJECT_DIR/systemd/cc-stt-daemon.service" ~/.config/systemd/user/

systemctl --user daemon-reload
systemctl --user enable cc-stt-daemon

echo "Service installed. Start with: systemctl --user start cc-stt-daemon"
```

**Step 3: Commit**

```bash
mkdir -p systemd scripts
git add systemd/cc-stt-daemon.service scripts/install-service.sh
git commit -m "feat: add systemd user service for daemon"
```

---

## Task 9: 更新 README

**Files:**
- Modify: `README.md`

**Step 1: 更新 README**

`README.md`:
```markdown
# cc-stt

Claude Code 语音助手，基于 FunASR + OpenWakeWord。

## 功能

- **语音唤醒**：说 "alexa" 触发录音
- **语音转文字**：FunASR 实时转写
- **编辑确认**：Tkinter 窗口编辑/确认
- **自动发送**：确认后自动发送到 Claude Code

## 安装

```bash
uv sync
```

系统依赖：

```bash
sudo apt install xdotool
```

## 使用

### 方式一：语音助手（推荐）

```bash
# 直接运行
uv run cc-stt-daemon

# 或安装为 systemd 服务
./scripts/install-service.sh
systemctl --user start cc-stt-daemon
```

### 方式二：MCP Server

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

## awesome WM 配置

添加浮动窗口规则：

```lua
ruled.client.append_rule {
    rule = { class = "Tk" },
    properties = { floating = true, placement = awful.placement.centered }
}
```

## 测试

```bash
uv run pytest tests/ -v
```
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: update README for voice assistant"
```

---

## Task 10: 集成测试

**Files:**
- Modify: `tests/test_integration.py`

**Step 1: 添加集成测试**

在 `tests/test_integration.py` 末尾添加：

```python
def test_daemon_components_init(tmp_path):
    """Test daemon components can be initialized"""
    from unittest.mock import patch, MagicMock

    with patch("cc_stt.daemon.WakewordDetector") as mock_wakeword, \
         patch("cc_stt.daemon.SpeechTranscriber") as mock_transcriber, \
         patch("cc_stt.daemon.Config") as mock_config:

        mock_config.load.return_value = MagicMock(
            audio=MagicMock(sample_rate=16000, channels=1, max_duration=30,
                          silence_threshold=0.01, silence_duration=2.0),
            model=MagicMock(name="paraformer-zh")
        )

        from cc_stt.daemon import Daemon
        daemon = Daemon()

        assert daemon.wakeword is not None
        assert daemon.recorder is not None
        assert daemon.sender is not None
```

**Step 2: 运行所有测试**

Run: `uv run pytest tests/ -v`
Expected: ALL PASS

**Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add daemon integration test"
```

---

Plan complete and saved to `docs/plans/2026-01-28-voice-assistant-impl.md`. Two execution options:

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

**Which approach?**
