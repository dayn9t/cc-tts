# Claude Code 语音输入工具设计方案

**日期：** 2026-01-23
**项目：** cc-stt (Claude Code Speech-to-Text)
**目标：** 为 Claude Code 提供基于 FunASR 的语音输入能力

---

## 一、系统架构

### 整体架构

这是一个基于 **MCP (Model Context Protocol)** 的语音转文字工具，作为本地服务运行，供 Claude Code 调用。

**核心组件：**

1. **MCP Server** - 使用 Python MCP SDK，暴露两个工具：
   - `transcribe`: 录音并识别，返回文字
   - `configure_hotwords`: 动态更新热词配置

2. **Audio Recorder** - 使用 `sounddevice` 库实时录音：
   - 按住说话模式：调用时开始录音，调用结束时停止
   - 录音参数：16kHz 采样率，单声道，WAV 格式

3. **Speech Recognizer** - 基于 FunASR 的 Paraformer-zh 模型：
   - 非流式识别，准确率优先
   - 支持热词增强识别
   - 本地推理，无需网络

4. **Hotwords Manager** - 热词配置管理：
   - 从 `~/.config/cc-stt/hotwords.txt` 读取
   - 支持运行时动态更新
   - 每行一个热词，支持注释

**数据流：**
```
Claude Code → MCP Client → transcribe 工具
  → 开始录音 → 用户说话 → 停止录音
  → FunASR 识别（应用热词）→ 返回文字 → Claude Code
```

---

## 二、MCP 工具接口

### 工具 1: `transcribe`

**功能：** 录音并转换为文字

**参数：**
- `max_duration` (可选, 默认 30): 最大录音时长（秒），防止无限录音
- `sample_rate` (可选, 默认 16000): 采样率，FunASR 推荐 16kHz

**返回：**
- 成功：识别的文字内容（字符串）
- 失败：MCP 错误响应，包含具体错误信息

**行为：**
1. 检查麦克风设备可用性
2. 开始录音（自动检测或等待 max_duration）
3. 检测静音超过 2 秒自动停止（VAD 辅助）
4. 调用 FunASR 识别
5. 应用热词增强
6. 返回结果

### 工具 2: `configure_hotwords`

**功能：** 更新热词配置

**参数：**
- `hotwords` (必需): 热词列表，字符串数组
- `mode` (可选, 默认 "append"):
  - `"replace"`: 替换全部热词
  - `"append"`: 追加到现有热词

**返回：**
- 成功：更新后的热词数量
- 失败：错误信息

**行为：**
1. 验证热词格式（非空，长度合理）
2. 根据 mode 更新内存中的热词列表
3. 写入配置文件持久化
4. 返回确认信息

---

## 三、录音实现

### 录音模块 (`recorder.py`)

**技术选型：** `sounddevice` + `numpy`
- 跨平台支持（Linux/macOS/Windows）
- 低延迟，适合实时录音
- 与 FunASR 的 numpy 数组格式兼容

**核心类：**

```python
class AudioRecorder:
    def __init__(self, sample_rate=16000, channels=1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.frames = []

    def record(self, max_duration=30) -> np.ndarray:
        """录音直到静音或超时"""
        # 使用 sounddevice.InputStream
        # 实时检测音量，静音 2 秒自动停止
        # 返回 numpy 数组

    def get_audio_devices(self) -> list:
        """列出可用麦克风设备"""
```

**静音检测（VAD 辅助）：**
- 计算音频帧的 RMS 能量
- 低于阈值（如 0.01）视为静音
- 连续 2 秒静音自动停止录音
- 避免用户忘记停止导致超长录音

**错误处理：**
- 麦克风权限被拒：返回明确错误 "Microphone permission denied"
- 设备不可用：返回 "No audio input device found"
- 录音超时：返回 "Recording timeout exceeded {max_duration}s"

---

## 四、语音识别实现

### 识别模块 (`transcriber.py`)

**技术选型：** FunASR + Paraformer-zh 模型

**核心类：**

```python
class SpeechTranscriber:
    def __init__(self, model_name="paraformer-zh", hotwords=None):
        self.model = AutoModel(model=model_name)
        self.hotwords = hotwords or []

    def transcribe(self, audio: np.ndarray, sample_rate=16000) -> str:
        """识别音频，返回文字"""
        # 调用 FunASR 模型
        # 应用热词增强
        # 返回识别结果

    def update_hotwords(self, hotwords: list):
        """更新热词列表"""
```

**热词应用策略：**
- FunASR 支持 `hotwords` 参数传入热词列表
- 格式：`["Claude Code", "MCP", "TypeScript", ...]`
- 热词权重：使用默认权重，避免过度矫正

**模型初始化：**
- 首次运行自动下载模型到 `~/.cache/modelscope/`
- 模型大小约 200MB
- 加载时间约 2-3 秒（首次），后续复用实例

**错误处理：**
- 模型加载失败：返回 "Failed to load ASR model"
- 音频格式错误：返回 "Invalid audio format"
- 识别超时（30 秒）：返回 "Transcription timeout"
- 识别结果为空：返回空字符串（不报错）

---

## 五、热词管理

### 热词模块 (`hotwords.py`)

**配置文件位置：** `~/.config/cc-stt/hotwords.txt`

**文件格式：**
```
# Claude Code 相关
Claude Code
MCP
Model Context Protocol

# 编程语言
TypeScript
JavaScript
Python

# 常用命令
git commit
npm install
```

**核心类：**

```python
class HotwordsManager:
    def __init__(self, config_path="~/.config/cc-stt/hotwords.txt"):
        self.config_path = Path(config_path).expanduser()
        self.hotwords = []
        self.load()

    def load(self) -> list[str]:
        """从文件加载热词"""
        # 创建目录和默认文件（如不存在）
        # 跳过空行和 # 开头的注释
        # 返回热词列表

    def save(self, hotwords: list[str], mode="replace"):
        """保存热词到文件"""
        # replace: 覆盖全部
        # append: 追加到末尾

    def get_hotwords(self) -> list[str]:
        """获取当前热词列表"""
```

**默认热词（首次运行）：**
```python
DEFAULT_HOTWORDS = [
    "Claude Code", "MCP", "Model Context Protocol",
    "TypeScript", "JavaScript", "Python", "Rust",
    "git", "npm", "pnpm", "bun", "uv",
    "API", "JSON", "YAML", "SQL"
]
```

**行为：**
- 首次运行自动创建配置目录和文件
- 写入默认热词作为示例
- 用户可手动编辑文件或通过 `configure_hotwords` 工具更新

---

## 六、配置管理

### 配置模块 (`config.py`)

**配置文件位置：** `~/.config/cc-stt/config.json`

**配置项：**

```json
{
  "audio": {
    "sample_rate": 16000,
    "channels": 1,
    "max_duration": 30,
    "silence_threshold": 0.01,
    "silence_duration": 2.0
  },
  "model": {
    "name": "paraformer-zh",
    "cache_dir": "~/.cache/modelscope"
  },
  "hotwords": {
    "file": "~/.config/cc-stt/hotwords.txt"
  }
}
```

**核心类：**

```python
@dataclass
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    max_duration: int = 30
    silence_threshold: float = 0.01
    silence_duration: float = 2.0

@dataclass
class ModelConfig:
    name: str = "paraformer-zh"
    cache_dir: str = "~/.cache/modelscope"

@dataclass
class Config:
    audio: AudioConfig
    model: ModelConfig
    hotwords_file: str

    @classmethod
    def load(cls, path="~/.config/cc-stt/config.json") -> "Config":
        """加载配置，不存在则创建默认配置"""
```

**行为：**
- 首次运行创建默认配置文件
- 用户可手动编辑调整参数
- 配置加载失败时使用默认值并记录警告

---

## 七、MCP Server 实现

### Server 模块 (`server.py`)

**技术选型：** `mcp` Python SDK

**核心实现：**

```python
from mcp.server import Server
from mcp.types import Tool, TextContent

app = Server("cc-stt")

# 全局实例（启动时初始化）
recorder = AudioRecorder()
transcriber = SpeechTranscriber()
hotwords_mgr = HotwordsManager()

@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="transcribe",
            description="录音并转换为文字",
            inputSchema={
                "type": "object",
                "properties": {
                    "max_duration": {"type": "number", "default": 30}
                }
            }
        ),
        Tool(
            name="configure_hotwords",
            description="更新热词配置",
            inputSchema={
                "type": "object",
                "properties": {
                    "hotwords": {"type": "array", "items": {"type": "string"}},
                    "mode": {"type": "string", "enum": ["replace", "append"]}
                },
                "required": ["hotwords"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "transcribe":
        # 录音 → 识别 → 返回
        audio = recorder.record(arguments.get("max_duration", 30))
        text = transcriber.transcribe(audio, hotwords_mgr.get_hotwords())
        return [TextContent(type="text", text=text)]

    elif name == "configure_hotwords":
        # 更新热词
        hotwords_mgr.save(arguments["hotwords"], arguments.get("mode", "append"))
        return [TextContent(type="text", text=f"Updated {len(hotwords_mgr.get_hotwords())} hotwords")]
```

**启动方式：**
```bash
uv run cc-stt
```

**Claude Code 配置：**
```json
{
  "mcpServers": {
    "cc-stt": {
      "command": "uv",
      "args": [
        "--directory",
        "/home/jiang/cc/cc-stt",
        "run",
        "cc-stt"
      ]
    }
  }
}
```

---

## 八、项目结构与依赖

### 目录结构

```
cc-stt/
├── src/
│   └── cc_stt/
│       ├── __init__.py      # main() 入口
│       ├── server.py         # MCP server 实现
│       ├── transcriber.py    # FunASR 封装
│       ├── recorder.py       # 麦克风录音
│       ├── hotwords.py       # 热词管理
│       └── config.py         # 配置加载
├── tests/                    # 测试（可选）
├── docs/
│   └── plans/               # 设计文档
├── pyproject.toml
├── README.md
└── .gitignore
```

### 依赖项 (pyproject.toml)

```toml
[project]
dependencies = [
    "mcp>=1.0.0",              # MCP Python SDK
    "funasr>=1.0.0",           # FunASR 语音识别
    "sounddevice>=0.4.6",      # 麦克风录音
    "numpy>=1.24.0",           # 音频数组处理
    "modelscope>=1.9.0",       # 模型下载
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "ruff>=0.1.0",
]
```

### 配置文件位置

```
~/.config/cc-stt/
├── config.json              # 主配置
└── hotwords.txt             # 热词列表

~/.cache/modelscope/
└── damo/                    # FunASR 模型缓存
    └── speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/
```

---

## 九、使用流程

### 初次使用

1. **安装依赖：**
   ```bash
   cd /home/jiang/cc/cc-stt
   uv sync
   ```

2. **配置 Claude Code：**
   编辑 `~/.claude/settings.json`，添加 MCP Server 配置

3. **首次运行：**
   - 自动下载 FunASR 模型（~200MB）
   - 创建默认配置文件和热词列表

### 日常使用

1. **在 Claude Code 中调用：**
   - Claude 自动调用 `transcribe` 工具
   - 用户按住说话，松开停止
   - 识别结果返回给 Claude

2. **更新热词：**
   - 手动编辑 `~/.config/cc-stt/hotwords.txt`
   - 或通过 `configure_hotwords` 工具动态更新

---

## 十、技术决策总结

| 决策点 | 选择 | 理由 |
|--------|------|------|
| 集成方式 | MCP Server | 标准化，Claude Code 原生支持 |
| 音频输入 | 实时麦克风录音 | 用户体验流畅，一步完成 |
| 录音控制 | 按住说话（Push-to-Talk） | 精确控制，避免多余录音 |
| 热词管理 | 可配置文件 | 灵活适应不同项目 |
| MCP 接口 | transcribe + configure_hotwords | 功能完整，支持动态配置 |
| ASR 模型 | Paraformer-zh | 中文准确率高，速度快 |
| 错误处理 | 返回明确错误信息 | 便于调试和问题定位 |
| 项目结构 | 扁平结构 | 简洁清晰，适合小型项目 |

---

## 十一、后续优化方向

1. **中英混合优化：** 如果 Paraformer-zh 效果不佳，考虑切换到 SenseVoice
2. **热词权重调优：** 根据实际使用效果调整热词权重
3. **多设备支持：** 允许用户选择麦克风设备
4. **录音可视化：** 显示录音状态和音量指示
5. **历史记录：** 保存识别历史，支持回溯和纠错

---

**设计完成日期：** 2026-01-23
**设计者：** Claude Sonnet 4.5 + User
