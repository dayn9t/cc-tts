from mcp.server import Server
from mcp.types import Tool, TextContent, ErrorData
import mcp.server.stdio
from .recorder import AudioRecorder
from .transcriber import SpeechTranscriber
from .hotwords import HotwordsManager
from .config import Config

# Initialize components
config = Config.load()
recorder = AudioRecorder(
    sample_rate=config.audio.sample_rate,
    channels=config.audio.channels
)
transcriber = SpeechTranscriber(model_name=config.model.name)
hotwords_mgr = HotwordsManager(config.hotwords_file)

# Update transcriber with loaded hotwords
transcriber.update_hotwords(hotwords_mgr.get_hotwords())

app = Server("cc-stt")

@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="transcribe",
            description="录音并转换为文字",
            inputSchema={
                "type": "object",
                "properties": {
                    "max_duration": {
                        "type": "number",
                        "description": "最大录音时长（秒）",
                        "default": 30
                    },
                    "sample_rate": {
                        "type": "number",
                        "description": "采样率",
                        "default": 16000
                    }
                }
            }
        ),
        Tool(
            name="configure_hotwords",
            description="更新热词配置",
            inputSchema={
                "type": "object",
                "properties": {
                    "hotwords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "热词列表"
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["replace", "append"],
                        "description": "更新模式：replace 替换全部，append 追加",
                        "default": "append"
                    }
                },
                "required": ["hotwords"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict):
    try:
        if name == "transcribe":
            max_duration = arguments.get("max_duration", 30)

            # Record audio
            audio = recorder.record(
                max_duration=max_duration,
                silence_threshold=config.audio.silence_threshold,
                silence_duration=config.audio.silence_duration
            )

            # Transcribe
            text = transcriber.transcribe(audio, config.audio.sample_rate)

            return [TextContent(type="text", text=text)]

        elif name == "configure_hotwords":
            hotwords = arguments["hotwords"]
            mode = arguments.get("mode", "append")

            # Validate hotwords
            if not hotwords or not all(isinstance(w, str) and w.strip() for w in hotwords):
                raise ValueError("Hotwords must be non-empty strings")

            # Update hotwords
            hotwords_mgr.save(hotwords, mode=mode)
            transcriber.update_hotwords(hotwords_mgr.get_hotwords())

            count = len(hotwords_mgr.get_hotwords())
            return [TextContent(type="text", text=f"Updated {count} hotwords")]

        else:
            raise ValueError(f"Unknown tool: {name}")

    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]

async def main():
    """Run MCP server"""
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )
