from mcp.server import Server
from mcp.types import Tool, TextContent
import mcp.server.stdio
from cc_stt.recorder import AudioRecorder
from cc_stt.transcriber import SpeechTranscriber
from cc_stt.hotwords import HotwordsManager
from cc_stt.config import Config

config = Config.load()
recorder = AudioRecorder(config.audio.sample_rate, config.audio.channels)
transcriber = SpeechTranscriber(config.model.name)
hotwords_mgr = HotwordsManager(config.hotwords_file)
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
                    "mode": {"type": "string", "enum": ["replace", "append"], "default": "append"}
                },
                "required": ["hotwords"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict):
    try:
        if name == "transcribe":
            audio = recorder.record(
                max_duration=arguments.get("max_duration", 30),
                silence_threshold=config.audio.silence_threshold,
                silence_duration=config.audio.silence_duration
            )
            text = transcriber.transcribe(audio, config.audio.sample_rate)
            return [TextContent(type="text", text=text)]

        elif name == "configure_hotwords":
            hotwords = arguments["hotwords"]
            if not hotwords or not all(isinstance(w, str) and w.strip() for w in hotwords):
                raise ValueError("Hotwords must be non-empty strings")
            hotwords_mgr.save(hotwords, mode=arguments.get("mode", "append"))
            transcriber.update_hotwords(hotwords_mgr.get_hotwords())
            return [TextContent(type="text", text=f"Updated {len(hotwords_mgr.get_hotwords())} hotwords")]

        else:
            raise ValueError(f"Unknown tool: {name}")
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {e}")]

async def main():
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())
