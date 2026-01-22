# cc-stt - Claude Code Speech-to-Text

Voice input tool for Claude Code using FunASR.

## Features

- MCP Server integration with Claude Code
- Real-time microphone recording with VAD silence detection
- FunASR Paraformer-zh model for Chinese speech recognition
- Configurable hotwords for technical terms
- Push-to-talk recording mode

## Installation

```bash
uv sync
```

## Configuration

Add to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "cc-stt": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/cc-stt",
        "run",
        "cc-stt"
      ]
    }
  }
}
```

## Usage

The MCP server provides two tools:

### transcribe

Record audio and convert to text.

Parameters:
- `max_duration` (optional): Maximum recording duration in seconds (default: 30)

### configure_hotwords

Update hotwords configuration.

Parameters:
- `hotwords`: Array of hotword strings
- `mode` (optional): "replace" or "append" (default: "append")

## Configuration Files

- `~/.config/cc-stt/config.json` - Main configuration
- `~/.config/cc-stt/hotwords.txt` - Hotwords list

## Development

Run tests:

```bash
uv run pytest tests/ -v
```

## License

MIT
