from pathlib import Path

DEFAULT_HOTWORDS = [
    "Claude Code", "MCP", "Model Context Protocol",
    "TypeScript", "JavaScript", "Python", "Rust",
    "git", "npm", "pnpm", "bun", "uv",
    "API", "JSON", "YAML", "SQL"
]

class HotwordsManager:
    def __init__(self, config_path: str = "~/.config/cc-stt/hotwords.txt"):
        self.config_path = Path(config_path).expanduser()
        self.hotwords: list[str] = []
        self.load()

    def load(self) -> list[str]:
        if not self.config_path.exists():
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_default()

        self.hotwords = [
            line.strip() for line in self.config_path.read_text().splitlines()
            if line.strip() and not line.startswith('#')
        ]
        return self.hotwords

    def _write_default(self):
        self.config_path.write_text('\n'.join(DEFAULT_HOTWORDS) + '\n')

    def save(self, hotwords: list[str], mode: str = "replace"):
        if mode == "append":
            self.hotwords.extend(hotwords)
        else:
            self.hotwords = hotwords
        self.config_path.write_text('\n'.join(self.hotwords) + '\n')

    def get_hotwords(self) -> list[str]:
        return self.hotwords
