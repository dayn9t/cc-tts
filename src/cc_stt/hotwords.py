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
        """Load hotwords from file, create default if not exists"""
        if not self.config_path.exists():
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_default()

        self.hotwords = []
        with open(self.config_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    self.hotwords.append(line)

        return self.hotwords

    def _write_default(self):
        """Write default hotwords to file"""
        with open(self.config_path, 'w') as f:
            f.write("# Claude Code related\n")
            for word in DEFAULT_HOTWORDS[:3]:
                f.write(f"{word}\n")
            f.write("\n# Programming languages\n")
            for word in DEFAULT_HOTWORDS[3:7]:
                f.write(f"{word}\n")
            f.write("\n# Common commands\n")
            for word in DEFAULT_HOTWORDS[7:]:
                f.write(f"{word}\n")

    def save(self, hotwords: list[str], mode: str = "replace"):
        """Save hotwords to file"""
        if mode == "append":
            self.hotwords.extend(hotwords)
        else:  # replace
            self.hotwords = hotwords

        with open(self.config_path, 'w') as f:
            for word in self.hotwords:
                f.write(f"{word}\n")

    def get_hotwords(self) -> list[str]:
        """Get current hotwords list"""
        return self.hotwords
