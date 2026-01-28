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
