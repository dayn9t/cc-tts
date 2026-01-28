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
