#!/usr/bin/env python3
"""简化版守护进程 - 唤醒后直接显示窗口，跳过录音转写"""

import os
import sys
import numpy as np
import sounddevice as sd

if not os.environ.get('DISPLAY'):
    os.environ['DISPLAY'] = ':0'

from cc_stt.config import Config
from cc_stt.wakeword import WakewordDetector
from cc_stt.editor import EditorWindow

def log(msg: str):
    print(msg, file=sys.stderr, flush=True)

class SimpleDaemon:
    def __init__(self):
        self.config = Config.load()
        self.wakeword = WakewordDetector(
            wakeword=self.config.wakeword.name,
            threshold=self.config.wakeword.threshold
        )
        self.audio_gain = self.config.wakeword.gain
        self.running = False
        self.triggered = False

    def _on_confirm(self, text: str):
        log(f"[confirm] 确认: '{text}'")

    def _on_cancel(self):
        log("[cancel] 取消")

    def _audio_callback(self, indata, frames, time, status):
        audio = indata[:, 0].astype(np.float32) * self.audio_gain
        audio = np.clip(audio, -1.0, 1.0)
        audio_level = np.abs(audio).mean()

        if audio_level > 0.01:
            log(f"[audio] 电平: {audio_level:.4f}")

        if self.wakeword.process_audio(audio):
            log(f"[audio] 唤醒词触发！电平: {audio_level:.4f}")
            self.triggered = True

    def run(self):
        self.running = True
        log(f"[daemon] 启动，唤醒词: {self.config.wakeword.name}")

        loop_count = 0
        while self.running:
            loop_count += 1
            log(f"[daemon] 循环 #{loop_count}")
            try:
                with sd.InputStream(
                    samplerate=self.config.audio.sample_rate,
                    channels=1,
                    callback=self._audio_callback,
                    blocksize=1280
                ):
                    log("[daemon] 等待唤醒...")
                    # 使用短睡眠循环检查标志位
                    while not self.triggered:
                        sd.sleep(100)
                    log("[daemon] 触发标志位被设置，退出监听")

                # 退出 with 块后继续处理
                log("[daemon] 唤醒词捕获！显示窗口...")

                try:
                    log("[daemon] 创建 EditorWindow...")
                    editor = EditorWindow(
                        "唤醒成功！这是测试文本。",
                        self._on_confirm,
                        self._on_cancel
                    )
                    log("[daemon] 运行 editor.run()...")
                    editor.run()
                    log("[daemon] 窗口关闭")
                except Exception as e:
                    log(f"[daemon] 窗口错误: {e}")
                    import traceback
                    traceback.print_exc()

                log("[daemon] 重置状态...")
                self.wakeword.reset()
                self.triggered = False
                log("[daemon] 继续监听...")

            except KeyboardInterrupt:
                self.running = False
                log("\n[daemon] 停止")

            except Exception as e:
                log(f"[daemon] 其他异常: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                log("[daemon] 继续...")

if __name__ == "__main__":
    daemon = SimpleDaemon()
    daemon.run()
