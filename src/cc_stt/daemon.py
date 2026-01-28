import sys
import numpy as np
import sounddevice as sd
from cc_stt.config import Config
from cc_stt.wakeword import WakewordDetector
from cc_stt.recorder import AudioRecorder
from cc_stt.transcriber import SpeechTranscriber
from cc_stt.editor import EditorWindow
from cc_stt.voice_edit import VoiceEditor
from cc_stt.sender import Sender


def log(msg: str):
    """无缓冲输出到 stderr"""
    print(msg, file=sys.stderr, flush=True)


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
        log(f"语音助手已启动，唤醒词: {self.wakeword.wakeword}")

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
                log("检测到唤醒词，开始录音...")

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
                    log(f"转写结果: {text}")

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
        log("\n语音助手已停止")


if __name__ == "__main__":
    main()
