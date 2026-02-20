import os
import sys
import numpy as np
import sounddevice as sd
from cc_stt.config import Config
from cc_stt.wakeword import create_wakeword_backend
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

        # Read config
        wakeword_name = getattr(self.config.wakeword, 'name', wakeword)
        threshold = self.config.wakeword.threshold
        backend = getattr(self.config.wakeword, 'backend', 'openwakeword')

        # Use factory
        if backend == "wekws":
            model_path = self.config.wakeword.model_path or "~/.local/share/cc-stt/models/wekws/kws_zh.onnx"
            model_path = os.path.expanduser(model_path)
            self.wakeword = create_wakeword_backend(
                backend="wekws",
                name=wakeword_name,
                threshold=threshold,
                model_path=model_path,
                window_size=self.config.wakeword.window_size,
            )
        elif backend == "sherpa-onnx":
            model_dir = self.config.wakeword.sherpa_model_dir
            if not model_dir:
                model_dir = "~/.local/share/cc-stt/models/sherpa-kws/sherpa-onnx-kws-zipformer-wenetspeech-3.3M"
            model_dir = os.path.expanduser(model_dir)

            # Expand keywords_file path if provided
            keywords_file = self.config.wakeword.sherpa_keywords_file
            if keywords_file:
                keywords_file = os.path.expanduser(keywords_file)

            # Auto-download if needed
            from cc_stt.models.sherpa_kws import ensure_model_exists
            model_dir = ensure_model_exists(model_dir)

            self.wakeword = create_wakeword_backend(
                backend="sherpa-onnx",
                name=wakeword_name,
                model_dir=model_dir,
                keywords=self.config.wakeword.sherpa_keywords,
                keywords_file=keywords_file,
                num_threads=self.config.wakeword.sherpa_num_threads,
            )
        else:
            self.wakeword = create_wakeword_backend(
                backend="openwakeword",
                name=wakeword_name,
                threshold=threshold,
            )
        self.audio_gain = self.config.wakeword.gain
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
        audio = indata[:, 0].astype(np.float32) * self.audio_gain
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
                    blocksize=5120,  # 320ms chunks for better performance
                    latency='low'
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
