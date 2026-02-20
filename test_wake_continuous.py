"""连续测试唤醒词检测 - 只检测唤醒词，不做其他操作"""

import sys
import time
import numpy as np
import sounddevice as sd

from cc_stt.config import Config
from cc_stt.wakeword import create_wakeword_backend


def log(msg: str):
    """输出到 stderr"""
    print(msg, file=sys.stderr, flush=True)


def main():
    # 加载配置
    config = Config.load()
    ww_config = config.wakeword

    log("=" * 60)
    log("连续唤醒词测试")
    log("=" * 60)
    log(f"唤醒词: {ww_config.name}")
    log(f"后端: {ww_config.backend}")
    log(f"音频增益: {ww_config.gain}")
    log(f"采样率: {config.audio.sample_rate}")
    log("=" * 60)
    log("按 Ctrl+C 停止")
    log("")

    # 创建唤醒词后端
    if ww_config.backend == "sherpa-onnx":
        import os
        model_dir = os.path.expanduser(ww_config.sherpa_model_dir)
        keywords_file = os.path.expanduser(ww_config.sherpa_keywords_file) if ww_config.sherpa_keywords_file else None

        backend = create_wakeword_backend(
            backend="sherpa-onnx",
            name=ww_config.name,
            model_dir=model_dir,
            keywords_file=keywords_file,
            num_threads=ww_config.sherpa_num_threads,
        )
    elif ww_config.backend == "openwakeword":
        backend = create_wakeword_backend(
            backend="openwakeword",
            name=ww_config.name,
            threshold=ww_config.threshold,
        )
    else:
        log(f"不支持的后端: {ww_config.backend}")
        sys.exit(1)

    log(f"后端初始化完成: {backend.wakeword}")
    log("")

    # 统计
    detection_count = 0
    total_calls = 0
    last_status_time = time.time()

    def audio_callback(indata, frames, time_info, status):
        nonlocal detection_count, total_calls

        # 应用增益并归一化
        audio = indata[:, 0].astype(np.float32)
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val * 0.9
        audio = np.clip(audio * ww_config.gain, -1.0, 1.0)

        total_calls += 1

        if backend.process_audio(audio):
            detection_count += 1
            log(f"【唤醒词检测 #{detection_count}】t={time.time()-start_time:.1f}s")
            backend.reset()

    # 开始监听
    start_time = time.time()
    log("开始监听...")
    log("")

    try:
        with sd.InputStream(
            samplerate=config.audio.sample_rate,
            channels=config.audio.channels,
            callback=audio_callback,
            blocksize=5120,
            latency='low'
        ):
            while True:
                time.sleep(1)
                # 每秒打印状态
                now = time.time()
                if now - last_status_time >= 5:  # 每5秒打印一次状态
                    elapsed = now - start_time
                    log(f"[状态] 运行: {elapsed:.0f}s, 调用: {total_calls}, 检测: {detection_count}")
                    last_status_time = now

    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        log("")
        log("=" * 60)
        log("测试停止")
        log("=" * 60)
        log(f"运行时间: {elapsed:.1f} 秒")
        log(f"总调用次数: {total_calls}")
        log(f"唤醒词检测: {detection_count} 次")
        if total_calls > 0:
            log(f"检测率: {detection_count / total_calls * 100:.4f}%")
        log("=" * 60)


if __name__ == "__main__":
    main()
