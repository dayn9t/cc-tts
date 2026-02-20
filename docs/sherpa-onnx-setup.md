# Sherpa-ONNX 中文唤醒词配置指南

## 快速开始

### 1. 下载模型

```bash
uv run python -m cc_stt.models.sherpa_kws.download --model wenetspeech --output models/sherpa_kws/
```

### 2. 配置 daemon

编辑 `~/.config/cc-stt/config.json`：

```json
{
  "wakeword": {
    "backend": "sherpa-onnx",
    "name": "小爱同学",
    "sherpa_model_dir": "models/sherpa_kws/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01",
    "sherpa_keywords_file": "models/sherpa_kws/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/keywords.txt",
    "sherpa_num_threads": 4
  }
}
```

### 3. 运行测试

```bash
uv run python test_chinese_wakeword.py
uv run python test_daemon_sherpa.py
```

### 4. 启动语音助手

```bash
uv run cc-stt-daemon
```

## 支持的唤醒词

- 你好军哥、小爱同学、小艺小艺、小米小米
- 蛋哥蛋哥、你好问问、林美丽、你好西西
