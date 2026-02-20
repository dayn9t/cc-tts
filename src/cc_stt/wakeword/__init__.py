"""Wakeword detection module."""

from cc_stt.wakeword.base import WakewordBackend
from cc_stt.wakeword.factory import create_wakeword_backend
from cc_stt.wakeword.openwakeword import OpenWakeWordBackend
from cc_stt.wakeword.sherpa_onnx import SherpaONNXBackend
from cc_stt.wakeword.wekws import WeKWSBackend

__all__ = [
    "WakewordBackend",
    "OpenWakeWordBackend",
    "WeKWSBackend",
    "SherpaONNXBackend",
    "create_wakeword_backend",
]
