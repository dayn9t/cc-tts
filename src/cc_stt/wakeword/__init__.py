"""Wakeword detection module."""

from cc_stt.wakeword.base import WakewordBackend
from cc_stt.wakeword.openwakeword import OpenWakeWordBackend
from cc_stt.wakeword.wekws import WeKWSBackend

__all__ = ["WakewordBackend", "OpenWakeWordBackend", "WeKWSBackend"]
