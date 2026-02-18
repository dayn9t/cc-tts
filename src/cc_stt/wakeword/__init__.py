"""Wakeword detection module."""

from cc_stt.wakeword.base import WakewordBackend
from cc_stt.wakeword.openwakeword import OpenWakeWordBackend

__all__ = ["WakewordBackend", "OpenWakeWordBackend"]
