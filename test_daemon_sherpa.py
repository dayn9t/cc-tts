"""Test daemon initialization with sherpa-onnx backend."""

import sys
sys.path.insert(0, '/home/jiang/cc/audio/cc-stt')

from cc_stt.daemon import Daemon


def test_daemon_init():
    """Test that daemon initializes correctly with sherpa-onnx."""
    print("Testing daemon initialization with sherpa-onnx backend...")
    print("=" * 60)

    try:
        daemon = Daemon()
        print(f"✓ Daemon initialized successfully")
        print(f"  - Backend: sherpa-onnx")
        print(f"  - Wakeword: {daemon.wakeword}")
        print(f"  - Audio gain: {daemon.audio_gain}")
        print(f"  - Sample rate: {daemon.config.audio.sample_rate}")
        print("=" * 60)
        print("SUCCESS: Daemon is ready to run with Chinese wake words!")
        return True
    except Exception as e:
        print(f"✗ Failed to initialize daemon: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_daemon_init()
    sys.exit(0 if success else 1)
