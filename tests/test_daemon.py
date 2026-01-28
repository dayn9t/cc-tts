def test_daemon_imports():
    """Test that Daemon can be imported"""
    # Skip if tkinter not available (Daemon imports EditorWindow)
    try:
        import tkinter
    except ImportError:
        return  # Skip test in headless environment

    from cc_stt.daemon import Daemon
    assert Daemon is not None
