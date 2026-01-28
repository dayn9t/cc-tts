def test_editor_window_imports():
    """Test that EditorWindow can be imported (without GUI)"""
    # Skip if tkinter not available
    try:
        import tkinter
    except ImportError:
        return  # Skip test in headless environment

    from cc_stt.editor import EditorWindow
    assert EditorWindow is not None
