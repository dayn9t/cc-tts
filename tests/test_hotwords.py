from cc_stt.hotwords import HotwordsManager, DEFAULT_HOTWORDS

def test_hotwords_manager_defaults(tmp_path):
    hotwords_file = tmp_path / "hotwords.txt"
    mgr = HotwordsManager(str(hotwords_file))
    assert len(mgr.get_hotwords()) > 0
    assert "Claude Code" in mgr.get_hotwords()

def test_hotwords_save_replace(tmp_path):
    hotwords_file = tmp_path / "hotwords.txt"
    mgr = HotwordsManager(str(hotwords_file))
    mgr.save(["test1", "test2"], mode="replace")
    assert mgr.get_hotwords() == ["test1", "test2"]

def test_hotwords_save_append(tmp_path):
    hotwords_file = tmp_path / "hotwords.txt"
    mgr = HotwordsManager(str(hotwords_file))
    original_count = len(mgr.get_hotwords())
    mgr.save(["new_word"], mode="append")
    assert len(mgr.get_hotwords()) == original_count + 1
