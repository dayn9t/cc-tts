from unittest.mock import patch, MagicMock
from cc_stt.voice_edit import VoiceEditor


def test_voice_editor_init():
    editor = VoiceEditor(model="qwen2.5:3b")
    assert editor.model == "qwen2.5:3b"


@patch("cc_stt.voice_edit.ollama.chat")
def test_voice_editor_apply_edit(mock_chat):
    mock_chat.return_value = {"message": {"content": "Hello World"}}

    editor = VoiceEditor()
    result = editor.apply_edit("hello world", "把 hello 改成 Hello")

    assert result == "Hello World"
    mock_chat.assert_called_once()
