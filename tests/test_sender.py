from unittest.mock import patch, call
from cc_stt.sender import Sender


def test_sender_init():
    sender = Sender(terminal_class="xfce4-terminal")
    assert sender.terminal_class == "xfce4-terminal"


@patch("cc_stt.sender.subprocess.run")
@patch("cc_stt.sender.pyperclip.copy")
def test_sender_send(mock_copy, mock_run):
    sender = Sender(terminal_class="xfce4-terminal")
    sender.send("hello world")

    mock_copy.assert_called_once_with("hello world")
    assert mock_run.call_count == 3  # windowactivate, paste, enter
