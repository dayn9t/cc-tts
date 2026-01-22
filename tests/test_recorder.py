from cc_stt.recorder import AudioRecorder

def test_recorder_init():
    recorder = AudioRecorder(sample_rate=16000, channels=1)
    assert recorder.sample_rate == 16000
    assert recorder.channels == 1

def test_recorder_list_devices():
    recorder = AudioRecorder()
    devices = recorder.get_audio_devices()
    assert isinstance(devices, list)
