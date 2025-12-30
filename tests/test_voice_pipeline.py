"""Tests for VoicePipeline - STT/TTS functionality"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from src.voice_pipeline import VoicePipeline, LatencyTracker
import tempfile
import os


class TestLatencyTracker:
    """Test latency tracking functionality"""

    def test_track_stt(self):
        tracker = LatencyTracker()
        tracker.track_stt(150.5)
        tracker.track_stt(200.3)

        assert len(tracker.stt_latencies) == 2
        assert tracker.stt_latencies[0] == 150.5
        assert tracker.stt_latencies[1] == 200.3

    def test_track_tts(self):
        tracker = LatencyTracker()
        tracker.track_tts(300.0)

        assert len(tracker.tts_latencies) == 1
        assert tracker.tts_latencies[0] == 300.0

    def test_track_total(self):
        tracker = LatencyTracker()
        tracker.track_total(500.0)

        assert len(tracker.total_latencies) == 1
        assert tracker.total_latencies[0] == 500.0

    def test_get_summary_empty(self):
        tracker = LatencyTracker()
        summary = tracker.get_summary()

        assert summary['avg_stt_ms'] == 0
        assert summary['avg_tts_ms'] == 0
        assert summary['avg_total_ms'] == 0
        assert summary['total_requests'] == 0

    def test_get_summary_with_data(self, latency_tracker_data):
        tracker = LatencyTracker()
        tracker.stt_latencies = latency_tracker_data['stt_latencies']
        tracker.tts_latencies = latency_tracker_data['tts_latencies']
        tracker.total_latencies = latency_tracker_data['total_latencies']

        summary = tracker.get_summary()

        # Check averages
        assert summary['avg_stt_ms'] == pytest.approx(177.17, 0.1)
        assert summary['avg_tts_ms'] == pytest.approx(297.17, 0.1)
        assert summary['avg_total_ms'] == pytest.approx(510.17, 0.1)
        assert summary['total_requests'] == 3


class TestVoicePipeline:
    """Test VoicePipeline initialization and configuration"""

    def test_initialization(self):
        pipeline = VoicePipeline(
            stt_model="base",
            tts_voice="en-IE-EmilyNeural",
            enable_latency_tracking=True
        )

        assert pipeline.stt_model_name == "base"
        assert pipeline.tts_voice == "en-IE-EmilyNeural"
        assert pipeline.latency_tracker is not None
        assert pipeline.whisper_model is None

    def test_initialization_without_tracking(self):
        pipeline = VoicePipeline(enable_latency_tracking=False)

        assert pipeline.latency_tracker is None

    def test_is_stt_available(self):
        pipeline = VoicePipeline()
        # Result depends on whether pywhispercpp is installed
        result = pipeline.is_stt_available()
        assert isinstance(result, bool)

    @patch('shutil.which')
    def test_is_tts_available(self, mock_which):
        mock_which.return_value = '/usr/bin/edge-tts'
        pipeline = VoicePipeline()

        assert pipeline.is_tts_available() is True

        mock_which.return_value = None
        assert pipeline.is_tts_available() is False

    @patch('src.voice_pipeline.WHISPER_AVAILABLE', True)
    @patch('src.voice_pipeline.WhisperModel')
    def test_load_whisper_model_success(self, mock_whisper_model_class):
        mock_model = Mock()
        mock_whisper_model_class.return_value = mock_model

        pipeline = VoicePipeline(stt_model="base")
        result = pipeline.load_whisper_model()

        assert result == mock_model
        assert pipeline.whisper_model == mock_model
        mock_whisper_model_class.assert_called_once_with("base")

    @patch('src.voice_pipeline.WHISPER_AVAILABLE', False)
    def test_load_whisper_model_unavailable(self):
        pipeline = VoicePipeline()
        result = pipeline.load_whisper_model()

        assert result is None

    @patch('src.voice_pipeline.WHISPER_AVAILABLE', True)
    @patch('src.voice_pipeline.WhisperModel')
    def test_load_whisper_model_error(self, mock_whisper_model_class):
        mock_whisper_model_class.side_effect = Exception("Model load failed")

        pipeline = VoicePipeline()
        result = pipeline.load_whisper_model()

        assert result is None


class TestVoicePipelineRecording:
    """Test audio recording functionality"""

    @pytest.mark.asyncio
    @patch('asyncio.create_subprocess_exec')
    @patch('shutil.which')
    async def test_record_audio_success(self, mock_which, mock_subprocess):
        mock_which.return_value = '/usr/bin/arecord'
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b'', b''))
        mock_subprocess.return_value = mock_process

        pipeline = VoicePipeline()
        audio_file = await pipeline.record_audio(duration=5)

        assert audio_file is not None
        assert audio_file.endswith('.wav')
        mock_subprocess.assert_called_once()

    @pytest.mark.asyncio
    @patch('shutil.which')
    async def test_record_audio_no_arecord(self, mock_which):
        mock_which.return_value = None

        pipeline = VoicePipeline()
        audio_file = await pipeline.record_audio(duration=5)

        assert audio_file is None

    @pytest.mark.asyncio
    @patch('asyncio.create_subprocess_exec')
    @patch('shutil.which')
    async def test_record_audio_failure(self, mock_which, mock_subprocess):
        mock_which.return_value = '/usr/bin/arecord'
        mock_process = AsyncMock()
        mock_process.returncode = 1
        mock_process.communicate = AsyncMock(return_value=(b'', b'error'))
        mock_subprocess.return_value = mock_process

        pipeline = VoicePipeline()
        audio_file = await pipeline.record_audio(duration=5)

        assert audio_file is None


class TestVoicePipelineTranscription:
    """Test audio transcription functionality"""

    @pytest.mark.asyncio
    @patch('src.voice_pipeline.WHISPER_AVAILABLE', False)
    async def test_transcribe_audio_whisper_unavailable(self, mock_audio_file):
        pipeline = VoicePipeline()
        result = await pipeline.transcribe_audio(mock_audio_file)

        assert result is None

    @pytest.mark.asyncio
    @patch('src.voice_pipeline.WHISPER_AVAILABLE', True)
    async def test_transcribe_audio_success(self, mock_audio_file, mock_whisper_model):
        pipeline = VoicePipeline()
        pipeline.whisper_model = mock_whisper_model

        with patch.object(pipeline, 'load_whisper_model', return_value=mock_whisper_model):
            result = await pipeline.transcribe_audio(mock_audio_file)

        assert result == "This is transcribed text"

    @pytest.mark.asyncio
    @patch('src.voice_pipeline.WHISPER_AVAILABLE', True)
    async def test_transcribe_audio_empty_result(self, mock_audio_file):
        mock_model = Mock()
        mock_segment = Mock()
        mock_segment.text = ""
        mock_model.transcribe = Mock(return_value=[mock_segment])

        pipeline = VoicePipeline()
        with patch.object(pipeline, 'load_whisper_model', return_value=mock_model):
            result = await pipeline.transcribe_audio(mock_audio_file)

        assert result is None

    @pytest.mark.asyncio
    @patch('src.voice_pipeline.WHISPER_AVAILABLE', True)
    async def test_transcribe_audio_latency_tracking(self, mock_audio_file, mock_whisper_model):
        pipeline = VoicePipeline(enable_latency_tracking=True)
        pipeline.whisper_model = mock_whisper_model

        with patch.object(pipeline, 'load_whisper_model', return_value=mock_whisper_model):
            await pipeline.transcribe_audio(mock_audio_file)

        assert len(pipeline.latency_tracker.stt_latencies) == 1
        assert pipeline.latency_tracker.stt_latencies[0] > 0


class TestVoicePipelineSynthesis:
    """Test speech synthesis functionality"""

    @pytest.mark.asyncio
    @patch('asyncio.create_subprocess_exec')
    async def test_synthesize_speech_success(self, mock_subprocess):
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b'', b''))
        mock_subprocess.return_value = mock_process

        pipeline = VoicePipeline()
        with patch.object(pipeline, '_play_audio', AsyncMock()):
            audio_file = await pipeline.synthesize_speech("Hello world", play_audio=True)

        assert audio_file is not None
        assert audio_file.endswith('.mp3')
        mock_subprocess.assert_called_once()

    @pytest.mark.asyncio
    @patch('asyncio.create_subprocess_exec')
    async def test_synthesize_speech_failure(self, mock_subprocess):
        mock_process = AsyncMock()
        mock_process.returncode = 1
        mock_process.communicate = AsyncMock(return_value=(b'', b'TTS error'))
        mock_subprocess.return_value = mock_process

        pipeline = VoicePipeline()
        audio_file = await pipeline.synthesize_speech("Hello world")

        assert audio_file is None

    @pytest.mark.asyncio
    @patch('asyncio.create_subprocess_exec')
    async def test_synthesize_speech_latency_tracking(self, mock_subprocess):
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b'', b''))
        mock_subprocess.return_value = mock_process

        pipeline = VoicePipeline(enable_latency_tracking=True)
        with patch.object(pipeline, '_play_audio', AsyncMock()):
            await pipeline.synthesize_speech("Hello world")

        assert len(pipeline.latency_tracker.tts_latencies) == 1
        assert pipeline.latency_tracker.tts_latencies[0] > 0


class TestVoicePipelineHelpers:
    """Test helper methods"""

    @pytest.mark.asyncio
    @patch('asyncio.create_subprocess_exec')
    @patch('shutil.which')
    async def test_play_audio_with_player(self, mock_which, mock_subprocess):
        mock_which.return_value = '/usr/bin/mpg123'
        mock_process = AsyncMock()
        mock_process.wait = AsyncMock()
        mock_subprocess.return_value = mock_process

        pipeline = VoicePipeline()
        await pipeline._play_audio('/tmp/test.mp3')

        mock_subprocess.assert_called_once()

    @pytest.mark.asyncio
    @patch('shutil.which')
    async def test_play_audio_no_player(self, mock_which):
        mock_which.return_value = None

        pipeline = VoicePipeline()
        # Should not raise error
        await pipeline._play_audio('/tmp/test.mp3')

    @pytest.mark.asyncio
    @patch('subprocess.Popen')
    async def test_play_beep_on(self, mock_popen):
        mock_proc = Mock()
        mock_proc.communicate = Mock(return_value=(b'', b''))
        mock_popen.return_value = mock_proc

        pipeline = VoicePipeline()
        await pipeline.play_beep("on")

        mock_popen.assert_called_once()
        call_args = mock_popen.call_args
        assert 'paplay' in call_args[0][0]

    @pytest.mark.asyncio
    @patch('subprocess.Popen')
    async def test_play_beep_off(self, mock_popen):
        mock_proc = Mock()
        mock_proc.communicate = Mock(return_value=(b'', b''))
        mock_popen.return_value = mock_proc

        pipeline = VoicePipeline()
        await pipeline.play_beep("off")

        mock_popen.assert_called_once()


class TestVoicePipelineComposite:
    """Test composite operations"""

    @pytest.mark.asyncio
    async def test_listen_and_transcribe(self):
        pipeline = VoicePipeline(enable_latency_tracking=True)

        with patch.object(pipeline, 'record_audio', AsyncMock(return_value='/tmp/test.wav')):
            with patch.object(pipeline, 'transcribe_audio', AsyncMock(return_value='test transcript')):
                result = await pipeline.listen_and_transcribe(duration=5)

        assert result == 'test transcript'
        assert len(pipeline.latency_tracker.total_latencies) == 1

    @pytest.mark.asyncio
    async def test_listen_and_transcribe_record_failure(self):
        pipeline = VoicePipeline()

        with patch.object(pipeline, 'record_audio', AsyncMock(return_value=None)):
            result = await pipeline.listen_and_transcribe(duration=5)

        assert result is None

    @pytest.mark.asyncio
    async def test_speak_and_listen(self):
        pipeline = VoicePipeline()

        with patch.object(pipeline, 'synthesize_speech', AsyncMock(return_value='/tmp/test.mp3')):
            with patch.object(pipeline, 'listen_and_transcribe', AsyncMock(return_value='user response')):
                result = await pipeline.speak_and_listen("Hello", listen_duration=5)

        assert result == 'user response'

    def test_get_latency_summary(self):
        pipeline = VoicePipeline(enable_latency_tracking=True)
        pipeline.latency_tracker.track_stt(150)
        pipeline.latency_tracker.track_tts(300)
        pipeline.latency_tracker.track_total(500)

        summary = pipeline.get_latency_summary()

        assert 'avg_stt_ms' in summary
        assert 'avg_tts_ms' in summary
        assert 'avg_total_ms' in summary
        assert summary['total_requests'] == 1

    def test_get_latency_summary_disabled(self):
        pipeline = VoicePipeline(enable_latency_tracking=False)
        summary = pipeline.get_latency_summary()

        assert summary == {}
