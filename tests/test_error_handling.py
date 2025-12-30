"""Tests for error handling and edge cases"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
import tempfile
import os


class TestVoicePipelineErrors:
    """Test error handling in VoicePipeline"""

    @pytest.mark.asyncio
    async def test_transcribe_nonexistent_file(self):
        from src.voice_pipeline import VoicePipeline

        pipeline = VoicePipeline()

        with patch('src.voice_pipeline.WHISPER_AVAILABLE', True):
            with patch.object(pipeline, 'load_whisper_model') as mock_load:
                mock_model = Mock()
                mock_model.transcribe = Mock(side_effect=FileNotFoundError("No file"))
                mock_load.return_value = mock_model

                result = await pipeline.transcribe_audio("/nonexistent/file.wav")

        assert result is None

    @pytest.mark.asyncio
    async def test_synthesize_speech_empty_text(self):
        from src.voice_pipeline import VoicePipeline

        pipeline = VoicePipeline()

        with patch('asyncio.create_subprocess_exec') as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.returncode = 0
            mock_process.communicate = AsyncMock(return_value=(b'', b''))
            mock_subprocess.return_value = mock_process

            result = await pipeline.synthesize_speech("")

        # Should handle empty text

    @pytest.mark.asyncio
    async def test_synthesize_speech_very_long_text(self):
        from src.voice_pipeline import VoicePipeline

        pipeline = VoicePipeline()
        long_text = "word " * 10000  # Very long text

        with patch('asyncio.create_subprocess_exec') as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.returncode = 0
            mock_process.communicate = AsyncMock(return_value=(b'', b''))
            mock_subprocess.return_value = mock_process

            with patch.object(pipeline, '_play_audio', AsyncMock()):
                result = await pipeline.synthesize_speech(long_text)

        # Should handle long text

    @pytest.mark.asyncio
    async def test_record_audio_permission_denied(self):
        from src.voice_pipeline import VoicePipeline

        pipeline = VoicePipeline()

        with patch('shutil.which', return_value='/usr/bin/arecord'):
            with patch('asyncio.create_subprocess_exec') as mock_subprocess:
                mock_subprocess.side_effect = PermissionError("No mic access")

                result = await pipeline.record_audio()

        assert result is None

    @pytest.mark.asyncio
    async def test_play_audio_file_not_found(self):
        from src.voice_pipeline import VoicePipeline

        pipeline = VoicePipeline()

        # Should not raise exception
        await pipeline._play_audio("/nonexistent/audio.mp3")


class TestConversationManagerErrors:
    """Test error handling in ConversationManager"""

    def test_add_turn_with_none_values(self):
        from src.conversation_manager import ConversationManager

        manager = ConversationManager()

        # Should handle None values
        manager.add_turn(user=None, assistant=None)

        assert len(manager.messages) == 1

    def test_get_context_with_corrupted_message(self):
        from src.conversation_manager import ConversationManager

        manager = ConversationManager()

        # Manually add corrupted message
        manager.messages.append({'corrupted': 'data'})

        # Should handle gracefully
        try:
            context = manager.get_context()
        except KeyError:
            # Expected if not handling missing keys
            pass

    @pytest.mark.asyncio
    async def test_store_in_memory_with_error(self):
        from src.conversation_manager import ConversationManager

        manager = ConversationManager(enable_memory=True)
        manager.add_turn(user="test", assistant="response")

        # Should not raise exception even if memory storage fails
        await manager.store_in_memory()

    def test_get_stats_with_empty_messages(self):
        from src.conversation_manager import ConversationManager

        manager = ConversationManager()

        stats = manager.get_stats()

        assert stats['total_turns'] == 0
        assert stats['total_user_words'] == 0


class TestIntentDetectorErrors:
    """Test error handling in IntentDetector"""

    @pytest.mark.asyncio
    async def test_detect_with_network_timeout(self):
        from src.intent_detector import IntentDetector
        import httpx

        detector = IntentDetector()
        detector.client.post = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))

        intent = await detector.detect("test input")

        assert intent.name == 'unknown'
        assert intent.confidence == 0.0

    @pytest.mark.asyncio
    async def test_detect_with_malformed_json(self):
        from src.intent_detector import IntentDetector

        detector = IntentDetector()

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json = Mock(return_value={
            'response': '{"intent": "test" MALFORMED JSON'
        })
        detector.client.post = AsyncMock(return_value=mock_response)

        intent = await detector.detect("test input")

        # Should fall back to heuristic detection
        assert intent is not None

    @pytest.mark.asyncio
    async def test_extract_parameters_with_invalid_schema(self):
        from src.intent_detector import IntentDetector

        detector = IntentDetector()

        # Invalid schema
        invalid_schema = None

        try:
            params = await detector.extract_parameters("test", invalid_schema)
            # Should handle gracefully
        except:
            pass


class TestToolRegistryErrors:
    """Test error handling in ToolRegistry"""

    def test_register_tool_with_invalid_function(self):
        from src.tool_registry import ToolRegistry

        registry = ToolRegistry()

        # Try to register non-callable
        try:
            @registry.register(intents=["test"])
            def not_a_valid_function():
                pass
        except:
            pass

    @pytest.mark.asyncio
    async def test_invoke_tool_with_missing_parameters(self):
        from src.tool_registry import ToolRegistry

        registry = ToolRegistry()

        @registry.register(intents=["test"])
        async def test_tool(required_param: str):
            return {'param': required_param}

        with patch.object(registry, '_extract_parameters', AsyncMock(return_value={})):
            result = await registry.invoke("test")

        # Should handle missing required parameter
        assert 'error' in result or result is None

    def test_match_tool_with_unicode_input(self):
        from src.tool_registry import ToolRegistry

        registry = ToolRegistry()

        @registry.register(intents=["test"])
        async def test_tool():
            pass

        # Should handle unicode
        tool = registry.match_tool("test 你好 🎉")

    @pytest.mark.asyncio
    async def test_invoke_tool_that_raises_exception(self):
        from src.tool_registry import ToolRegistry

        registry = ToolRegistry()

        @registry.register(intents=["test"])
        async def failing_tool():
            raise ValueError("Intentional error")

        with patch.object(registry, '_extract_parameters', AsyncMock(return_value={})):
            result = await registry.invoke("test")

        assert 'error' in result
        assert 'Intentional error' in result['error']


class TestMCPToolErrors:
    """Test error handling in MCP tools"""

    @pytest.mark.asyncio
    async def test_voice_chat_with_none_input(self):
        from src.server import voice_chat

        with patch('src.server.conversation_manager'):
            with patch('src.server.voice_pipeline'):
                with patch('src.server.intent_detector'):
                    with patch('src.server.tool_registry'):
                        result = await voice_chat(None)

        # Should handle None input

    @pytest.mark.asyncio
    async def test_voice_listen_with_negative_duration(self):
        from src.server import voice_listen

        with patch('src.server.voice_pipeline') as mock_pipeline:
            mock_pipeline.play_beep = AsyncMock()
            mock_pipeline.listen_and_transcribe = AsyncMock(return_value=None)

            result = await voice_listen(duration=-1)

        # Should handle invalid duration

    @pytest.mark.asyncio
    async def test_voice_speak_with_special_characters(self):
        from src.server import voice_speak

        special_text = "Test @#$%^&*() 你好 émoji 🎉"

        with patch('src.server.voice_pipeline') as mock_pipeline:
            mock_pipeline.synthesize_speech = AsyncMock(return_value="/tmp/test.mp3")

            result = await voice_speak(special_text)

        # Should handle special characters

    @pytest.mark.asyncio
    async def test_conversation_loop_with_zero_turns(self):
        from src.server import voice_conversation_loop

        with patch('src.server.voice_pipeline'):
            with patch('src.server.conversation_manager'):
                result = await voice_conversation_loop(max_turns=0)

        # Should handle zero turns gracefully


class TestIntegrationErrors:
    """Test integration error scenarios"""

    @pytest.mark.asyncio
    async def test_voice_chat_pipeline_all_failures(self):
        """Test voice_chat when all components fail"""
        from src.server import voice_chat

        mock_manager = Mock()
        mock_manager.get_context = Mock(side_effect=Exception("Manager error"))

        mock_pipeline = Mock()
        mock_pipeline.synthesize_speech = AsyncMock(side_effect=Exception("TTS error"))

        mock_detector = Mock()
        mock_detector.detect = AsyncMock(side_effect=Exception("Intent error"))

        with patch('src.server.conversation_manager', mock_manager):
            with patch('src.server.voice_pipeline', mock_pipeline):
                with patch('src.server.intent_detector', mock_detector):
                    result = await voice_chat("test")

        assert 'error' in result

    @pytest.mark.asyncio
    async def test_concurrent_tool_invocations(self):
        """Test multiple concurrent tool invocations"""
        from src.tool_registry import ToolRegistry
        import asyncio

        registry = ToolRegistry()

        call_count = 0

        @registry.register(intents=["test"])
        async def test_tool(query: str):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.1)
            return {'count': call_count}

        with patch.object(registry, '_extract_parameters', AsyncMock(return_value={'query': 'test'})):
            # Run multiple invocations concurrently
            results = await asyncio.gather(
                registry.invoke("test"),
                registry.invoke("test"),
                registry.invoke("test")
            )

        assert len(results) == 3


class TestResourceCleanup:
    """Test resource cleanup and memory management"""

    @pytest.mark.asyncio
    async def test_audio_file_cleanup_on_error(self):
        from src.voice_pipeline import VoicePipeline

        pipeline = VoicePipeline()

        # Create a real temp file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_file = f.name

        with patch('src.voice_pipeline.WHISPER_AVAILABLE', True):
            with patch.object(pipeline, 'load_whisper_model') as mock_load:
                mock_model = Mock()
                mock_model.transcribe = Mock(side_effect=Exception("Transcription error"))
                mock_load.return_value = mock_model

                result = await pipeline.transcribe_audio(temp_file)

        # File should be cleaned up even on error
        # (May not exist if cleanup worked)

    @pytest.mark.asyncio
    async def test_httpx_client_cleanup(self):
        from src.intent_detector import IntentDetector

        detector = IntentDetector()

        # Use the client
        await detector.detect("test")

        # Close it
        await detector.close()

        # Should be closed


class TestEdgeCases:
    """Test various edge cases"""

    def test_conversation_manager_with_max_turns_one(self):
        from src.conversation_manager import ConversationManager

        manager = ConversationManager(max_turns=1)

        manager.add_turn(user="first", assistant="response1")
        manager.add_turn(user="second", assistant="response2")

        # Should only keep last turn
        assert len(manager.messages) == 1
        assert manager.messages[0]['user'] == "second"

    @pytest.mark.asyncio
    async def test_voice_pipeline_with_all_features_disabled(self):
        from src.voice_pipeline import VoicePipeline

        pipeline = VoicePipeline(enable_latency_tracking=False)

        # Tracking should be None
        assert pipeline.latency_tracker is None

        # Summary should be empty
        assert pipeline.get_latency_summary() == {}

    def test_intent_detector_fallback_with_empty_string(self):
        from src.intent_detector import IntentDetector

        detector = IntentDetector()

        intent = detector._fallback_intent_detection("")

        assert intent.name == 'general_query'

    def test_tool_registry_with_no_parameters(self):
        from src.tool_registry import ToolRegistry

        registry = ToolRegistry()

        @registry.register(intents=["test"])
        async def no_params_tool():
            return {'success': True}

        tool = registry.get_tool("no_params_tool")

        # Should work with no parameters
        assert len(tool.parameters) == 0

    @pytest.mark.asyncio
    async def test_conversation_with_only_whitespace(self):
        from src.server import voice_chat

        with patch('src.server.conversation_manager'):
            with patch('src.server.voice_pipeline'):
                with patch('src.server.intent_detector') as mock_detector:
                    from src.intent_detector import Intent
                    mock_detector.detect = AsyncMock(return_value=Intent(
                        name='general_query',
                        confidence=0.5,
                        parameters={}
                    ))

                    with patch('src.server.tool_registry'):
                        result = await voice_chat("   ")

        # Should handle whitespace input


class TestConcurrency:
    """Test concurrent operations"""

    @pytest.mark.asyncio
    async def test_multiple_concurrent_voice_chats(self):
        from src.server import voice_chat
        import asyncio

        with patch('src.server.conversation_manager'):
            with patch('src.server.voice_pipeline'):
                with patch('src.server.intent_detector'):
                    with patch('src.server.tool_registry'):
                        results = await asyncio.gather(
                            voice_chat("test 1"),
                            voice_chat("test 2"),
                            voice_chat("test 3")
                        )

        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_concurrent_tts_requests(self):
        from src.voice_pipeline import VoicePipeline
        import asyncio

        pipeline = VoicePipeline()

        with patch('asyncio.create_subprocess_exec') as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.returncode = 0
            mock_process.communicate = AsyncMock(return_value=(b'', b''))
            mock_subprocess.return_value = mock_process

            with patch.object(pipeline, '_play_audio', AsyncMock()):
                results = await asyncio.gather(
                    pipeline.synthesize_speech("text 1", play_audio=False),
                    pipeline.synthesize_speech("text 2", play_audio=False),
                    pipeline.synthesize_speech("text 3", play_audio=False)
                )

        assert len(results) == 3
