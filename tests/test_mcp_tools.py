"""Tests for MCP tool endpoints - FastMCP integration"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from src.server import (
    voice_chat, voice_listen, voice_speak, voice_conversation_loop,
    get_conversation_context, clear_conversation, list_voice_tools,
    get_voice_stats
)


class TestVoiceChat:
    """Test voice_chat MCP tool"""

    @pytest.mark.asyncio
    async def test_voice_chat_basic(
        self,
        mock_conversation_manager,
        mock_voice_pipeline,
        mock_intent_detector
    ):
        with patch('src.server.conversation_manager', mock_conversation_manager):
            with patch('src.server.voice_pipeline', mock_voice_pipeline):
                with patch('src.server.intent_detector', mock_intent_detector):
                    with patch('src.server.tool_registry') as mock_registry:
                        mock_registry.match_tool = Mock(return_value=None)

                        result = await voice_chat("Hello, how are you?")

        assert 'response' in result
        assert 'intent' in result
        assert result['conversation_turns'] >= 0

    @pytest.mark.asyncio
    async def test_voice_chat_with_tool_invocation(
        self,
        mock_conversation_manager,
        mock_voice_pipeline,
        mock_intent_detector
    ):
        mock_tool = Mock()
        mock_tool.name = "search_memory"

        with patch('src.server.conversation_manager', mock_conversation_manager):
            with patch('src.server.voice_pipeline', mock_voice_pipeline):
                with patch('src.server.intent_detector', mock_intent_detector):
                    with patch('src.server.tool_registry') as mock_registry:
                        mock_registry.match_tool = Mock(return_value=mock_tool)
                        mock_registry.invoke = AsyncMock(return_value={'result': 'found'})

                        result = await voice_chat("search memory for robots")

        assert 'tool_used' in result
        assert result['tool_used'] == 'search_memory'
        assert 'tool_result' in result

    @pytest.mark.asyncio
    async def test_voice_chat_with_listen_response(
        self,
        mock_conversation_manager,
        mock_voice_pipeline,
        mock_intent_detector
    ):
        mock_voice_pipeline.listen_and_transcribe = AsyncMock(return_value="next input")

        with patch('src.server.conversation_manager', mock_conversation_manager):
            with patch('src.server.voice_pipeline', mock_voice_pipeline):
                with patch('src.server.intent_detector', mock_intent_detector):
                    with patch('src.server.tool_registry') as mock_registry:
                        mock_registry.match_tool = Mock(return_value=None)

                        result = await voice_chat("Hello", listen_for_response=True)

        assert 'next_input' in result
        assert result['next_input'] == "next input"

    @pytest.mark.asyncio
    async def test_voice_chat_error_handling(
        self,
        mock_conversation_manager,
        mock_voice_pipeline,
        mock_intent_detector
    ):
        mock_intent_detector.detect = AsyncMock(side_effect=Exception("Detection error"))

        with patch('src.server.conversation_manager', mock_conversation_manager):
            with patch('src.server.voice_pipeline', mock_voice_pipeline):
                with patch('src.server.intent_detector', mock_intent_detector):
                    result = await voice_chat("test input")

        assert 'error' in result


class TestVoiceListen:
    """Test voice_listen MCP tool"""

    @pytest.mark.asyncio
    async def test_voice_listen_success(self, mock_voice_pipeline):
        mock_voice_pipeline.listen_and_transcribe = AsyncMock(
            return_value="transcribed text"
        )

        with patch('src.server.voice_pipeline', mock_voice_pipeline):
            result = await voice_listen(duration=5)

        assert result['success'] is True
        assert result['text'] == "transcribed text"
        assert result['duration'] == 5

    @pytest.mark.asyncio
    async def test_voice_listen_no_speech(self, mock_voice_pipeline):
        mock_voice_pipeline.listen_and_transcribe = AsyncMock(return_value=None)

        with patch('src.server.voice_pipeline', mock_voice_pipeline):
            result = await voice_listen(duration=5)

        assert result['success'] is False
        assert 'error' in result
        assert result['text'] is None

    @pytest.mark.asyncio
    async def test_voice_listen_error(self, mock_voice_pipeline):
        mock_voice_pipeline.listen_and_transcribe = AsyncMock(
            side_effect=Exception("Recording error")
        )

        with patch('src.server.voice_pipeline', mock_voice_pipeline):
            result = await voice_listen(duration=5)

        assert result['success'] is False
        assert 'error' in result


class TestVoiceSpeak:
    """Test voice_speak MCP tool"""

    @pytest.mark.asyncio
    async def test_voice_speak_success(self, mock_voice_pipeline):
        mock_voice_pipeline.synthesize_speech = AsyncMock(
            return_value="/tmp/audio.mp3"
        )

        with patch('src.server.voice_pipeline', mock_voice_pipeline):
            result = await voice_speak("Hello world", wait_for_completion=True)

        assert result['success'] is True
        assert result['audio_file'] == "/tmp/audio.mp3"
        assert result['text_length'] > 0

    @pytest.mark.asyncio
    async def test_voice_speak_without_wait(self, mock_voice_pipeline):
        mock_voice_pipeline.synthesize_speech = AsyncMock(
            return_value="/tmp/audio.mp3"
        )

        with patch('src.server.voice_pipeline', mock_voice_pipeline):
            result = await voice_speak("Hello", wait_for_completion=False)

        assert result['success'] is True

    @pytest.mark.asyncio
    async def test_voice_speak_failure(self, mock_voice_pipeline):
        mock_voice_pipeline.synthesize_speech = AsyncMock(return_value=None)

        with patch('src.server.voice_pipeline', mock_voice_pipeline):
            result = await voice_speak("Hello")

        assert result['success'] is False
        assert 'error' in result

    @pytest.mark.asyncio
    async def test_voice_speak_error(self, mock_voice_pipeline):
        mock_voice_pipeline.synthesize_speech = AsyncMock(
            side_effect=Exception("TTS error")
        )

        with patch('src.server.voice_pipeline', mock_voice_pipeline):
            result = await voice_speak("Hello")

        assert result['success'] is False
        assert 'error' in result


class TestVoiceConversationLoop:
    """Test voice_conversation_loop MCP tool"""

    @pytest.mark.asyncio
    async def test_conversation_loop_basic(
        self,
        mock_voice_pipeline,
        mock_conversation_manager
    ):
        # Mock greeting and exit
        mock_voice_pipeline.listen_and_transcribe = AsyncMock(
            side_effect=["hello", "goodbye"]
        )

        with patch('src.server.voice_pipeline', mock_voice_pipeline):
            with patch('src.server.conversation_manager', mock_conversation_manager):
                with patch('src.server.voice_chat', AsyncMock(return_value={})):
                    result = await voice_conversation_loop(max_turns=5)

        assert 'turns' in result
        assert 'summary' in result
        assert 'stats' in result

    @pytest.mark.asyncio
    async def test_conversation_loop_exit_commands(
        self,
        mock_voice_pipeline,
        mock_conversation_manager
    ):
        exit_commands = ['exit', 'quit', 'stop', 'goodbye', 'bye']

        for exit_cmd in exit_commands:
            mock_voice_pipeline.listen_and_transcribe = AsyncMock(
                return_value=exit_cmd
            )

            with patch('src.server.voice_pipeline', mock_voice_pipeline):
                with patch('src.server.conversation_manager', mock_conversation_manager):
                    result = await voice_conversation_loop(max_turns=10)

            # Should exit on first turn
            assert result['turns'] <= 1

    @pytest.mark.asyncio
    async def test_conversation_loop_no_speech(
        self,
        mock_voice_pipeline,
        mock_conversation_manager
    ):
        # Return None (no speech detected), then exit
        mock_voice_pipeline.listen_and_transcribe = AsyncMock(
            side_effect=[None, "exit"]
        )

        with patch('src.server.voice_pipeline', mock_voice_pipeline):
            with patch('src.server.conversation_manager', mock_conversation_manager):
                with patch('src.server.voice_chat', AsyncMock(return_value={})):
                    result = await voice_conversation_loop(max_turns=5)

        # Should handle None gracefully

    @pytest.mark.asyncio
    async def test_conversation_loop_error_handling(
        self,
        mock_voice_pipeline,
        mock_conversation_manager
    ):
        mock_voice_pipeline.synthesize_speech = AsyncMock(
            side_effect=Exception("TTS error")
        )

        with patch('src.server.voice_pipeline', mock_voice_pipeline):
            with patch('src.server.conversation_manager', mock_conversation_manager):
                result = await voice_conversation_loop(max_turns=5)

        assert 'error' in result


class TestGetConversationContext:
    """Test get_conversation_context MCP tool"""

    @pytest.mark.asyncio
    async def test_get_conversation_context(self, mock_conversation_manager):
        mock_conversation_manager.get_context = Mock(return_value="context text")
        mock_conversation_manager.get_conversation_summary = Mock(
            return_value={'session_id': 'test_123'}
        )
        mock_conversation_manager.get_stats = Mock(
            return_value={'total_turns': 5}
        )
        mock_conversation_manager.user_context = {'name': 'Marc'}

        with patch('src.server.conversation_manager', mock_conversation_manager):
            result = await get_conversation_context()

        assert 'context' in result
        assert 'summary' in result
        assert 'stats' in result
        assert 'user_context' in result
        assert result['user_context']['name'] == 'Marc'

    @pytest.mark.asyncio
    async def test_get_conversation_context_error(self, mock_conversation_manager):
        mock_conversation_manager.get_context = Mock(
            side_effect=Exception("Context error")
        )

        with patch('src.server.conversation_manager', mock_conversation_manager):
            result = await get_conversation_context()

        assert 'error' in result


class TestClearConversation:
    """Test clear_conversation MCP tool"""

    @pytest.mark.asyncio
    async def test_clear_conversation_success(self, mock_conversation_manager):
        with patch('src.server.conversation_manager', mock_conversation_manager):
            result = await clear_conversation()

        assert result['success'] is True
        assert 'message' in result
        mock_conversation_manager.clear_context.assert_called_once()

    @pytest.mark.asyncio
    async def test_clear_conversation_error(self, mock_conversation_manager):
        mock_conversation_manager.clear_context = Mock(
            side_effect=Exception("Clear error")
        )

        with patch('src.server.conversation_manager', mock_conversation_manager):
            result = await clear_conversation()

        assert result['success'] is False
        assert 'error' in result


class TestListVoiceTools:
    """Test list_voice_tools MCP tool"""

    @pytest.mark.asyncio
    async def test_list_voice_tools_success(self, sample_tool_registry):
        with patch('src.server.tool_registry', sample_tool_registry):
            result = await list_voice_tools()

        assert 'tools' in result
        assert 'count' in result
        assert result['count'] > 0
        assert isinstance(result['tools'], list)

    @pytest.mark.asyncio
    async def test_list_voice_tools_error():
        mock_registry = Mock()
        mock_registry.list_tools = Mock(side_effect=Exception("List error"))

        with patch('src.server.tool_registry', mock_registry):
            result = await list_voice_tools()

        assert 'error' in result


class TestGetVoiceStats:
    """Test get_voice_stats MCP tool"""

    @pytest.mark.asyncio
    async def test_get_voice_stats_success(
        self,
        mock_voice_pipeline,
        mock_conversation_manager
    ):
        mock_voice_pipeline.get_latency_summary = Mock(
            return_value={'avg_stt_ms': 150}
        )
        mock_voice_pipeline.is_stt_available = Mock(return_value=True)
        mock_voice_pipeline.is_tts_available = Mock(return_value=True)

        mock_conversation_manager.get_stats = Mock(
            return_value={'total_turns': 5}
        )

        mock_registry = Mock()
        mock_registry.get_tool_count = Mock(return_value=10)

        with patch('src.server.voice_pipeline', mock_voice_pipeline):
            with patch('src.server.conversation_manager', mock_conversation_manager):
                with patch('src.server.tool_registry', mock_registry):
                    result = await get_voice_stats()

        assert 'latency' in result
        assert 'stt_available' in result
        assert 'tts_available' in result
        assert 'conversation_stats' in result
        assert 'registered_tools' in result
        assert result['stt_available'] is True
        assert result['registered_tools'] == 10

    @pytest.mark.asyncio
    async def test_get_voice_stats_error(self, mock_voice_pipeline):
        mock_voice_pipeline.get_latency_summary = Mock(
            side_effect=Exception("Stats error")
        )

        with patch('src.server.voice_pipeline', mock_voice_pipeline):
            result = await get_voice_stats()

        assert 'error' in result


class TestRegisteredVoiceTools:
    """Test registered voice-callable tools"""

    @pytest.mark.asyncio
    async def test_search_agi_memory_tool(self):
        from src.server import search_agi_memory

        with patch('src.server.voice_pipeline') as mock_pipeline:
            mock_pipeline.synthesize_speech = AsyncMock()

            result = await search_agi_memory("test query")

        assert 'query' in result
        assert result['query'] == "test query"

    @pytest.mark.asyncio
    async def test_create_goal_from_voice_tool(self):
        from src.server import create_goal_from_voice

        with patch('src.server.voice_pipeline') as mock_pipeline:
            mock_pipeline.synthesize_speech = AsyncMock()

            result = await create_goal_from_voice("Build a robot")

        assert 'goal_id' in result
        assert 'description' in result
        assert result['status'] == 'active'

    @pytest.mark.asyncio
    async def test_list_pending_tasks_tool(self):
        from src.server import list_pending_tasks

        with patch('src.server.voice_pipeline') as mock_pipeline:
            mock_pipeline.synthesize_speech = AsyncMock()

            result = await list_pending_tasks(limit=5)

        assert 'tasks' in result
        assert 'count' in result

    @pytest.mark.asyncio
    async def test_check_system_status_tool(self):
        from src.server import check_system_status

        with patch('src.server.voice_pipeline') as mock_pipeline:
            mock_pipeline.synthesize_speech = AsyncMock()

            result = await check_system_status()

        assert 'system' in result
        assert 'active_agents' in result

    @pytest.mark.asyncio
    async def test_remember_name_tool(self):
        from src.server import remember_name

        with patch('src.server.conversation_manager') as mock_manager:
            mock_manager.update_user_context = Mock()

            with patch('src.server.voice_pipeline') as mock_pipeline:
                mock_pipeline.synthesize_speech = AsyncMock()

                result = await remember_name("Marc")

        assert result['name'] == "Marc"
        assert result['stored'] is True

    @pytest.mark.asyncio
    async def test_recall_name_tool_with_name(self):
        from src.server import recall_name

        with patch('src.server.conversation_manager') as mock_manager:
            mock_manager.get_user_context = Mock(return_value="Marc")

            with patch('src.server.voice_pipeline') as mock_pipeline:
                mock_pipeline.synthesize_speech = AsyncMock()

                result = await recall_name()

        assert result['name'] == "Marc"

    @pytest.mark.asyncio
    async def test_recall_name_tool_without_name(self):
        from src.server import recall_name

        with patch('src.server.conversation_manager') as mock_manager:
            mock_manager.get_user_context = Mock(return_value=None)

            with patch('src.server.voice_pipeline') as mock_pipeline:
                mock_pipeline.synthesize_speech = AsyncMock()

                result = await recall_name()

        assert result['name'] is None

    @pytest.mark.asyncio
    async def test_trigger_consolidation_tool(self):
        from src.server import trigger_consolidation

        with patch('src.server.voice_pipeline') as mock_pipeline:
            mock_pipeline.synthesize_speech = AsyncMock()

            result = await trigger_consolidation()

        assert 'status' in result
        assert result['status'] == 'completed'

    @pytest.mark.asyncio
    async def test_start_research_tool(self):
        from src.server import start_research

        with patch('src.server.voice_pipeline') as mock_pipeline:
            mock_pipeline.synthesize_speech = AsyncMock()

            result = await start_research("AI topics")

        assert 'research_id' in result
        assert result['topic'] == "AI topics"
        assert result['status'] == 'started'

    @pytest.mark.asyncio
    async def test_decompose_goal_tool(self):
        from src.server import decompose_goal

        with patch('src.server.voice_pipeline') as mock_pipeline:
            mock_pipeline.synthesize_speech = AsyncMock()

            result = await decompose_goal("Build a website")

        assert 'goal' in result
        assert 'tasks' in result
        assert isinstance(result['tasks'], list)


class TestToolErrorHandling:
    """Test error handling in registered tools"""

    @pytest.mark.asyncio
    async def test_tool_tts_error(self):
        from src.server import search_agi_memory

        with patch('src.server.voice_pipeline') as mock_pipeline:
            mock_pipeline.synthesize_speech = AsyncMock(
                side_effect=Exception("TTS failed")
            )

            result = await search_agi_memory("test")

        # Should handle TTS error gracefully
        assert 'error' in result or 'query' in result
