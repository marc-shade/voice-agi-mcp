"""pytest fixtures for voice-agi-mcp tests"""

import pytest
import asyncio
import tempfile
import os
from unittest.mock import Mock, AsyncMock, MagicMock
from datetime import datetime
from typing import Dict, Any


@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_audio_file():
    """Create a temporary mock audio file"""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        # Write minimal WAV header
        f.write(b'RIFF')
        f.write((36).to_bytes(4, 'little'))
        f.write(b'WAVE')
        f.write(b'fmt ')
        f.write((16).to_bytes(4, 'little'))
        f.write((1).to_bytes(2, 'little'))  # Audio format
        f.write((1).to_bytes(2, 'little'))  # Channels
        f.write((16000).to_bytes(4, 'little'))  # Sample rate
        f.write((32000).to_bytes(4, 'little'))  # Byte rate
        f.write((2).to_bytes(2, 'little'))  # Block align
        f.write((16).to_bytes(2, 'little'))  # Bits per sample
        f.write(b'data')
        f.write((0).to_bytes(4, 'little'))
        temp_path = f.name

    yield temp_path

    # Cleanup
    try:
        os.remove(temp_path)
    except:
        pass


@pytest.fixture
def mock_whisper_model():
    """Mock Whisper model for STT testing"""
    mock_model = Mock()

    # Mock segment
    mock_segment = Mock()
    mock_segment.text = "This is transcribed text"

    # Mock transcribe method
    mock_model.transcribe = Mock(return_value=[mock_segment])

    return mock_model


@pytest.fixture
def mock_tts_response():
    """Mock TTS response data"""
    return {
        'audio_file': '/tmp/test_audio.mp3',
        'success': True,
        'duration': 2.5
    }


@pytest.fixture
def sample_voice_commands():
    """Sample voice command patterns"""
    return {
        'create_goal': [
            "create a new goal to build a robot",
            "make a goal for finishing the project",
            "I want to create a goal"
        ],
        'search_memory': [
            "search memory for robot project",
            "what do you remember about yesterday",
            "find information about meetings"
        ],
        'list_tasks': [
            "list my tasks",
            "show pending tasks",
            "what's on my todo"
        ],
        'check_status': [
            "check system status",
            "how are you",
            "system health"
        ],
        'remember_name': [
            "my name is Marc",
            "call me John",
            "remember my name is Sarah"
        ],
        'recall_name': [
            "what is my name",
            "do you remember me",
            "who am I"
        ]
    }


@pytest.fixture
def mock_conversation_context():
    """Mock conversation context for testing"""
    return {
        'session_id': 'test_session_123',
        'turns': 3,
        'user_context': {'name': 'TestUser'},
        'last_intent': 'general_query'
    }


@pytest.fixture
def mock_intent_response():
    """Mock intent detection response"""
    return {
        'intent': 'create_goal',
        'confidence': 0.95,
        'parameters': {'description': 'build a robot'},
        'requires_memory': False,
        'requires_confirmation': False
    }


@pytest.fixture
def mock_ollama_response():
    """Mock Ollama API response"""
    return {
        'model': 'llama3.2',
        'response': '{"intent": "create_goal", "confidence": 0.95, "parameters": {"description": "test"}, "requires_memory": false, "requires_confirmation": false}',
        'done': True
    }


@pytest.fixture
def mock_httpx_client():
    """Mock httpx async client"""
    mock_client = AsyncMock()
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json = Mock(return_value={
        'response': '{"intent": "general_query", "confidence": 0.8}'
    })
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client.aclose = AsyncMock()
    return mock_client


@pytest.fixture
def sample_tool_registry():
    """Sample tool registry with registered tools"""
    from src.tool_registry import ToolRegistry

    registry = ToolRegistry()

    # Register a test tool
    @registry.register(
        name="test_tool",
        description="A test tool",
        intents=["test", "testing"],
        priority=5
    )
    async def test_function(query: str = "default"):
        return {'result': f'Test result: {query}'}

    return registry


@pytest.fixture
def latency_tracker_data():
    """Sample latency tracking data"""
    return {
        'stt_latencies': [150.5, 200.3, 180.7],
        'tts_latencies': [300.2, 280.5, 310.8],
        'total_latencies': [500.0, 520.3, 510.2]
    }


@pytest.fixture
def mock_asyncio_subprocess():
    """Mock asyncio subprocess for audio recording/playback"""
    mock_process = AsyncMock()
    mock_process.returncode = 0
    mock_process.communicate = AsyncMock(return_value=(b'', b''))
    mock_process.wait = AsyncMock()
    return mock_process


@pytest.fixture
def sample_conversation_messages():
    """Sample conversation messages"""
    return [
        {
            'user': 'Hello, how are you?',
            'assistant': 'I am doing well, thank you!',
            'timestamp': '2024-01-01T10:00:00',
            'metadata': {'intent': 'greeting'}
        },
        {
            'user': 'Create a goal for my project',
            'assistant': 'Goal created with ID goal_123',
            'timestamp': '2024-01-01T10:01:00',
            'metadata': {'intent': 'create_goal', 'tool': 'create_goal_from_voice'}
        }
    ]


@pytest.fixture
def mock_edge_tts_subprocess():
    """Mock subprocess for edge-tts"""
    mock_proc = Mock()
    mock_proc.returncode = 0
    mock_proc.communicate = Mock(return_value=(b'', b''))
    return mock_proc


@pytest.fixture
def mock_voice_pipeline():
    """Mock VoicePipeline instance"""
    from unittest.mock import Mock, AsyncMock

    pipeline = Mock()
    pipeline.synthesize_speech = AsyncMock(return_value='/tmp/test.mp3')
    pipeline.listen_and_transcribe = AsyncMock(return_value='test input')
    pipeline.play_beep = AsyncMock()
    pipeline.get_latency_summary = Mock(return_value={
        'avg_stt_ms': 150.0,
        'avg_tts_ms': 300.0,
        'avg_total_ms': 500.0,
        'total_requests': 10
    })
    pipeline.is_stt_available = Mock(return_value=True)
    pipeline.is_tts_available = Mock(return_value=True)

    return pipeline


@pytest.fixture
def mock_conversation_manager():
    """Mock ConversationManager instance"""
    from unittest.mock import Mock, AsyncMock

    manager = Mock()
    manager.messages = []
    manager.user_context = {}
    manager.add_turn = Mock()
    manager.get_context = Mock(return_value='')
    manager.get_conversation_summary = Mock(return_value={
        'session_id': 'test_123',
        'total_turns': 0,
        'user_context': {}
    })
    manager.get_stats = Mock(return_value={
        'total_turns': 0,
        'total_user_words': 0,
        'total_assistant_words': 0
    })
    manager.update_user_context = Mock()
    manager.get_user_context = Mock(return_value=None)
    manager.clear_context = Mock()
    manager.store_in_memory = AsyncMock()

    return manager


@pytest.fixture
def mock_intent_detector():
    """Mock IntentDetector instance"""
    from unittest.mock import Mock, AsyncMock
    from src.intent_detector import Intent

    detector = Mock()
    detector.detect = AsyncMock(return_value=Intent(
        name='general_query',
        confidence=0.8,
        parameters={'query': 'test'},
        requires_memory=False,
        requires_confirmation=False
    ))
    detector.extract_parameters = AsyncMock(return_value={'test': 'value'})
    detector.close = AsyncMock()

    return detector
