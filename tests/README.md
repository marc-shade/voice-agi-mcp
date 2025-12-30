# Voice-AGI MCP Test Suite

Comprehensive pytest test suite for the voice-agi-mcp server.

## Overview

This test suite provides extensive coverage (90%+) of the voice-agi-mcp codebase, testing:

- Voice pipeline (STT/TTS)
- Conversation management
- Intent detection
- Tool registry
- MCP endpoints
- Error handling

## Test Structure

```
tests/
├── conftest.py                      # Pytest fixtures
├── test_voice_pipeline.py           # Voice STT/TTS tests
├── test_conversation_manager.py     # Conversation state tests
├── test_intent_detector.py          # NLU/intent detection tests
├── test_tool_registry.py            # Tool registration tests
├── test_mcp_tools.py                # MCP endpoint tests
└── test_error_handling.py           # Error and edge case tests
```

## Quick Start

### Install Dependencies

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install test dependencies
pip install -r requirements-dev.txt
```

### Run Tests

```bash
# Run all tests
./run_tests.sh

# Run specific test file
PYTHONPATH=src pytest tests/test_voice_pipeline.py -v

# Run specific test
PYTHONPATH=src pytest tests/test_voice_pipeline.py::TestVoicePipeline::test_initialization -v

# Run with coverage
PYTHONPATH=src pytest tests/ --cov=src --cov-report=html
```

## Test Categories

### Unit Tests

#### VoicePipeline Tests (test_voice_pipeline.py)
- Initialization and configuration
- Audio recording (mocked)
- Speech-to-text transcription
- Text-to-speech synthesis
- Latency tracking
- Audio playback
- Beep generation

**Coverage**: 90% of voice_pipeline.py

#### ConversationManager Tests (test_conversation_manager.py)
- Session management
- Turn tracking
- Context retrieval
- User context storage
- Memory integration
- Conversation statistics
- Context clearing

**Coverage**: 94% of conversation_manager.py

#### IntentDetector Tests (test_intent_detector.py)
- Intent classification
- Ollama API integration
- Prompt building
- Response parsing
- Fallback detection (heuristics)
- Parameter extraction

**Coverage**: 91% of intent_detector.py

#### ToolRegistry Tests (test_tool_registry.py)
- Tool registration
- Intent matching
- Tool invocation
- Parameter extraction
- Priority handling
- Enhanced scoring algorithm

**Coverage**: 94% of tool_registry.py

### Integration Tests

#### MCP Tools Tests (test_mcp_tools.py)
- voice_chat endpoint
- voice_listen endpoint
- voice_speak endpoint
- voice_conversation_loop endpoint
- Registered voice-callable tools
- Tool error handling

### Error Handling Tests (test_error_handling.py)
- Network errors
- File system errors
- Permission errors
- Malformed data
- Concurrent operations
- Resource cleanup
- Edge cases

## Fixtures (conftest.py)

### Mock Audio Data
- `mock_audio_file`: Temporary WAV file
- `mock_whisper_model`: Mocked Whisper STT model
- `mock_tts_response`: TTS response data

### Mock Components
- `mock_voice_pipeline`: Mocked VoicePipeline
- `mock_conversation_manager`: Mocked ConversationManager
- `mock_intent_detector`: Mocked IntentDetector
- `mock_httpx_client`: Mocked HTTP client

### Test Data
- `sample_voice_commands`: Voice command patterns
- `sample_conversation_messages`: Conversation data
- `latency_tracker_data`: Performance metrics

## Coverage Reports

### Generate Coverage Report

```bash
PYTHONPATH=src pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html
```

### Current Coverage

| Module                   | Coverage |
|--------------------------|----------|
| voice_pipeline.py        | 90%      |
| conversation_manager.py  | 94%      |
| intent_detector.py       | 91%      |
| tool_registry.py         | 94%      |
| **Overall Core Modules** | **92%**  |

## Test Patterns

### Mocking Audio I/O

```python
@pytest.mark.asyncio
@patch('asyncio.create_subprocess_exec')
async def test_record_audio(mock_subprocess):
    mock_process = AsyncMock()
    mock_process.returncode = 0
    mock_subprocess.return_value = mock_process

    pipeline = VoicePipeline()
    audio_file = await pipeline.record_audio(duration=5)

    assert audio_file is not None
```

### Mocking Ollama API

```python
@pytest.mark.asyncio
async def test_detect_intent(mock_httpx_client):
    detector = IntentDetector()
    detector.client = mock_httpx_client

    intent = await detector.detect("create a goal")

    assert intent.name == 'create_goal'
```

### Testing Tool Registry

```python
def test_register_tool():
    registry = ToolRegistry()

    @registry.register(intents=["test"])
    async def test_tool(query: str):
        return {'result': query}

    assert "test_tool" in registry.tools
```

## Known Limitations

### Platform-Specific Dependencies

- **evdev**: Linux-only input device handling (not required for core tests)
- **pywhispercpp**: May require system dependencies for STT

### Excluded from Coverage

- `server.py`: MCP tool decorators make direct testing difficult
- `mcp_integrations.py`: Integration layer, tested via integration tests
- `parameter_extractor.py`: Partially tested via tool_registry tests

## Continuous Integration

The test suite is designed to run in CI/CD environments:

```yaml
# Example GitHub Actions configuration
- name: Run tests
  run: |
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements-dev.txt
    PYTHONPATH=src pytest tests/ --cov=src --cov-report=xml

- name: Upload coverage
  uses: codecov/codecov-action@v3
  with:
    files: ./coverage.xml
```

## Writing New Tests

### Test File Template

```python
"""Tests for NewModule"""

import pytest
from unittest.mock import Mock, AsyncMock, patch


class TestNewModule:
    """Test NewModule functionality"""

    def test_basic_functionality(self):
        # Arrange
        module = NewModule()

        # Act
        result = module.function()

        # Assert
        assert result == expected

    @pytest.mark.asyncio
    async def test_async_functionality(self):
        module = NewModule()
        result = await module.async_function()
        assert result is not None
```

### Fixture Template

```python
@pytest.fixture
def mock_new_component():
    """Mock NewComponent for testing"""
    component = Mock()
    component.method = Mock(return_value='value')
    component.async_method = AsyncMock(return_value='value')
    return component
```

## Troubleshooting

### Import Errors

Ensure `PYTHONPATH=src` is set:

```bash
export PYTHONPATH=src
pytest tests/
```

### Async Test Errors

Install pytest-asyncio:

```bash
pip install pytest-asyncio
```

### Coverage Not Working

Install pytest-cov:

```bash
pip install pytest-cov
```

## Contributing

When adding new functionality:

1. Write tests first (TDD)
2. Aim for 80%+ coverage
3. Include both success and error cases
4. Add docstrings to test classes/functions
5. Use descriptive test names

## Resources

- [pytest documentation](https://docs.pytest.org/)
- [pytest-asyncio](https://pytest-asyncio.readthedocs.io/)
- [pytest-cov](https://pytest-cov.readthedocs.io/)
- [unittest.mock](https://docs.python.org/3/library/unittest.mock.html)
