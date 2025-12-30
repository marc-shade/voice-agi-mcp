"""Tests for IntentDetector - NLU and intent classification"""

import pytest
import json
from unittest.mock import Mock, AsyncMock, patch
from src.intent_detector import IntentDetector, Intent


class TestIntent:
    """Test Intent dataclass"""

    def test_intent_creation(self):
        intent = Intent(
            name='create_goal',
            confidence=0.95,
            parameters={'description': 'test'},
            requires_memory=False,
            requires_confirmation=True
        )

        assert intent.name == 'create_goal'
        assert intent.confidence == 0.95
        assert intent.parameters == {'description': 'test'}
        assert intent.requires_memory is False
        assert intent.requires_confirmation is True

    def test_intent_defaults(self):
        intent = Intent(
            name='test',
            confidence=0.8,
            parameters={}
        )

        assert intent.requires_memory is False
        assert intent.requires_confirmation is False


class TestIntentDetectorInit:
    """Test IntentDetector initialization"""

    def test_initialization_default(self):
        detector = IntentDetector()

        assert detector.model == "llama3.2"
        assert detector.ollama_url is not None
        assert detector.client is not None

    def test_initialization_custom(self):
        detector = IntentDetector(
            ollama_url="http://custom:11434",
            model="custom-model"
        )

        assert detector.ollama_url == "http://custom:11434"
        assert detector.model == "custom-model"

    @pytest.mark.asyncio
    async def test_close(self):
        detector = IntentDetector()
        detector.client = AsyncMock()

        await detector.close()

        detector.client.aclose.assert_called_once()


class TestIntentDetection:
    """Test intent detection functionality"""

    @pytest.mark.asyncio
    async def test_detect_success(self, mock_httpx_client):
        detector = IntentDetector()
        detector.client = mock_httpx_client

        # Override response
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json = Mock(return_value={
            'response': '{"intent": "create_goal", "confidence": 0.95, "parameters": {"description": "test"}, "requires_memory": false, "requires_confirmation": false}'
        })
        mock_httpx_client.post = AsyncMock(return_value=mock_response)

        intent = await detector.detect("create a goal to test")

        assert intent.name == 'create_goal'
        assert intent.confidence == 0.95
        assert 'description' in intent.parameters

    @pytest.mark.asyncio
    async def test_detect_with_context(self, mock_httpx_client):
        detector = IntentDetector()
        detector.client = mock_httpx_client

        intent = await detector.detect(
            "create a goal",
            context="Previous conversation about projects"
        )

        # Should include context in prompt
        assert intent is not None

    @pytest.mark.asyncio
    async def test_detect_with_available_tools(self, mock_httpx_client):
        detector = IntentDetector()
        detector.client = mock_httpx_client

        tools = [
            {'name': 'search_memory', 'description': 'Search memory'},
            {'name': 'create_goal', 'description': 'Create goal'}
        ]

        intent = await detector.detect(
            "search for something",
            available_tools=tools
        )

        assert intent is not None

    @pytest.mark.asyncio
    async def test_detect_api_error(self):
        detector = IntentDetector()

        mock_response = AsyncMock()
        mock_response.status_code = 500
        detector.client.post = AsyncMock(return_value=mock_response)

        intent = await detector.detect("test input")

        # Should return unknown intent on error
        assert intent.name == 'unknown'
        assert intent.confidence == 0.0

    @pytest.mark.asyncio
    async def test_detect_exception(self):
        detector = IntentDetector()
        detector.client.post = AsyncMock(side_effect=Exception("Connection error"))

        intent = await detector.detect("test input")

        assert intent.name == 'unknown'
        assert intent.confidence == 0.0


class TestPromptBuilding:
    """Test prompt building for intent detection"""

    def test_build_intent_prompt_basic(self):
        detector = IntentDetector()

        prompt = detector._build_intent_prompt(
            user_input="create a goal",
            context=None,
            available_tools=None
        )

        assert "create_goal" in prompt
        assert "search_memory" in prompt
        assert "create a goal" in prompt
        assert "JSON response:" in prompt

    def test_build_intent_prompt_with_context(self):
        detector = IntentDetector()

        prompt = detector._build_intent_prompt(
            user_input="test",
            context="Previous turns...",
            available_tools=None
        )

        assert "Previous turns..." in prompt

    def test_build_intent_prompt_with_tools(self):
        detector = IntentDetector()

        tools = [
            {'name': 'tool1', 'description': 'Tool 1 desc'},
            {'name': 'tool2', 'description': 'Tool 2 desc'}
        ]

        prompt = detector._build_intent_prompt(
            user_input="test",
            context=None,
            available_tools=tools
        )

        assert "tool1" in prompt
        assert "Tool 1 desc" in prompt


class TestOllamaIntegration:
    """Test Ollama API integration"""

    @pytest.mark.asyncio
    async def test_call_ollama_success(self):
        detector = IntentDetector()

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json = Mock(return_value={
            'response': 'test response',
            'done': True
        })
        detector.client.post = AsyncMock(return_value=mock_response)

        result = await detector._call_ollama("test prompt")

        assert result == 'test response'
        detector.client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_call_ollama_error_status(self):
        detector = IntentDetector()

        mock_response = AsyncMock()
        mock_response.status_code = 404
        detector.client.post = AsyncMock(return_value=mock_response)

        result = await detector._call_ollama("test prompt")

        assert result == ""

    @pytest.mark.asyncio
    async def test_call_ollama_exception(self):
        detector = IntentDetector()
        detector.client.post = AsyncMock(side_effect=Exception("Network error"))

        result = await detector._call_ollama("test prompt")

        assert result == ""


class TestIntentParsing:
    """Test intent response parsing"""

    def test_parse_intent_response_valid_json(self):
        detector = IntentDetector()

        response = '''
        Here is the intent:
        {
            "intent": "create_goal",
            "confidence": 0.95,
            "parameters": {"description": "test goal"},
            "requires_memory": false,
            "requires_confirmation": true
        }
        Additional text after
        '''

        intent = detector._parse_intent_response(response, "create a goal")

        assert intent.name == 'create_goal'
        assert intent.confidence == 0.95
        assert intent.parameters['description'] == 'test goal'
        assert intent.requires_confirmation is True

    def test_parse_intent_response_invalid_json(self):
        detector = IntentDetector()

        response = "This is not JSON at all"

        intent = detector._parse_intent_response(response, "create a goal")

        # Should fall back to heuristic detection
        assert intent.name == 'create_goal'
        assert intent.confidence > 0

    def test_parse_intent_response_partial_json(self):
        detector = IntentDetector()

        response = '{"intent": "test"}'  # Missing required fields

        intent = detector._parse_intent_response(response, "test input")

        assert intent.name == 'test'


class TestFallbackDetection:
    """Test fallback heuristic intent detection"""

    def test_fallback_create_goal(self):
        detector = IntentDetector()

        intent = detector._fallback_intent_detection("create a new goal for project")

        assert intent.name == 'create_goal'
        assert intent.confidence == 0.7

    def test_fallback_search_memory(self):
        detector = IntentDetector()

        test_inputs = [
            "search memory for project",
            "what do you remember about yesterday",
            "find information on robots"
        ]

        for user_input in test_inputs:
            intent = detector._fallback_intent_detection(user_input)
            assert intent.name == 'search_memory'
            assert intent.requires_memory is True

    def test_fallback_list_tasks(self):
        detector = IntentDetector()

        test_inputs = [
            "list my tasks",
            "show pending todos",
            "what are my tasks"
        ]

        for user_input in test_inputs:
            intent = detector._fallback_intent_detection(user_input)
            assert intent.name == 'list_tasks'

    def test_fallback_check_status(self):
        detector = IntentDetector()

        intent = detector._fallback_intent_detection("how is the system")

        assert intent.name == 'check_status'

    def test_fallback_consolidation(self):
        detector = IntentDetector()

        intent = detector._fallback_intent_detection("run memory consolidation")

        assert intent.name == 'trigger_consolidation'
        assert intent.confidence == 0.8

    def test_fallback_research(self):
        detector = IntentDetector()

        intent = detector._fallback_intent_detection("research AI topics")

        assert intent.name == 'start_research'
        assert 'topic' in intent.parameters

    def test_fallback_confirmation_yes(self):
        detector = IntentDetector()

        test_inputs = ['yes', 'yeah', 'yep', 'sure', 'ok', 'okay']

        for user_input in test_inputs:
            intent = detector._fallback_intent_detection(user_input)
            assert intent.name == 'confirmation'
            assert intent.parameters['confirmed'] is True

    def test_fallback_confirmation_no(self):
        detector = IntentDetector()

        test_inputs = ['no', 'nope', 'nah', 'cancel']

        for user_input in test_inputs:
            intent = detector._fallback_intent_detection(user_input)
            assert intent.name == 'confirmation'
            assert intent.parameters['confirmed'] is False

    def test_fallback_general_query(self):
        detector = IntentDetector()

        intent = detector._fallback_intent_detection("random unmatched input")

        assert intent.name == 'general_query'
        assert intent.confidence == 0.5


class TestParameterExtraction:
    """Test parameter extraction from user input"""

    @pytest.mark.asyncio
    async def test_extract_parameters_success(self):
        detector = IntentDetector()

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json = Mock(return_value={
            'response': '{"name": "Marc", "age": 30}'
        })
        detector.client.post = AsyncMock(return_value=mock_response)

        schema = {
            'name': {'type': 'string'},
            'age': {'type': 'integer'}
        }

        params = await detector.extract_parameters(
            "My name is Marc and I'm 30 years old",
            schema
        )

        assert params['name'] == 'Marc'
        assert params['age'] == 30

    @pytest.mark.asyncio
    async def test_extract_parameters_invalid_json(self):
        detector = IntentDetector()

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json = Mock(return_value={
            'response': 'not json'
        })
        detector.client.post = AsyncMock(return_value=mock_response)

        params = await detector.extract_parameters("test", {})

        assert params == {}

    @pytest.mark.asyncio
    async def test_extract_parameters_exception(self):
        detector = IntentDetector()
        detector.client.post = AsyncMock(side_effect=Exception("Error"))

        params = await detector.extract_parameters("test", {})

        assert params == {}


class TestIntentDetectorEdgeCases:
    """Test edge cases and error handling"""

    @pytest.mark.asyncio
    async def test_detect_empty_input(self, mock_httpx_client):
        detector = IntentDetector()
        detector.client = mock_httpx_client

        intent = await detector.detect("")

        assert intent is not None

    @pytest.mark.asyncio
    async def test_detect_very_long_input(self, mock_httpx_client):
        detector = IntentDetector()
        detector.client = mock_httpx_client

        long_input = "word " * 1000

        intent = await detector.detect(long_input)

        assert intent is not None

    @pytest.mark.asyncio
    async def test_detect_special_characters(self, mock_httpx_client):
        detector = IntentDetector()
        detector.client = mock_httpx_client

        special_input = "create goal @#$% with émoji 🎉"

        intent = await detector.detect(special_input)

        assert intent is not None
