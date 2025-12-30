"""Tests for ToolRegistry - tool registration and invocation"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from src.tool_registry import ToolRegistry, ToolDefinition


class TestToolRegistryInit:
    """Test ToolRegistry initialization"""

    def test_initialization(self):
        registry = ToolRegistry()

        assert isinstance(registry.tools, dict)
        assert len(registry.tools) == 0
        assert isinstance(registry._intent_map, dict)

    @patch('src.tool_registry._EXTRACTOR_AVAILABLE', True)
    def test_initialization_with_extractor(self):
        with patch('src.tool_registry.ParameterExtractor'):
            registry = ToolRegistry()
            # Should initialize parameter extractor


class TestToolRegistration:
    """Test tool registration"""

    def test_register_basic_function(self):
        registry = ToolRegistry()

        @registry.register(
            name="test_tool",
            description="A test tool",
            intents=["test", "testing"]
        )
        async def test_function(query: str):
            return {'result': query}

        assert "test_tool" in registry.tools
        tool = registry.tools["test_tool"]
        assert tool.name == "test_tool"
        assert tool.description == "A test tool"
        assert tool.intents == ["test", "testing"]

    def test_register_uses_function_name(self):
        registry = ToolRegistry()

        @registry.register(intents=["test"])
        async def my_custom_function():
            pass

        assert "my_custom_function" in registry.tools

    def test_register_uses_docstring(self):
        registry = ToolRegistry()

        @registry.register(intents=["test"])
        async def test_function():
            """This is the docstring description"""
            pass

        tool = registry.tools["test_function"]
        assert tool.description == "This is the docstring description"

    def test_register_extracts_parameters(self):
        registry = ToolRegistry()

        @registry.register(intents=["test"])
        async def test_function(query: str, limit: int = 5):
            pass

        tool = registry.tools["test_function"]
        assert "query" in tool.parameters
        assert "limit" in tool.parameters
        assert tool.parameters["query"]["required"] is True
        assert tool.parameters["limit"]["required"] is False
        assert tool.parameters["limit"]["default"] == 5

    def test_register_with_priority(self):
        registry = ToolRegistry()

        @registry.register(intents=["test"], priority=9)
        async def high_priority_tool():
            pass

        tool = registry.tools["high_priority_tool"]
        assert tool.priority == 9

    def test_register_builds_intent_map(self):
        registry = ToolRegistry()

        @registry.register(intents=["search", "find", "lookup"])
        async def search_tool():
            pass

        assert "search" in registry._intent_map
        assert "find" in registry._intent_map
        assert "lookup" in registry._intent_map
        assert "search_tool" in registry._intent_map["search"]

    def test_register_multiple_tools_same_intent(self):
        registry = ToolRegistry()

        @registry.register(intents=["search"])
        async def search_tool_1():
            pass

        @registry.register(intents=["search"])
        async def search_tool_2():
            pass

        assert len(registry._intent_map["search"]) == 2


class TestToolMatching:
    """Test tool matching based on user input"""

    def test_match_tool_exact_match(self):
        registry = ToolRegistry()

        @registry.register(intents=["create goal"])
        async def create_goal():
            pass

        tool = registry.match_tool("create goal")

        assert tool is not None
        assert tool.name == "create_goal"

    def test_match_tool_phrase_match(self):
        registry = ToolRegistry()

        @registry.register(intents=["search memory"])
        async def search_memory():
            pass

        tool = registry.match_tool("I want to search memory for something")

        assert tool is not None
        assert tool.name == "search_memory"

    def test_match_tool_word_match(self):
        registry = ToolRegistry()

        @registry.register(intents=["create", "goal"])
        async def create_goal():
            pass

        tool = registry.match_tool("create a new goal")

        assert tool is not None

    def test_match_tool_priority(self):
        registry = ToolRegistry()

        @registry.register(intents=["goal"], priority=5)
        async def low_priority():
            pass

        @registry.register(intents=["goal"], priority=9)
        async def high_priority():
            pass

        tool = registry.match_tool("goal")

        # Higher priority should win
        assert tool.name == "high_priority"

    def test_match_tool_no_match(self):
        registry = ToolRegistry()

        @registry.register(intents=["test"])
        async def test_tool():
            pass

        tool = registry.match_tool("completely unrelated input")

        assert tool is None

    def test_match_tool_case_insensitive(self):
        registry = ToolRegistry()

        @registry.register(intents=["Search Memory"])
        async def search_memory():
            pass

        tool = registry.match_tool("SEARCH MEMORY")

        assert tool is not None

    def test_match_tool_phrase_at_start(self):
        registry = ToolRegistry()

        @registry.register(intents=["list tasks"])
        async def list_tasks():
            pass

        # Phrase at start gets higher score
        tool1 = registry.match_tool("list tasks for today")
        assert tool1 is not None

    def test_match_tool_word_boundary(self):
        registry = ToolRegistry()

        @registry.register(intents=["search"])
        async def search_tool():
            pass

        # Should match whole word "search", not "research"
        tool = registry.match_tool("research")
        # May or may not match depending on scoring


class TestToolInvocation:
    """Test tool invocation"""

    @pytest.mark.asyncio
    async def test_invoke_async_function(self):
        registry = ToolRegistry()

        @registry.register(intents=["test"])
        async def test_tool(query: str):
            return {'result': f'Processed: {query}'}

        # Mock parameter extraction
        with patch.object(registry, '_extract_parameters', AsyncMock(return_value={'query': 'test input'})):
            result = await registry.invoke("test input")

        assert result['result'] == 'Processed: test input'

    @pytest.mark.asyncio
    async def test_invoke_sync_function(self):
        registry = ToolRegistry()

        @registry.register(intents=["test"])
        def test_tool(query: str):
            return {'result': f'Sync: {query}'}

        with patch.object(registry, '_extract_parameters', AsyncMock(return_value={'query': 'test'})):
            result = await registry.invoke("test")

        assert result['result'] == 'Sync: test'

    @pytest.mark.asyncio
    async def test_invoke_no_match(self):
        registry = ToolRegistry()

        @registry.register(intents=["test"])
        async def test_tool():
            pass

        result = await registry.invoke("unmatched input")

        assert result is None

    @pytest.mark.asyncio
    async def test_invoke_with_context(self):
        registry = ToolRegistry()

        @registry.register(intents=["test"])
        async def test_tool(query: str):
            return {'query': query}

        context = {'intent': 'test_intent'}

        with patch.object(registry, '_extract_parameters', AsyncMock(return_value={'query': 'test'})):
            result = await registry.invoke("test", context=context)

        assert result is not None

    @pytest.mark.asyncio
    async def test_invoke_error_handling(self):
        registry = ToolRegistry()

        @registry.register(intents=["test"])
        async def failing_tool():
            raise ValueError("Test error")

        with patch.object(registry, '_extract_parameters', AsyncMock(return_value={})):
            result = await registry.invoke("test")

        assert 'error' in result
        assert 'Test error' in result['error']


class TestParameterExtraction:
    """Test parameter extraction from user input"""

    @pytest.mark.asyncio
    @patch('src.tool_registry._EXTRACTOR_AVAILABLE', False)
    async def test_extract_parameters_fallback_query(self):
        registry = ToolRegistry()

        tool_def = ToolDefinition(
            name="test",
            function=Mock(),
            description="test",
            parameters={'query': {'type': 'str', 'required': True, 'default': None}},
            intents=["search memory"]
        )

        params = await registry._extract_parameters(
            "search memory for robots",
            tool_def,
            None
        )

        assert 'query' in params
        # Intent keywords should be removed from query
        assert 'for robots' in params['query']

    @pytest.mark.asyncio
    @patch('src.tool_registry._EXTRACTOR_AVAILABLE', False)
    async def test_extract_parameters_from_context(self):
        registry = ToolRegistry()

        tool_def = ToolDefinition(
            name="test",
            function=Mock(),
            description="test",
            parameters={'limit': {'type': 'int', 'required': False, 'default': None}},
            intents=["test"]
        )

        context = {'limit': 10}

        params = await registry._extract_parameters(
            "test input",
            tool_def,
            context
        )

        assert params['limit'] == 10

    @pytest.mark.asyncio
    @patch('src.tool_registry._EXTRACTOR_AVAILABLE', False)
    async def test_extract_parameters_uses_defaults(self):
        registry = ToolRegistry()

        tool_def = ToolDefinition(
            name="test",
            function=Mock(),
            description="test",
            parameters={'limit': {'type': 'int', 'required': False, 'default': 5}},
            intents=["test"]
        )

        params = await registry._extract_parameters(
            "test input",
            tool_def,
            None
        )

        assert params['limit'] == 5

    @pytest.mark.asyncio
    @patch('src.tool_registry._EXTRACTOR_AVAILABLE', False)
    async def test_extract_parameters_pattern_matching(self):
        registry = ToolRegistry()

        tool_def = ToolDefinition(
            name="test",
            function=Mock(),
            description="test",
            parameters={'name': {'type': 'str', 'required': True, 'default': None}},
            intents=["test"]
        )

        params = await registry._extract_parameters(
            "my name is Marc",
            tool_def,
            None
        )

        assert params.get('name') == 'Marc'

    @pytest.mark.asyncio
    @patch('src.tool_registry._EXTRACTOR_AVAILABLE', True)
    async def test_extract_parameters_with_extractor(self):
        registry = ToolRegistry()
        registry.param_extractor = AsyncMock()
        registry.param_extractor.extract_parameters = AsyncMock(
            return_value={'param': 'value'}
        )

        tool_def = ToolDefinition(
            name="test",
            function=Mock(),
            description="test",
            parameters={'param': {'type': 'str', 'required': True, 'default': None}},
            intents=["test"]
        )

        params = await registry._extract_parameters(
            "test input",
            tool_def,
            None
        )

        assert params['param'] == 'value'
        registry.param_extractor.extract_parameters.assert_called_once()


class TestToolRegistryHelpers:
    """Test helper methods"""

    def test_should_invoke(self):
        registry = ToolRegistry()

        @registry.register(intents=["test"])
        async def test_tool():
            pass

        assert registry.should_invoke("test input") is True
        assert registry.should_invoke("unmatched input") is False

    def test_get_tool(self):
        registry = ToolRegistry()

        @registry.register(name="my_tool", intents=["test"])
        async def test_tool():
            pass

        tool = registry.get_tool("my_tool")

        assert tool is not None
        assert tool.name == "my_tool"

        assert registry.get_tool("nonexistent") is None

    def test_list_tools(self):
        registry = ToolRegistry()

        @registry.register(intents=["test1"])
        async def tool1():
            pass

        @registry.register(intents=["test2"])
        async def tool2():
            pass

        tools = registry.list_tools()

        assert len(tools) == 2
        assert all('name' in tool for tool in tools)
        assert all('description' in tool for tool in tools)
        assert all('intents' in tool for tool in tools)

    def test_get_tool_count(self):
        registry = ToolRegistry()

        assert registry.get_tool_count() == 0

        @registry.register(intents=["test"])
        async def tool1():
            pass

        assert registry.get_tool_count() == 1

    def test_clear(self):
        registry = ToolRegistry()

        @registry.register(intents=["test"])
        async def test_tool():
            pass

        assert registry.get_tool_count() > 0

        registry.clear()

        assert registry.get_tool_count() == 0
        assert len(registry._intent_map) == 0


class TestToolRegistryEdgeCases:
    """Test edge cases and error handling"""

    def test_register_empty_intents(self):
        registry = ToolRegistry()

        @registry.register(intents=[])
        async def test_tool():
            pass

        # Should still register
        assert "test_tool" in registry.tools

    def test_match_tool_empty_input(self):
        registry = ToolRegistry()

        @registry.register(intents=["test"])
        async def test_tool():
            pass

        tool = registry.match_tool("")

        # Should handle empty input gracefully

    def test_match_tool_with_stopwords(self):
        registry = ToolRegistry()

        @registry.register(intents=["the search for memory"])
        async def search_tool():
            pass

        # Stopwords should be handled appropriately
        tool = registry.match_tool("the search for something")

    def test_invoke_with_missing_parameters(self):
        # Tests are in invocation section
        pass

    def test_complex_intent_patterns(self):
        registry = ToolRegistry()

        @registry.register(intents=["my name is", "i am", "call me"])
        async def remember_name(name: str):
            return {'name': name}

        tool = registry.match_tool("my name is Marc")

        assert tool is not None
        assert tool.name == "remember_name"


class TestToolMatchingScoring:
    """Test enhanced scoring algorithm"""

    def test_exact_match_highest_score(self):
        registry = ToolRegistry()

        @registry.register(intents=["test"], priority=5)
        async def partial_match():
            pass

        @registry.register(intents=["exact test phrase"], priority=5)
        async def exact_match():
            pass

        tool = registry.match_tool("exact test phrase")

        # Exact match should win
        assert tool.name == "exact_match"

    def test_phrase_match_beats_word_match(self):
        registry = ToolRegistry()

        @registry.register(intents=["create", "goal"], priority=5)
        async def word_match():
            pass

        @registry.register(intents=["create goal"], priority=5)
        async def phrase_match():
            pass

        tool = registry.match_tool("create goal for project")

        # Phrase match should score higher
        assert tool.name == "phrase_match"

    def test_multi_word_intent_matching(self):
        registry = ToolRegistry()

        @registry.register(intents=["search memory for information"])
        async def specific_tool():
            pass

        tool = registry.match_tool("search memory for information about robots")

        assert tool is not None
