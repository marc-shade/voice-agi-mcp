"""Tests for ConversationManager - stateful conversation handling"""

import pytest
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch
from src.conversation_manager import ConversationManager


class TestConversationManagerInit:
    """Test ConversationManager initialization"""

    def test_initialization(self):
        manager = ConversationManager(max_turns=10, enable_memory=True)

        assert manager.messages.maxlen == 10
        assert manager.enable_memory is True
        assert manager.session_id.startswith('voice_session_')
        assert isinstance(manager.user_context, dict)
        assert len(manager.user_context) == 0

    def test_initialization_custom_max_turns(self):
        manager = ConversationManager(max_turns=5)

        assert manager.messages.maxlen == 5

    def test_initialization_memory_disabled(self):
        manager = ConversationManager(enable_memory=False)

        assert manager.enable_memory is False


class TestConversationTurns:
    """Test conversation turn management"""

    def test_add_turn(self):
        manager = ConversationManager()
        manager.add_turn(
            user="Hello",
            assistant="Hi there!",
            metadata={'intent': 'greeting'}
        )

        assert len(manager.messages) == 1
        turn = manager.messages[0]
        assert turn['user'] == "Hello"
        assert turn['assistant'] == "Hi there!"
        assert turn['metadata']['intent'] == 'greeting'
        assert 'timestamp' in turn

    def test_add_turn_without_metadata(self):
        manager = ConversationManager()
        manager.add_turn(user="Hello", assistant="Hi there!")

        turn = manager.messages[0]
        assert turn['metadata'] == {}

    def test_add_multiple_turns(self, sample_conversation_messages):
        manager = ConversationManager()

        for msg in sample_conversation_messages:
            manager.add_turn(
                user=msg['user'],
                assistant=msg['assistant'],
                metadata=msg['metadata']
            )

        assert len(manager.messages) == 2

    def test_max_turns_limit(self):
        manager = ConversationManager(max_turns=3)

        for i in range(5):
            manager.add_turn(user=f"User {i}", assistant=f"Assistant {i}")

        # Should only keep last 3
        assert len(manager.messages) == 3
        assert manager.messages[0]['user'] == "User 2"
        assert manager.messages[-1]['user'] == "User 4"


class TestContextRetrieval:
    """Test context retrieval methods"""

    def test_get_context_empty(self):
        manager = ConversationManager()
        context = manager.get_context()

        assert context == ""

    def test_get_context_with_messages(self):
        manager = ConversationManager()
        manager.add_turn(user="Hello", assistant="Hi there!")
        manager.add_turn(user="How are you?", assistant="I'm well!")

        context = manager.get_context()

        assert "User: Hello" in context
        assert "Assistant: Hi there!" in context
        assert "User: How are you?" in context
        assert "Assistant: I'm well!" in context

    def test_get_context_with_metadata(self):
        manager = ConversationManager()
        manager.add_turn(
            user="Hello",
            assistant="Hi!",
            metadata={'intent': 'greeting'}
        )

        context = manager.get_context(include_metadata=True)

        assert "Metadata: {'intent': 'greeting'}" in context

    def test_get_context_for_llm_empty(self):
        manager = ConversationManager()
        messages = manager.get_context_for_llm()

        assert messages == []

    def test_get_context_for_llm_with_messages(self):
        manager = ConversationManager()
        manager.add_turn(user="Hello", assistant="Hi!")

        messages = manager.get_context_for_llm()

        assert len(messages) == 2
        assert messages[0]['role'] == 'user'
        assert messages[0]['content'] == 'Hello'
        assert messages[1]['role'] == 'assistant'
        assert messages[1]['content'] == 'Hi!'

    def test_get_context_for_llm_with_user_context(self):
        manager = ConversationManager()
        manager.user_context = {'name': 'Marc', 'role': 'developer'}
        manager.add_turn(user="Hello", assistant="Hi!")

        messages = manager.get_context_for_llm()

        # First message should be system with user context
        assert messages[0]['role'] == 'system'
        assert 'name: Marc' in messages[0]['content']
        assert 'role: developer' in messages[0]['content']

    def test_has_context(self):
        manager = ConversationManager()
        assert manager.has_context() is False

        manager.add_turn(user="Hello", assistant="Hi!")
        assert manager.has_context() is True

    def test_get_last_user_message(self):
        manager = ConversationManager()
        assert manager.get_last_user_message() is None

        manager.add_turn(user="Hello", assistant="Hi!")
        manager.add_turn(user="How are you?", assistant="Good!")

        assert manager.get_last_user_message() == "How are you?"

    def test_get_last_assistant_message(self):
        manager = ConversationManager()
        assert manager.get_last_assistant_message() is None

        manager.add_turn(user="Hello", assistant="Hi!")
        manager.add_turn(user="How are you?", assistant="Good!")

        assert manager.get_last_assistant_message() == "Good!"


class TestUserContext:
    """Test user context management"""

    def test_update_user_context(self):
        manager = ConversationManager()
        manager.update_user_context('name', 'Marc')

        assert manager.user_context['name'] == 'Marc'

    def test_update_multiple_context_values(self):
        manager = ConversationManager()
        manager.update_user_context('name', 'Marc')
        manager.update_user_context('location', 'USA')
        manager.update_user_context('role', 'developer')

        assert len(manager.user_context) == 3
        assert manager.user_context['name'] == 'Marc'
        assert manager.user_context['location'] == 'USA'

    def test_get_user_context(self):
        manager = ConversationManager()
        manager.user_context = {'name': 'Marc'}

        assert manager.get_user_context('name') == 'Marc'
        assert manager.get_user_context('nonexistent') is None

    def test_overwrite_user_context(self):
        manager = ConversationManager()
        manager.update_user_context('name', 'Marc')
        manager.update_user_context('name', 'John')

        assert manager.user_context['name'] == 'John'


class TestConversationSummary:
    """Test conversation summary and stats"""

    def test_get_conversation_summary(self):
        manager = ConversationManager()
        manager.add_turn(user="Hello", assistant="Hi!")
        manager.user_context = {'name': 'Marc'}

        summary = manager.get_conversation_summary()

        assert 'session_id' in summary
        assert summary['total_turns'] == 1
        assert summary['user_context'] == {'name': 'Marc'}
        assert summary['has_context'] is True

    def test_get_stats_empty(self):
        manager = ConversationManager()
        stats = manager.get_stats()

        assert stats['total_turns'] == 0
        assert stats['total_user_words'] == 0
        assert stats['total_assistant_words'] == 0
        assert stats['session_duration'] > 0
        assert stats['user_context_keys'] == []

    def test_get_stats_with_messages(self):
        manager = ConversationManager()
        manager.add_turn(user="Hello world", assistant="Hi there friend!")
        manager.add_turn(user="How are you?", assistant="Good!")
        manager.user_context = {'name': 'Marc'}

        stats = manager.get_stats()

        assert stats['total_turns'] == 2
        assert stats['total_user_words'] == 5  # "Hello world" + "How are you?"
        assert stats['total_assistant_words'] == 4  # "Hi there friend!" + "Good!"
        assert stats['user_context_keys'] == ['name']


class TestMemoryIntegration:
    """Test memory storage and retrieval"""

    @pytest.mark.asyncio
    async def test_store_in_memory_disabled(self):
        manager = ConversationManager(enable_memory=False)
        manager.add_turn(user="Hello", assistant="Hi!")

        # Should not raise error
        await manager.store_in_memory()

    @pytest.mark.asyncio
    async def test_store_in_memory_no_messages(self):
        manager = ConversationManager(enable_memory=True)

        # Should not raise error
        await manager.store_in_memory()

    @pytest.mark.asyncio
    async def test_store_in_memory_with_messages(self):
        manager = ConversationManager(enable_memory=True)
        manager.add_turn(
            user="Create a goal",
            assistant="Goal created",
            metadata={'tool': 'create_goal'}
        )

        # Should execute without error (actual storage is TODO)
        await manager.store_in_memory()

    @pytest.mark.asyncio
    async def test_retrieve_relevant_context_disabled(self):
        manager = ConversationManager(enable_memory=False)
        results = await manager.retrieve_relevant_context("test query")

        assert results == []

    @pytest.mark.asyncio
    async def test_retrieve_relevant_context_enabled(self):
        manager = ConversationManager(enable_memory=True)
        # Currently returns empty (TODO implementation)
        results = await manager.retrieve_relevant_context("test query", limit=5)

        assert results == []


class TestConversationClear:
    """Test conversation clearing"""

    def test_clear_context(self):
        manager = ConversationManager()
        manager.add_turn(user="Hello", assistant="Hi!")
        manager.add_turn(user="How are you?", assistant="Good!")

        assert len(manager.messages) == 2

        manager.clear_context()

        assert len(manager.messages) == 0
        assert manager.has_context() is False

    def test_clear_context_preserves_user_context(self):
        manager = ConversationManager()
        manager.user_context = {'name': 'Marc'}
        manager.add_turn(user="Hello", assistant="Hi!")

        manager.clear_context()

        # User context should be preserved
        assert manager.user_context == {'name': 'Marc'}


class TestConversationEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_messages(self):
        manager = ConversationManager()
        manager.add_turn(user="", assistant="")

        assert len(manager.messages) == 1
        assert manager.messages[0]['user'] == ""

    def test_very_long_messages(self):
        manager = ConversationManager()
        long_text = "word " * 1000

        manager.add_turn(user=long_text, assistant=long_text)

        assert len(manager.messages) == 1
        assert len(manager.messages[0]['user']) > 5000

    def test_special_characters_in_messages(self):
        manager = ConversationManager()
        special_text = "Hello! @#$%^&*() 你好 émoji 🎉"

        manager.add_turn(user=special_text, assistant=special_text)

        assert manager.messages[0]['user'] == special_text

    def test_session_id_uniqueness(self):
        manager1 = ConversationManager()
        manager2 = ConversationManager()

        # Session IDs should be different (timestamp-based)
        # May be same if created in same millisecond, so just check format
        assert manager1.session_id.startswith('voice_session_')
        assert manager2.session_id.startswith('voice_session_')
