from src.rag_assistant.memory import ConversationMemory


def test_add_turn_appends_a_question_and_answer_pair():
    memory = ConversationMemory()
    memory.add_turn("What is RAG?", "Retrieval-Augmented Generation.")

    assert memory.turns == [("What is RAG?", "Retrieval-Augmented Generation.")]


def test_oldest_turn_is_dropped_once_max_turns_is_exceeded():
    memory = ConversationMemory(max_turns=2, max_tokens=100_000)

    memory.add_turn("Q1", "A1")
    memory.add_turn("Q2", "A2")
    memory.add_turn("Q3", "A3")

    assert memory.turns == [("Q2", "A2"), ("Q3", "A3")]


def test_oldest_turns_are_dropped_once_max_tokens_is_exceeded():
    # ~4 chars per token, so a 400-character turn is roughly 100 tokens.
    memory = ConversationMemory(max_turns=100, max_tokens=150)

    memory.add_turn("Q1", "A" * 400)
    memory.add_turn("Q2", "A" * 400)

    assert memory.turns == [("Q2", "A" * 400)]


def test_pop_last_turn_removes_and_returns_the_most_recent_turn():
    memory = ConversationMemory()
    memory.add_turn("Q1", "A1")
    memory.add_turn("Q2", "A2")

    popped = memory.pop_last_turn()

    assert popped == ("Q2", "A2")
    assert memory.turns == [("Q1", "A1")]


def test_pop_last_turn_returns_none_when_memory_is_empty():
    memory = ConversationMemory()

    assert memory.pop_last_turn() is None


def test_as_text_renders_turns_in_order():
    memory = ConversationMemory()
    memory.add_turn("What is RAG?", "Retrieval-Augmented Generation.")

    assert memory.as_text() == (
        "User: What is RAG?\nAssistant: Retrieval-Augmented Generation."
    )


def test_as_text_placeholder_when_memory_is_empty():
    memory = ConversationMemory()

    assert memory.as_text() == "(no previous conversation)"


def test_clear_empties_all_turns():
    memory = ConversationMemory()
    memory.add_turn("Q1", "A1")

    memory.clear()

    assert memory.turns == []
