# =============================================
# Day 53 - Testing the Database Layer
# Author: Abdelrhman
# Date: July 2026
# =============================================
# Tests for Days 50-51's database code using pytest fixtures
# and an in-memory test database — never touching the real
# rag_history.db file, so tests are fast, isolated, and safe
# to run repeatedly without polluting real data.
# =============================================

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from day50_database import Base, Document, Question, add_document, add_question
from day51_chat_history import ChatMessage, start_new_session, add_message, get_conversation_history


@pytest.fixture
def test_db_session():
    """
    Creates all tables from BOTH Day 50's and Day 51's models.
    Since day50_database.py and day51_chat_history.py each
    define their own separate declarative_base(), we must
    call create_all() using EACH Base to ensure every table
    actually gets created in the test database.
    """
    from day50_database import Base as Base50
    from day51_chat_history import Base as Base51

    engine = create_engine('sqlite:///:memory:')
    Base50.metadata.create_all(engine)
    Base51.metadata.create_all(engine)

    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def test_add_document(test_db_session):
    """Tests that a document can be added and has correct fields."""
    doc = add_document(test_db_session, "test.pdf", 5)
    assert doc.filename == "test.pdf"
    assert doc.chunk_count == 5
    assert doc.id is not None


def test_add_question_links_to_document(test_db_session):
    """Tests that a question is correctly linked to its document."""
    doc = add_document(test_db_session, "test.pdf", 5)
    question = add_question(test_db_session, doc.id, "What is this?", "A test document")

    assert question.document_id == doc.id
    assert question.question_text == "What is this?"


def test_document_questions_relationship(test_db_session):
    """
    Tests the relationship() link works correctly — that
    doc.questions returns exactly the questions belonging
    to THAT document, matching Day 50's proven JOIN behavior.
    """
    doc1 = add_document(test_db_session, "doc1.pdf", 3)
    doc2 = add_document(test_db_session, "doc2.pdf", 4)

    add_question(test_db_session, doc1.id, "Question about doc1", "Answer 1")
    add_question(test_db_session, doc2.id, "Question about doc2", "Answer 2")

    test_db_session.refresh(doc1)
    assert len(doc1.questions) == 1
    assert doc1.questions[0].question_text == "Question about doc1"


def test_chat_session_isolation(test_db_session):
    """
    Tests Day 51's proven finding: two different session_ids
    keep their conversations completely separate.
    """
    session_a = start_new_session()
    session_b = start_new_session()

    add_message(test_db_session, session_a, "user", "Message in session A")
    add_message(test_db_session, session_b, "user", "Message in session B")

    history_a = get_conversation_history(test_db_session, session_a)
    history_b = get_conversation_history(test_db_session, session_b)

    assert len(history_a) == 1
    assert len(history_b) == 1
    assert history_a[0].content == "Message in session A"
    assert history_b[0].content == "Message in session B"


def test_conversation_order_preserved(test_db_session):
    """Tests that messages come back in the order they were sent."""
    session_id = start_new_session()

    add_message(test_db_session, session_id, "user", "First message")
    add_message(test_db_session, session_id, "assistant", "Second message")
    add_message(test_db_session, session_id, "user", "Third message")

    history = get_conversation_history(test_db_session, session_id)

    assert len(history) == 3
    assert history[0].content == "First message"
    assert history[1].content == "Second message"
    assert history[2].content == "Third message"