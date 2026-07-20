# =============================================
# Day 51 - Storing Chat History + User Data
# Author: Abdelrhman
# Date: July 2026
# =============================================
# Extends Day 50's database with a conversation history table,
# giving Week 5's stateless agents persistent memory across
# sessions — solving a real gap: every ask_agent() call
# previously started with zero awareness of prior questions.
# =============================================

from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import uuid

engine = create_engine('sqlite:///rag_history.db', echo=False)
Base = declarative_base()


class ChatMessage(Base):
    """
    One message in a conversation. session_id groups messages
    belonging to the same conversation together — similar to
    how a real chat app (or Week 5's agents) would track a
    single user's ongoing session.
    """
    __tablename__ = 'chat_messages'

    id = Column(Integer, primary_key=True)
    session_id = Column(String, nullable=False, index=True)
    # index=True speeds up queries that filter by session_id —
    # important once this table has thousands of rows
    role = Column(String, nullable=False)  # "user" or "assistant"
    content = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<ChatMessage(role='{self.role}', content='{self.content[:30]}...')>"


def setup_database():
    Base.metadata.create_all(engine)
    print("Chat history table created (or already existed)!")


Session = sessionmaker(bind=engine)


def get_session():
    return Session()


def start_new_session():
    """Generates a unique session ID for a new conversation."""
    return str(uuid.uuid4())


def add_message(db_session, session_id, role, content):
    """Stores one message (user question or assistant answer)."""
    message = ChatMessage(session_id=session_id, role=role, content=content)
    db_session.add(message)
    db_session.commit()
    return message


def get_conversation_history(db_session, session_id):
    """
    Retrieves all messages for a given session, in
    chronological order — exactly what a real chatbot would
    need to reconstruct "what was said so far" when a user
    returns to continue a conversation.
    """
    messages = (db_session.query(ChatMessage)
                .filter_by(session_id=session_id)
                .order_by(ChatMessage.timestamp)
                .all())
    return messages


def print_conversation(messages):
    print(f"\nConversation ({len(messages)} messages):")
    for msg in messages:
        print(f"  [{msg.role}] {msg.content}")


if __name__ == "__main__":
    setup_database()

    db_session = get_session()

    # Simulate one conversation session
    session_id = start_new_session()
    print(f"New session started: {session_id}")

    add_message(db_session, session_id, "user", "What is RAG?")
    add_message(db_session, session_id, "assistant",
                "RAG stands for Retrieval Augmented Generation...")
    add_message(db_session, session_id, "user",
                "How does it prevent hallucination?")
    add_message(db_session, session_id, "assistant",
                "By restricting answers to only retrieved context.")

    # Retrieve and display the full conversation
    history = get_conversation_history(db_session, session_id)
    print_conversation(history)

    db_session.close()