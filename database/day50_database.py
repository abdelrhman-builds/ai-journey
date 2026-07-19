# =============================================
# Day 50 - SQL Databases Fundamentals
# Author: Abdelrhman
# Date: July 2026
# =============================================
# Builds a persistent SQLite database for the RAG system,
# solving the in-memory-only limitation from Day 33 where
# vectorstore = None disappeared on every restart.
# =============================================

from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, func
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime

# =============================================
# SECTION 1: Database Connection
# =============================================
# SQLite: the entire database lives in ONE FILE (rag_history.db).
# No separate server process needed — this file will persist
# on disk even after this script stops running.

engine = create_engine('sqlite:///rag_history.db', echo=False)
# echo=False: don't print every SQL statement (set True to
# see the actual SQL SQLAlchemy generates, useful for learning)

Base = declarative_base()
# Base is the parent class all our table definitions inherit from


# =============================================
# SECTION 2: Define Tables as Python Classes
# =============================================

class Document(Base):
    """
    Represents one uploaded document — persists what Day 33's
    load_text()/load_document() tools processed, so this
    information survives across sessions instead of vanishing
    when vectorstore = None resets.
    """
    __tablename__ = 'documents'

    id = Column(Integer, primary_key=True)
    filename = Column(String, nullable=False)
    upload_date = Column(DateTime, default=datetime.utcnow)
    chunk_count = Column(Integer, default=0)

    # This creates a Python-side link: document.questions gives
    # you all Question rows related to this document
    questions = relationship("Question", back_populates="document")

    def __repr__(self):
        return f"<Document(filename='{self.filename}', chunks={self.chunk_count})>"


class Question(Base):
    """
    Represents one question asked about a document — persists
    Q&A history that currently only exists in memory during
    an active agent session (Days 29-34).
    """
    __tablename__ = 'questions'

    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey('documents.id'), nullable=False)
    question_text = Column(String, nullable=False)
    answer_text = Column(String)
    asked_at = Column(DateTime, default=datetime.utcnow)

    # The other side of the relationship — question.document
    # gives you the Document row this question was about
    document = relationship("Document", back_populates="questions")

    def __repr__(self):
        return f"<Question(text='{self.question_text[:30]}...')>"


# =============================================
# SECTION 3: Create the Actual Tables
# =============================================

def setup_database():
    """
    Reads all Base subclasses (Document, Question) and creates
    matching SQL tables in rag_history.db if they don't already
    exist. Safe to call multiple times — won't recreate/wipe
    existing tables.
    """
    Base.metadata.create_all(engine)
    print("✅ Database tables created (or already existed)!")


# =============================================
# SECTION 4: Session Setup (How We Talk to the DB)
# =============================================

Session = sessionmaker(bind=engine)


def get_session():
    """
    Creates a new database session — the object we use to
    actually add, query, update, and delete rows. Each
    session represents one "conversation" with the database.
    """
    return Session()


# =============================================
# SECTION 5: CRUD Operations
# =============================================

def add_document(session, filename, chunk_count):
    """CREATE — adds a new document record to the database."""
    new_doc = Document(filename=filename, chunk_count=chunk_count)
    session.add(new_doc)
    session.commit()  # actually writes the change to the database file
    print(f"✅ Added document: {new_doc}")
    return new_doc


def add_question(session, document_id, question_text, answer_text):
    """CREATE — adds a new question record, linked to a document."""
    new_question = Question(
        document_id=document_id,
        question_text=question_text,
        answer_text=answer_text
    )
    session.add(new_question)
    session.commit()
    print(f"✅ Added question: {new_question}")
    return new_question


def get_all_documents(session):
    """READ — retrieves every document row from the database."""
    documents = session.query(Document).all()
    print(f"\n📄 All documents ({len(documents)} total):")
    for doc in documents:
        print(f"  ID {doc.id}: {doc.filename} ({doc.chunk_count} chunks, uploaded {doc.upload_date})")
    return documents


def get_questions_for_document(session, document_id):
    """
    READ with JOIN — retrieves all questions asked about ONE
    specific document, demonstrating the relationship() link
    between the two tables.
    """
    document = session.query(Document).filter_by(id=document_id).first()
    if not document:
        print(f"No document found with id {document_id}")
        return []

    print(f"\n❓ Questions asked about '{document.filename}':")
    for q in document.questions:
        print(f"  Q: {q.question_text}")
        print(f"  A: {q.answer_text}")
        print()
    return document.questions


def update_chunk_count(session, document_id, new_count):
    """UPDATE — modifies an existing document's chunk count."""
    document = session.query(Document).filter_by(id=document_id).first()
    if document:
        old_count = document.chunk_count
        document.chunk_count = new_count
        session.commit()
        print(f"✅ Updated '{document.filename}': {old_count} → {new_count} chunks")


def delete_document(session, document_id):
    """DELETE — removes a document (and cascades... need to check related questions)."""
    document = session.query(Document).filter_by(id=document_id).first()
    if document:
        filename = document.filename
        session.delete(document)
        session.commit()
        print(f"✅ Deleted document: {filename}")


# =============================================
# SECTION 6: Run a Complete Demo
# =============================================

if __name__ == "__main__":
    setup_database()

    session = get_session()

    print("\n--- CREATE: Adding documents ---")
    doc1 = add_document(session, "contract.pdf", 12)
    doc2 = add_document(session, "report.docx", 8)

    print("\n--- CREATE: Adding questions ---")
    add_question(session, doc1.id, "What is the contract term?", "12 months")
    add_question(session, doc1.id, "Who are the parties involved?", "Company A and Company B")
    add_question(session, doc2.id, "What is the total revenue?", "500,000 EGP")

    print("\n--- READ: All documents ---")
    get_all_documents(session)

    print("\n--- READ (JOIN): Questions for contract.pdf ---")
    get_questions_for_document(session, doc1.id)

    print("\n--- UPDATE: Changing chunk count ---")
    update_chunk_count(session, doc1.id, 15)

    session.close()