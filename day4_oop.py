# =============================================
# Day 4 - Object Oriented Programming
# Author: Abdelrhman
# Date: June 2026
# =============================================

# --- SECTION 1: Basic Class ---

class AIProject:
    """Represents an AI project in your portfolio"""

    def __init__(self, name, tech_stack, status="in progress"):
        self.name = name
        self.tech_stack = tech_stack
        self.status = status
        self.progress = 0

    def describe(self):
        return f"Project: {self.name} | Stack: {', '.join(self.tech_stack)} | Status: {self.status}"

    def update_progress(self, percentage):
        self.progress = percentage
        return f"{self.name} is now {self.progress}% complete"

    def complete(self):
        self.status = "completed"
        self.progress = 100
        return f"🎉 {self.name} is now complete!"


# --- Create objects ---
project1 = AIProject(
    name="RAG Chatbot",
    tech_stack=["Python", "LangChain", "Chroma", "FastAPI"]
)

project2 = AIProject(
    name="Arabic Sentiment API",
    tech_stack=["Python", "AraBERT", "FastAPI"],
    status="planned"
)

# --- Use the objects ---
print(project1.describe())
print(project2.describe())
print()
print(project1.update_progress(25))
print(project1.update_progress(75))
print(project1.complete())
print()
print(project1.describe())


# --- SECTION 2: Class with Real Logic ---

class StudyTracker:
    """Tracks your 90-day learning progress"""

    def __init__(self, student_name, total_days=90):
        self.student_name = student_name
        self.total_days = total_days
        self.completed_days = 0
        self.skills_learned = []
        self.projects_built = []

    def complete_day(self, day_number, skill_learned):
        self.completed_days += 1
        self.skills_learned.append(skill_learned)
        return f"Day {day_number} complete! Learned: {skill_learned}"

    def add_project(self, project_name):
        self.projects_built.append(project_name)
        return f"Project added: {project_name}"

    def get_progress(self):
        percentage = (self.completed_days / self.total_days) * 100
        return f"{self.student_name}: {percentage:.1f}% complete ({self.completed_days}/{self.total_days} days)"

    def get_summary(self):
        return {
            "student": self.student_name,
            "days_completed": self.completed_days,
            "skills_learned": self.skills_learned,
            "projects_built": self.projects_built,
            "progress": f"{(self.completed_days / self.total_days) * 100:.1f}%"
        }


# --- Use the tracker ---
tracker = StudyTracker("Abdelrhman")

print(tracker.complete_day(1, "Environment Setup"))
print(tracker.complete_day(2, "Python Fundamentals"))
print(tracker.complete_day(3, "File Handling + Git"))
print(tracker.complete_day(4, "OOP"))
print()
print(tracker.add_project("RAG Chatbot"))
print()
print(tracker.get_progress())
print()

summary = tracker.get_summary()
print("=== Your Journey Summary ===")
for key, value in summary.items():
    print(f"{key}: {value}")



# --- SECTION 3: Inheritance ---

# Parent class — base AI model
class BaseAIModel:
    """Base class for all AI models"""

    def __init__(self, model_name, version):
        self.model_name = model_name
        self.version = version
        self.is_loaded = False

    def load(self):
        self.is_loaded = True
        return f"{self.model_name} v{self.version} loaded successfully"

    def describe(self):
        status = "loaded" if self.is_loaded else "not loaded"
        return f"Model: {self.model_name} | Version: {self.version} | Status: {status}"


# Child class — inherits from BaseAIModel
class TextClassifier(BaseAIModel):
    """Specialized model for text classification"""

    def __init__(self, model_name, version, labels):
        super().__init__(model_name, version)  # call parent __init__
        self.labels = labels                    # add new attribute

    def classify(self, text):
        if not self.is_loaded:
            return "Error: Model not loaded yet! Call load() first."
        return f"Text: '{text}' → Label: {self.labels[0]} (confidence: 95%)"

    def describe(self):
        base = super().describe()          # get parent description
        return f"{base} | Labels: {self.labels}"


# Child class — another specialization
class TextGenerator(BaseAIModel):
    """Specialized model for text generation"""

    def __init__(self, model_name, version, max_tokens=100):
        super().__init__(model_name, version)
        self.max_tokens = max_tokens

    def generate(self, prompt):
        if not self.is_loaded:
            return "Error: Model not loaded yet! Call load() first."
        return f"Prompt: '{prompt}' → Generated text (max {self.max_tokens} tokens)"


# --- Use the classes ---
print("=== Text Classifier ===")
classifier = TextClassifier(
    model_name="AraBERT",
    version="2.0",
    labels=["positive", "negative", "neutral"]
)

print(classifier.describe())
print(classifier.classify("This product is amazing!"))
print(classifier.load())
print(classifier.classify("This product is amazing!"))
print()

print("=== Text Generator ===")
generator = TextGenerator(
    model_name="GPT-4",
    version="turbo",
    max_tokens=500
)

print(generator.describe())
print(generator.load())
print(generator.generate("Explain RAG in simple terms"))


# --- SECTION 4: Putting It All Together ---

class DocumentLoader:
    """
    Simplified version of what LangChain's document loader does.
    In Week 3 we replace this with the real LangChain version.
    """

    def __init__(self, name):
        self.name = name
        self.documents = []
        self.total_chars = 0

    def load_text(self, filename):
        """Load a text file and store its content"""
        try:
            with open(filename, "r") as file:
                content = file.read()
            self.documents.append({
                "filename": filename,
                "content": content,
                "chars": len(content)
            })
            self.total_chars += len(content)
            return f"Loaded '{filename}' — {len(content)} characters"
        except FileNotFoundError:
            return f"Error: '{filename}' not found!"

    def search(self, query):
        """Simple search — finds documents containing the query"""
        results = []
        for doc in self.documents:
            if query.lower() in doc["content"].lower():
                results.append(doc["filename"])
        if results:
            return f"Query '{query}' found in: {results}"
        return f"Query '{query}' not found in any document"

    def get_stats(self):
        """Returns statistics about loaded documents"""
        return {
            "loader_name": self.name,
            "documents_loaded": len(self.documents),
            "total_characters": self.total_chars,
            "filenames": [doc["filename"] for doc in self.documents]
        }


# --- Use the DocumentLoader ---
loader = DocumentLoader("MyRAGLoader")

print("=== Loading Documents ===")
print(loader.load_text("ai_document.txt"))
print(loader.load_text("my_notes.txt"))
print(loader.load_text("missing.txt"))

print("\n=== Searching ===")
print(loader.search("RAG"))
print(loader.search("Python"))
print(loader.search("Kubernetes"))

print("\n=== Statistics ===")
stats = loader.get_stats()
for key, value in stats.items():
    print(f"{key}: {value}")