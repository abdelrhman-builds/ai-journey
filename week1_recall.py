# =============================================
# Week 1 Active Recall — Day 7
# Author: Abdelrhman
# Date: June 2026
# =============================================


# --- PART 1: File Loader Function ---

def load_file(filename):
    """
    Reads a file safely and returns structured data.

    Args:
        filename: Path to the file to read

    Returns:
        Dictionary with lines, total, first, last
        or None if file not found
    """
    try:
        with open(filename, "r") as file:
            lines = file.readlines()
        cleaned = [line.strip() for line in lines]
        return {
            "lines": cleaned,
            "total": len(cleaned),
            "first": cleaned[0],
            "last":  cleaned[-1]
        }
    except FileNotFoundError:
        return None


# --- PART 2: WeekOneReview Class ---

class WeekOneReview:
    """Tracks concepts learned during Week 1"""

    def __init__(self, student_name, week=1):
        self.student_name = student_name
        self.week = week
        self.concepts = []

    def add_concept(self, concept):
        """Adds a concept to the learned list"""
        self.concepts.append(concept)
        return f"{concept} added successfully"

    def get_summary(self):
        """Returns a summary dictionary"""
        return {
            "student": self.student_name,
            "week": self.week,
            "total_concepts": len(self.concepts),
            "concepts": self.concepts
        }

    def display(self):
        """Prints everything nicely"""
        print(f"=== Week {self.week} Review — {self.student_name} ===")
        for index, concept in enumerate(self.concepts):
            print(f"  {index + 1}. {concept}")
        print(f"Total concepts learned: {len(self.concepts)}")


# --- PART 3: Use Both Together ---

# Test the file loader
print("=== Testing File Loader ===")
result = load_file("my_notes.txt")
if result:
    print(f"Total lines: {result['total']}")
    print(f"First line: {result['first']}")
    print(f"Last line: {result['last']}")
else:
    print("File not found!")

# Test with missing file
result2 = load_file("missing.txt")
if result2 is None:
    print("Missing file handled gracefully!")

# Use the class
print()
tracker = WeekOneReview("Abdelrhman")
print(tracker.add_concept("Python OOP"))
print(tracker.add_concept("File Handling"))
print(tracker.add_concept("Error Handling"))
print(tracker.add_concept("Git Branching"))
print(tracker.add_concept("CSV Analyzer"))
print()
tracker.display()
print()

# Show summary dictionary
summary = tracker.get_summary()
print("=== Summary Dictionary ===")
for key, value in summary.items():
    print(f"{key}: {value}")