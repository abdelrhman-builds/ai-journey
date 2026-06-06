class WeekOneReview:
    def __init__(self, student_name, week=1):
        self.student_name = student_name
        self.week = week
        self.concepts = []

    def add_concept(self, concept):
        self.concepts.append(concept)
        return f"{concept} added successfully"

    def get_summary(self):
        return {
            "student": self.student_name,
            "week": self.week,
            "total_concepts": len(self.concepts),
            "concepts": self.concepts
        }

    def display(self):
        print(f"=== Week {self.week} Review — {self.student_name} ===")
        for index, concept in enumerate(self.concepts):
            print(f"  {index + 1}. {concept}")
        print(f"Total concepts learned: {len(self.concepts)}")


# --- Create object and use it ---
tracker = WeekOneReview("Abdelrhman")
tracker.add_concept("Python OOP")
tracker.add_concept("File Handling")
tracker.add_concept("Error Handling")
tracker.add_concept("Git Branching")
tracker.add_concept("CSV Analyzer")
tracker.display()