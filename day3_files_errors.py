# =============================================
# Day 3 - File Handling & Error Handling
# Author: Abdelrhman
# Date: June 2026
# =============================================

# --- SECTION 1: Writing Files ---

# Write a file
with open("my_notes.txt", "w") as file:
    file.write("Day 1: Environment setup\n")
    file.write("Day 2: Python fundamentals\n")
    file.write("Day 3: File handling\n")

print("File written successfully!")

# --- SECTION 2: Reading Files ---

# Read entire file at once
with open("my_notes.txt", "r") as file:
    content = file.read()

print("=== File Contents ===")
print(content)

# Read line by line.
with open("my_notes.txt", "r") as file:
    lines = file.readlines()

print("=== Lines as List ===")
print(lines)
print("Number of lines:", len(lines))
print("First line:", lines[0])

# --- SECTION 3: Appending to Files ---

# Add a new day without destroying existing content
with open("my_notes.txt", "a") as file:
    file.write("Day 4: OOP concepts\n")
    file.write("Day 5: Python project\n")

print("Lines appended successfully!")

# Read and clean the lines
with open("my_notes.txt", "r") as file:
    lines = file.readlines()

print("\n=== Cleaned Lines ===")
for index, line in enumerate(lines):
    clean_line = line.strip()        # remove \n
    print(f"{index + 1}. {clean_line}")

print(f"\nTotal days logged: {len(lines)}")


# --- SECTION 4: Error Handling ---

# Example 1 — Handle missing file
print("=== Example 1: Missing File ===")
try:
    with open("missing_file.txt", "r") as file:
        content = file.read()
    print(content)
except FileNotFoundError:
    print("Error: File not found! Please check the filename.")

# Example 2 — Handle division by zero
print("\n=== Example 2: Division by Zero ===")
def safe_divide(a, b):
    try:
        result = a / b
        return f"{a} / {b} = {result}"
    except ZeroDivisionError:
            return "Error: Cannot divide by zero!"

print(safe_divide(10, 2))
print(safe_divide(10, 0))

# Example 3 — Handle wrong input type
print("\n=== Example 3: Wrong Input ===")
def safe_convert(value):
    try:
        number = int(value)
        return f"Converted successfully: {number}"
    except ValueError:
        return f"Error: '{value}' cannot be converted to a number!"
    
print(safe_convert("42"))
print(safe_convert("hello"))
print(safe_convert("3.14"))

# Example 4 — Multiple except blocks
print("\n=== Example 4: Multiple Errors ===")
def read_file_safely(filename):
    try:
        with open(filename, "r") as file:
            content = file.read()
            return content
    except FileNotFoundError:
        return f"Error: '{filename}' does not exist!"
    except PermissionError:
        return f"Error: No permission to read '{filename}'!"
    except Exception as e:
        return f"Unexpected error: {e}" 

print(read_file_safely("my_notes.txt"))
print(read_file_safely("missing.txt"))  


# --- SECTION 5: Real File Processor ---

def create_sample_document(filename):
    """Creates a sample document for testing"""
    content = """Introduction to AI Engineering
    
    Artificial Intelligence is transforming every industry.
    Machine learning models can now understand text, images, and speech.
    
    What is RAG?
    RAG stands for Retrieval Augmented Generation.
    It combines document search with language models.
    Companies use RAG to build intelligent document search systems.
    
    What is Fine-tuning?
    Fine-tuning adapts a pre-trained model to a specific task.
    It requires less data than training from scratch.
    """
    with open(filename, "w") as file:
        file.write(content)
    print(f"Document '{filename}' created successfully!")


def read_document(filename):
    """Reads a document and returns its content"""
    try:
        with open(filename, "r") as file:
            content = file.read()
        return content
    except FileNotFoundError:
        return None
    

def clean_text(text):
    """Cleans text by stripping whitespace from each line"""
    lines = text.split("\n")
    cleaned = [line.strip() for line in lines if line.strip()]
    return cleaned


def process_document(filename):
    """Main function — reads, cleans, and analyzes a document"""
    print(f"\n=== Processing: {filename} ===")

    # Step 1 — Read
    content = read_document(filename)
    if content is None:
        print(f"Could not process '{filename}' — file not found!")
        return
    
    # Step 2 — Clean
    lines = clean_text(content)

    # Step 3 — Analyze
    print(f"Total lines: {len(lines)}")
    print(f"Total characters: {len(content)}")
    print(f"First line: {lines[0]}")
    print(f"Last line: {lines[-1]}")


    # Step 4 — Return structured result
    return {
        "filename": filename,
        "total_lines": len(lines),
        "total_chars": len(content),
        "lines": lines
    }


# --- Run the processor ---
create_sample_document("ai_document.txt")
result = process_document("ai_document.txt")
result2 = process_document("nonexistent.txt")

# Print all cleaned lines
print("\n=== All Cleaned Lines ===")
for index, line in enumerate(result["lines"]):
    print(f"{index + 1}. {line}")