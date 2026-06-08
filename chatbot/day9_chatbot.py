# =============================================
# Day 9 - Multi-turn Chatbot with Memory
# Author: Abdelrhman
# Date: June 2026
# =============================================
# Key concept: AI has no memory between calls.
# Solution: Send the FULL conversation history
# with every API call. We store it as a list.
# =============================================

import os
import google.genai as genai
from google.genai import types
from dotenv import load_dotenv

# Load API key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

# =============================================
# SECTION 1: Core Chat Function
# =============================================

def chat(history, user_message, system_prompt):
    """
    Sends a message and gets a response.
    Includes full conversation history so AI remembers.

    Args:
        history: List of previous messages (role + content)
        user_message: The new message from the user
        system_prompt: AI persona and behavior instructions

    Returns:
        AI response as string
    """
    # Add the new user message to history
    history.append({
        "role": "user",
        "parts": [{"text": user_message}]
    })

    try:
        # Send FULL history to API — this is why AI remembers
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=history,       # entire conversation, not just latest message
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.7,
                max_output_tokens=500,
                thinking_config=types.ThinkingConfig(thinking_budget=0)
            )
        )

        # Get the AI's response text
        ai_response = response.text.strip()

        # Add AI response to history so next call remembers it too
        history.append({
            "role": "model",
            "parts": [{"text": ai_response}]
        })

        return ai_response

    except Exception as e:
        # Remove the user message we just added if call failed
        history.pop()
        return f"Error: {e}"


# =============================================
# SECTION 2: Display Functions
# =============================================

def show_history(history):
    """Displays the full conversation history"""
    if not history:
        print("No conversation history yet.")
        return

    print("\n=== Conversation History ===")
    for i, message in enumerate(history):
        # Get role and capitalize it
        role = message["role"].upper()
        # Get the text content
        text = message["parts"][0]["text"]
        print(f"\n[{i+1}] {role}:")
        print(f"  {text}")
    print("=" * 30)


def show_help():
    """Shows available commands"""
    print("""
=== Available Commands ===
/clear   → Reset conversation (start fresh)
/history → Show all messages so far
/help    → Show this menu
/quit    → Exit the chatbot
""")


# =============================================
# SECTION 3: Main Chatbot Loop
# =============================================

def run_chatbot():
    """
    Main function — runs the chatbot loop.
    Keeps asking for input until user types /quit
    """

    # Define the chatbot's persona via system prompt
    system_prompt = """You are Zaki, a friendly AI assistant helping
Abdelrhman learn AI engineering. You are encouraging, clear, and concise.
You remember everything discussed in the conversation.
When asked about previous messages, refer to them directly.
Keep responses under 150 words unless asked for more detail."""

    # Initialize empty conversation history
    # This list grows with every message sent and received
    history = []

    # Welcome message
    print("=" * 50)
    print("   🤖 Zaki — Your AI Engineering Assistant")
    print("=" * 50)
    print("Type /help for available commands")
    print("Type /quit to exit")
    print("-" * 50)

    # Main loop — keeps running until user quits
    while True:
        # Get input from user
        user_input = input("\nYou: ").strip()

        # Skip empty input
        if not user_input:
            continue

        # Handle commands
        if user_input.lower() == "/quit":
            print("\nZaki: Goodbye Abdelrhman! Keep building! 🚀")
            break

        elif user_input.lower() == "/clear":
            history = []    # empty the history list
            print("\n✅ Conversation cleared. Starting fresh!")
            continue

        elif user_input.lower() == "/history":
            show_history(history)
            continue

        elif user_input.lower() == "/help":
            show_help()
            continue

        # Regular message — send to AI
        print("\nZaki: ", end="", flush=True)
        response = chat(history, user_input, system_prompt)
        print(response)



# =============================================
# SECTION 4: Improved Chatbot with Stats
# =============================================

def chat_with_stats(history, user_message, system_prompt):
    """
    Enhanced chat function that also returns token usage.

    Returns:
        Tuple of (ai_response, tokens_used)
    """
    history.append({
        "role": "user",
        "parts": [{"text": user_message}]
    })

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=history,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.7,
                max_output_tokens=500,
                thinking_config=types.ThinkingConfig(thinking_budget=0)
            )
        )

        ai_response = response.text.strip()

        # Extract token usage from metadata
        tokens_used = response.usage_metadata.total_token_count

        history.append({
            "role": "model",
            "parts": [{"text": ai_response}]
        })

        return ai_response, tokens_used

    except Exception as e:
        history.pop()
        return f"Error: {e}", 0


def run_chatbot_v2():
    """
    Improved chatbot with message counter and token tracking.
    """
    system_prompt = """You are Zaki, a friendly AI assistant helping
Abdelrhman learn AI engineering. You are encouraging, clear, and concise.
You remember everything discussed in the conversation.
Keep responses under 150 words unless asked for more detail."""

    history = []
    message_count = 0       # counts total messages sent
    total_tokens = 0        # tracks total tokens used

    print("=" * 50)
    print("   🤖 Zaki v2 — With Usage Stats")
    print("=" * 50)
    print("Type /help for commands | /quit to exit")
    print("-" * 50)

    while True:
        user_input = input("\nYou: ").strip()

        if not user_input:
            continue

        if user_input.lower() == "/quit":
            # Show session summary before quitting
            print(f"\n{'=' * 50}")
            print(f"Session Summary:")
            print(f"  Messages sent:  {message_count}")
            print(f"  Total tokens:   {total_tokens}")
            print(f"  Avg per msg:    {total_tokens // max(message_count, 1)}")
            print(f"{'=' * 50}")
            print("Zaki: Goodbye! Keep building! 🚀")
            break

        elif user_input.lower() == "/clear":
            history = []
            message_count = 0
            total_tokens = 0
            print("✅ Conversation and stats cleared!")
            continue

        elif user_input.lower() == "/history":
            show_history(history)
            continue

        elif user_input.lower() == "/help":
            show_help()
            continue

        elif user_input.lower() == "/stats":
            print(f"\n📊 Current Stats:")
            print(f"  Messages: {message_count}")
            print(f"  Tokens used: {total_tokens}")
            print(f"  History length: {len(history)} entries")
            continue

        # Regular message
        message_count += 1
        print(f"\nZaki: ", end="", flush=True)
        response, tokens = chat_with_stats(history, user_input, system_prompt)
        total_tokens += tokens
        print(response)
        print(f"  [tokens: {tokens} | total: {total_tokens}]")


# Run the chatbot
if __name__ == "__main__":
    run_chatbot_v2()