# 🤖 Zaki — AI Engineering Assistant

A terminal chatbot with full conversation memory, built with Python and Google Gemini API.

## Features
- ✅ Multi-turn conversation with memory
- ✅ Token usage tracking
- ✅ Session statistics
- ✅ Commands: /clear, /history, /stats, /help, /quit

## Tech Stack
- Python 3.11
- Google Gemini API (gemini-2.5-flash)
- python-dotenv

## How to Run

```bash
# Install dependencies
pip install google-genai python-dotenv

# Add your API key to .env
GEMINI_API_KEY=your_key_here

# Run the chatbot
cd chatbot
python day9_chatbot.py
```

## Commands
| Command | Description |
|---------|-------------|
| /clear | Reset conversation |
| /history | Show all messages |
| /stats | Show token usage |
| /help | Show commands |
| /quit | Exit with summary |

## Example Conversation

You: My name is Abdelrhman
Zaki: Hello Abdelrhman! How can I help?
You: What is my name?
Zaki: Your name is Abdelrhman! 😊

## Built by
[Abdelrhman](https://github.com/abdelrhman-builds) — Day 9 of 90-day AI Engineering Journey