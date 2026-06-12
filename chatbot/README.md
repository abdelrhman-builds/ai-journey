# 🤖 Zaki — AI Engineering Assistant

A terminal chatbot with full conversation memory, token tracking,
and session statistics. Built with Python and Google Gemini API.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Gemini](https://img.shields.io/badge/Gemini-2.5--flash-orange)
![Status](https://img.shields.io/badge/Status-Complete-green)

---

## 🎯 What It Does

Zaki is a multi-turn AI chatbot that:
- Remembers the full conversation history
- Tracks token usage per message and session total
- Supports commands: `/clear`, `/history`, `/stats`, `/help`, `/quit`
- Shows session summary on exit

---

## 🧠 Key Technical Concepts

**Why AI needs conversation history:**
Each API call is stateless — the model remembers nothing between calls.
We solve this by sending the FULL conversation history with every call.
Longer conversations = more tokens per call (visible in `/stats`).

**Token tracking:**
Every API call returns `usage_metadata` with token counts.
This lets us monitor usage and stay within free tier limits.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.11 | Core language |
| Google Gemini API | LLM backend |
| google-genai | Gemini SDK |
| python-dotenv | API key management |

---

## ⚙️ Setup

```bash
# Install dependencies
pip install google-genai python-dotenv

# Create .env file
echo "GEMINI_API_KEY=your_key_here" > .env

# Run the chatbot
python day9_chatbot.py
```

---

## 💬 Commands

| Command | Description |
|---------|-------------|
| `/clear` | Reset conversation and stats |
| `/history` | Show all messages so far |
| `/stats` | Show token usage |
| `/help` | Show all commands |
| `/quit` | Exit with session summary |

---

## 📊 Example Session

You: My name is Abdelrhman and I study AI engineering
Zaki: Hello Abdelrhman! Great to meet you.
AI engineering is a fantastic field to study!
[tokens: 245 | total: 245]

You: What is my name?
Zaki: Your name is Abdelrhman!
[tokens: 312 | total: 557]

You: /stats
📊 Current Stats:
Messages: 2
Tokens used: 557
History length: 4 entries

You: /quit
Session Summary:
Messages sent:  2
Total tokens:   557
Avg per msg:    278

---

## 🔗 Part of 90-Day AI Engineering Journey

Built on Day 9 of my 90-day journey from financial auditor to AI engineer.

**GitHub:** [@abdelrhman-builds](https://github.com/abdelrhman-builds)