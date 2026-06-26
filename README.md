# Agent

Multi-provider AI agent framework with normalized streaming, tool calling, and reasoning support.

## Status

| Module                 | Status         | Notes                                                                                   |
| ---------------------- | -------------- | --------------------------------------------------------------------------------------- |
| `BaseProvider`         | ✅ Done        | Abstract interface                                                                      |
| `ChatCompletions`      | ✅ Done        | `client.chat.completions.create()`                                                      |
| `Responses API`        | 🔄 In progress | Stub only                                                                               |
| `Messages API`         | ❌ Not started | Placeholder only                                                                        |
| `Normalizer`           | 🔄 Partial     | Only handles `ChatCompletionChunk` — needs `ResponseStreamEvent` + `MessageStreamEvent` |
| `Capability Registry`  | ❌ Not started | Track which models support vision, audio, tools, etc.                                   |
| `Conversation Manager` | ❌ Not started | History, context window, token counting                                                 |
| `Tool Executor`        | ❌ Not started | Run function calls, inject results                                                      |
| `Audio I/O`            | ❌ Not started | STT + TTS                                                                               |

## Project Structure

```
core/
├── __init__.py              # Parallel import optimizer
├── providers/
│   ├── base.py              # Abstract BaseProvider
│   ├── chatcompletion.py    # OpenAIChatCompletion (done)
│   ├── messages.py          # Anthropic Messages API (placeholder)
│   ├── response.py          # OpenAI Responses API (stub)
│   └── models.py            # Shared Pydantic models & type aliases
└── normalizer/
    ├── models.py            # Normalized streaming event types
    └── normalizer.py        # ResponseNormalizer (partial — ChatCompletionChunk only)
```
## Setup

### Windows

```ps1
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
