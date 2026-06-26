# Agent

## Status

| Module                 | Status         | Notes                                                                                   |
| ---------------------- | -------------- | --------------------------------------------------------------------------------------- |
| `BaseProvider`         | ✅ Done        | Abstract interface                                                                      |
| `ChatCompletions`      | ✅ Done        | `client.chat.completions.create()`                                                      |
| `Responses API`        | 🔄 In progress |                                                                                         |
| `Messages API`         | ❌ Not started |                                                                                         |
| `Normalizer`           | 🔄 Partial     | Only handles `ChatCompletionChunk` — needs `ResponseStreamEvent` + `MessageStreamEvent` |
| `Messages API`         | ❌ Not started |                                                                                         |
| `Capability Registry`  | ❌ Not started | Track which models support vision, audio, tools, etc.                                   |
| `Conversation Manager` | ❌ Not started | History, context window, token counting                                                 |
| `Tool Executor`        | ❌ Not started | Run function calls, inject results                                                      |
| `Audio I/O`            | ❌ Not started | STT + TTS                                                                               |

## Setup

## Windows

```ps1
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Linux / MacOS

```bash
python -m venv .venv
source .venv/scripts/activate
pip install -r requirements.txt
```
