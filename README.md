# Agent

Multi-provider AI agent framework with normalized streaming, tool calling, and reasoning support.

## Implementation Status

### Layer 1: Providers

```
[x] ChatCompletion Provider      — core/providers/chatcompletion.py
[x] ResponseAPI Provider         — core/providers/response.py
[x] Messages API Provider        — core/providers/messages.py
```

### Layer 2: Normalizers

```
[x] Normalizer for ChatCompletion   — core/normalizer/normalizer.py (handles ChatCompletionChunk + ChatCompletion)
[x] Normalizer for ResponseAPI      — core/normalizer/normalizer.py (handles OpenAIResponseStreamEvent + Response)
[x] Normalizer for Messages API     — core/normalizer/normalizer.py (handles RawMessageStreamEvent + Message)
```

### Layer 3: Adapter

```
[ ] Input Adapter — Accept text / files / images, detect modality support, route to appropriate provider
[ ] Vision detection — Check capability registry if model supports vision
```

### Layer 4: Tool Calling

```
[ ] Tool Executor — Run function calls, inject results back
[ ] MCP Client — Model Context Protocol client for external tool servers
```

### Layer 5: User Interface

```
[ ] TUI — Built with OpenTUI (JS framework)
[ ] WebUI — Browser-based interface
[ ] Voice Interaction Layer:
      [ ] Reactive ring animation — Waveform/orb animation for listening/speaking state
      [ ] TTS — Text-to-speech output
      [ ] STT — Speech-to-text input
      [ ] OpenWakeWord — Wake word detection ("Hey Agent")
```

## Project Structure

```
core/
├── __init__.py              # Parallel import optimizer
├── providers/
│   ├── base.py              # Abstract BaseProvider
│   ├── chatcompletion.py    # OpenAIChatCompletion (done)
│   ├── messages.py          # Anthropic Messages API
│   ├── response.py          # OpenAI Responses API
│   └── models.py            # Shared Pydantic models & type aliases
└── normalizer/
    ├── models.py            # Normalized streaming event types
    └── normalizer.py        # ResponseNormalizer (ChatCompletion, Response API, Anthropic Messages)
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
