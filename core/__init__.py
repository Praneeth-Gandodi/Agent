# Parallel eager imports to cut SDK startup overhead.
# openai and anthropic share heavy deps (httpx, pydantic, etc).
# Strategy: pre-warm shared deps first, then load both SDKs in parallel.
# Without this: sequential load takes ~3-4s.
# With this: wall time drops to ~2.5-2.7s.

from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib
import sys

_SHARED_DEPS = [
    "httpx",
    "pydantic",
    "certifi",
    "httpcore",
    "anyio",
    "sniffio",
    "h11",
    "idna",
]


def _parallel_import(*names: str) -> None:
    needed = [n for n in names if n not in sys.modules]
    if not needed:
        return
    with ThreadPoolExecutor(max_workers=len(needed)) as ex:
        fs = {ex.submit(importlib.import_module, n): n for n in needed}
        for f in as_completed(fs):
            f.result()


_parallel_import(*_SHARED_DEPS)
_parallel_import("openai", "anthropic")
