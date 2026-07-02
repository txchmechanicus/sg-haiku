from __future__ import annotations

from collections.abc import AsyncIterator

import httpx


async def iter_sse_data(response: httpx.Response) -> AsyncIterator[str]:
    async for line in response.aiter_lines():
        if not line or line.startswith(":") or not line.startswith("data:"):
            continue
        yield line.removeprefix("data:").strip()
