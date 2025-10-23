from __future__ import annotations

import time

from app.deps import get_app_state
from app.services.retrieve_service import retrieve


def main() -> None:
    state = get_app_state()
    decision = state.router.route("Who is the CEO today?", None, signals=None)
    start = time.time()
    retrieve("Who is the CEO today?", decision.window, decision.mode, top_k=3)
    duration = (time.time() - start) * 1000
    print(f"Retrieve p50 ~= {duration:.2f} ms")


if __name__ == "__main__":
    main()
