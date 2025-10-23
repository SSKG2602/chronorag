def make_cache_key(tenant: str, query: str, axis: str, window: str) -> str:
    return f"chronorag:{tenant}:{axis}:{window}:{hash(query)}"
