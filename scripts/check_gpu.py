import sys

try:
    import llama_cpp  # type: ignore
    print("llama_cpp available")
except Exception as exc:  # pragma: no cover
    print("llama_cpp not available:", exc)

print("OK")
