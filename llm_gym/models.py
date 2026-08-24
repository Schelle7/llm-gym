"""The models this workbench can run, named by their ollama tag.

A model's id is its tag, which is also what a reply carries back in
`response_metadata["model"]`, so a chat row can name what wrote it
without a lookup that could drift.
"""

MODELS = (
    "gemma4:e2b",
    "gemma4:26b",
    "qwen3.5:4b",
    "qwen3.5:9b",
    "qwen2.5-coder:7b",
)
