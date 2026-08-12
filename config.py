from openai import OpenAI


MODEL = "gemma4:e2b"
MAX_ITERATIONS = 5
SYSTEM_PROMPT = (
    "You are a local assistant. Use only the tools provided. "
    "Do not invent tool results. If no tool is needed, answer directly."
)

client = OpenAI(
    base_url="http://127.0.0.1:11434/v1",
    api_key="ollama",
)

def create_messages(user_prompt):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
