import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2:3b"

SYSTEM_PROMPT = """
You are a helpful AI/ML learning assistant.
Explain concepts clearly and simply for a beginner.
Use concise examples when useful.
Do not invent facts.
"""

prompts = [
    "What is supervised learning?",
    "What is overfitting and how can it be reduced?",
    "Explain classification versus regression.",
    "What is cosine similarity?",
    "What is a vector database and why is it useful for semantic search?",
]


def ask_ollama(prompt):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "system": SYSTEM_PROMPT,
            "prompt": prompt,
            "stream": False,
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["response"]


print("=" * 70)
print("W5D5: Local Q&A Bot - Ollama")
print("=" * 70)
print(f"Model: {MODEL}")
print("System prompt: AI/ML beginner learning assistant")

for i, prompt in enumerate(prompts, start=1):
    print("\n" + "=" * 70)
    print(f"PROMPT {i}")
    print("=" * 70)
    print(f"Question: {prompt}")

    answer = ask_ollama(prompt)

    print("\nAnswer:")
    print(answer)

print("\n" + "=" * 70)
print("MANUAL VERIFICATION")
print("=" * 70)
print("1. Ollama API connection successful.")
print("2. Custom system prompt used.")
print("3. Five prompts tested successfully.")
print("4. Local LLM responses generated successfully.")
print("\nW5D5 Ollama Q&A bot completed successfully.")
