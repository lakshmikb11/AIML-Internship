import requests

OLLAMA_URL = "http://localhost:11434/api/generate"

MODELS = [
    "llama3.2:3b",
    "qwen2.5:3b",
]

QUESTIONS = [
    "What is supervised learning? Give a simple example.",
    "What is overfitting and how can it be reduced?",
    "Explain classification versus regression with examples.",
]


def ask_ollama(model, question):
    system_prompt = """
You are an AI/ML learning assistant.
Explain concepts clearly for a beginner.
Give accurate, concise answers with simple examples.
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "system": system_prompt,
            "prompt": question,
            "stream": False,
        },
        timeout=120,
    )

    response.raise_for_status()
    return response.json()["response"]


print("=" * 70)
print("W5D5: Ollama Model Comparison")
print("=" * 70)

for question_number, question in enumerate(QUESTIONS, start=1):
    print("\n" + "=" * 70)
    print(f"QUESTION {question_number}")
    print("=" * 70)
    print(question)

    for model in MODELS:
        print("\n" + "-" * 70)
        print(f"MODEL: {model}")
        print("-" * 70)

        answer = ask_ollama(model, question)
        print(answer)

print("\n" + "=" * 70)
print("MANUAL VERIFICATION")
print("=" * 70)
print("1. Same 3 questions tested on both models.")
print("2. llama3.2:3b responses generated successfully.")
print("3. qwen2.5:3b responses generated successfully.")
print("4. Responses are ready for manual quality comparison.")
print("\nW5D5 model comparison completed successfully.")
