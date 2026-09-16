"""
W5D2: Llama 3.2 3B vs Qwen 2.5 3B Model Comparison

This script:
1. Calls both local Ollama models using the same API.
2. Uses the same system prompt for both models.
3. Tests the same three questions.
4. Prints responses for qualitative comparison.
"""

import json
from urllib.error import URLError
from urllib.request import Request, urlopen


OLLAMA_API_URL = "http://localhost:11434/api/chat"

MODELS = [
    "llama3.2:3b",
    "qwen2.5:3b",
]

SYSTEM_PROMPT = (
    "You are an AI/ML tutor for beginners. "
    "Explain concepts clearly using simple language. "
    "Give a short example when useful. "
    "Structure your answers with concise points and avoid unnecessary jargon."
)

QUESTIONS = [
    "What is supervised learning?",
    "Explain overfitting in simple terms.",
    "What is the difference between classification and regression?",
]


def ask_ollama(model, question):
    """Send a question to the selected Ollama model."""

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "stream": False,
    }

    request = Request(
        OLLAMA_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))

        return result["message"]["content"]

    except URLError as error:
        return f"ERROR: Could not connect to Ollama. {error}"
    except (KeyError, json.JSONDecodeError) as error:
        return f"ERROR: Unexpected Ollama response. {error}"


def main():
    """Compare both models using the same three questions."""

    print("=" * 80)
    print("W5D2 - Llama 3.2 3B vs Qwen 2.5 3B")
    print("=" * 80)

    print("\nSystem Prompt:")
    print(SYSTEM_PROMPT)

    for question_number, question in enumerate(QUESTIONS, start=1):
        print("\n" + "=" * 80)
        print(f"QUESTION {question_number}")
        print("=" * 80)
        print(f"\nQuestion: {question}")

        for model in MODELS:
            print("\n" + "-" * 80)
            print(f"MODEL: {model}")
            print("-" * 80)

            response = ask_ollama(model, question)

            print(response)

    print("\n" + "=" * 80)
    print("Model comparison completed.")
    print("=" * 80)


if __name__ == "__main__":
    main()