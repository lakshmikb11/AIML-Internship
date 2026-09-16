"""
W5D1: Local LLM Inference with Ollama

This script:
1. Connects to the locally running Ollama API.
2. Uses a custom system prompt.
3. Tests the Llama 3.2 3B model with five prompts.
4. Prints the model responses for evaluation.
"""

import json
from urllib.error import URLError
from urllib.request import Request, urlopen


OLLAMA_API_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "llama3.2:3b"

SYSTEM_PROMPT = (
    "You are a helpful AI/ML learning assistant. "
    "Explain concepts clearly and simply for a beginner. "
    "Use short examples when useful and avoid unnecessary jargon."
)

TEST_PROMPTS = [
    "What is supervised learning?",
    "Explain overfitting in simple terms.",
    "What is the difference between classification and regression?",
    "Why do we split data into training and testing sets?",
    "What is a confusion matrix used for?",
]


def ask_ollama(user_prompt):
    """Send a prompt to the local Ollama API and return the response."""

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
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
    """Run five test prompts against the local LLM."""

    print("=" * 70)
    print("W5D1 - Ollama Local LLM Inference")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print("API:   http://localhost:11434/api/chat")
    print("\nCustom System Prompt:")
    print(SYSTEM_PROMPT)

    for index, prompt in enumerate(TEST_PROMPTS, start=1):
        print("\n" + "-" * 70)
        print(f"TEST {index}")
        print("-" * 70)
        print(f"Prompt: {prompt}")

        response = ask_ollama(prompt)

        print("\nResponse:")
        print(response)

    print("\n" + "=" * 70)
    print("All five prompts completed.")
    print("=" * 70)


if __name__ == "__main__":
    main()