"""
W7D4: Ollama — Local LLM Inference

Tasks:
1. Call Ollama locally using llama3.2:3b.
2. Use a custom system prompt.
3. Test 5 prompts.
4. Compare llama3.2:3b and qwen2.5:3b on 3 common questions.
"""

import requests

OLLAMA_URL = "http://localhost:11434/api/chat"

SYSTEM_PROMPT = (
    "You are a helpful AI/ML learning assistant. "
    "Explain technical concepts clearly and simply. "
    "Use concise answers with examples when useful."
)

LLAMA_MODEL = "llama3.2:3b"
QWEN_MODEL = "qwen2.5:3b"

TEST_PROMPTS = [
    "What is supervised learning?",
    "Explain overfitting in machine learning.",
    "What is the difference between classification and regression?",
    "What is a vector database?",
    "What is Retrieval-Augmented Generation (RAG)?",
]

COMPARISON_QUESTIONS = [
    "What is machine learning?",
    "Explain how RAG works.",
    "What are the advantages of using a vector database?",
]


def call_ollama(model, prompt):
    """Send a prompt to the local Ollama API."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=120,
    )
    response.raise_for_status()

    return response.json()["message"]["content"]


def run_task_2():
    """Test llama3.2:3b with five prompts."""
    print("=" * 70)
    print("TASK 2: OLLAMA API — CUSTOM SYSTEM PROMPT")
    print("=" * 70)
    print(f"Model: {LLAMA_MODEL}")
    print(f"System prompt: {SYSTEM_PROMPT}")
    print()

    results = []

    for index, prompt in enumerate(TEST_PROMPTS, start=1):
        print(f"PROMPT {index}: {prompt}")

        answer = call_ollama(LLAMA_MODEL, prompt)

        print("RESPONSE:")
        print(answer)
        print("-" * 70)

        results.append((prompt, answer))

    return results


def run_task_3():
    """Compare llama3.2:3b and qwen2.5:3b."""
    print()
    print("=" * 70)
    print("TASK 3: MODEL COMPARISON")
    print("=" * 70)

    comparisons = []

    for index, question in enumerate(COMPARISON_QUESTIONS, start=1):
        print(f"\nQUESTION {index}: {question}")
        print("-" * 70)

        llama_answer = call_ollama(LLAMA_MODEL, question)
        qwen_answer = call_ollama(QWEN_MODEL, question)

        print(f"\n{LLAMA_MODEL.upper()} RESPONSE:")
        print(llama_answer)

        print(f"\n{QWEN_MODEL.upper()} RESPONSE:")
        print(qwen_answer)

        print("-" * 70)

        comparisons.append(
            {
                "question": question,
                "llama": llama_answer,
                "qwen": qwen_answer,
            }
        )

    return comparisons


def save_results(task_2_results, comparisons):
    """Save all inference results as evidence."""
    output_path = (
        "output_evidence/w7d4/"
        "ollama_local_inference_results.txt"
    )

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("W7D4: Ollama Local LLM Inference\n")
        file.write("=" * 70 + "\n\n")

        file.write("TASK 1: FIRST LOCAL INFERENCE\n")
        file.write("-" * 70 + "\n")
        file.write(
            "llama3.2:3b successfully generated a local response "
            "about Retrieval-Augmented Generation (RAG).\n\n"
        )

        file.write("TASK 2: CUSTOM SYSTEM PROMPT — 5 TESTS\n")
        file.write("=" * 70 + "\n\n")

        for index, (prompt, answer) in enumerate(
            task_2_results, start=1
        ):
            file.write(f"PROMPT {index}: {prompt}\n")
            file.write("RESPONSE:\n")
            file.write(answer)
            file.write("\n\n" + "-" * 70 + "\n\n")

        file.write("TASK 3: LLAMA VS QWEN — 3 QUESTIONS\n")
        file.write("=" * 70 + "\n\n")

        for index, comparison in enumerate(comparisons, start=1):
            file.write(
                f"QUESTION {index}: {comparison['question']}\n\n"
            )

            file.write("LLAMA3.2:3B RESPONSE:\n")
            file.write(comparison["llama"])
            file.write("\n\n")

            file.write("QWEN2.5:3B RESPONSE:\n")
            file.write(comparison["qwen"])
            file.write("\n\n")

            file.write("-" * 70 + "\n\n")

    print(f"\nEvidence saved to: {output_path}")


def main():
    """Run W7D4 tasks."""
    task_2_results = run_task_2()
    comparisons = run_task_3()
    save_results(task_2_results, comparisons)

    print()
    print("=" * 70)
    print("W7D4 TASKS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()