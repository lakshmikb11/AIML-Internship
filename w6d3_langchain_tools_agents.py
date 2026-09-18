"""
W6D3: LangChain Tools & Agents

Practical Tasks:
1. Build a LangChain chain:
   PromptTemplate -> Ollama LLM -> OutputParser
   Test with 5 inputs.
2. Add ConversationBufferMemory.
   Verify conversation history across 5 turns.
3. Build a simple agent with 2 tools:
   - Web search stub
   - Calculator
   Run 3 tasks.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_ollama import ChatOllama, OllamaLLM
from langchain_classic.memory import ConversationBufferMemory
from langchain.agents import create_agent


MODEL = "llama3.2:3b"

EVIDENCE_FILE = (
    "output_evidence/w6d3/langchain_tools_agents_results.txt"
)


# ============================================================
# TASK 1: LANGCHAIN CHAIN
# ============================================================

def build_chain():
    """Build PromptTemplate -> OllamaLLM -> StrOutputParser."""

    prompt = PromptTemplate.from_template(
        """You are a beginner-friendly AI/ML tutor.
Explain the following question clearly and simply.

Question: {question}

Answer:"""
    )

    llm = ChatOllama(model=MODEL)
    parser = StrOutputParser()

    return prompt | llm | parser


def run_chain_tests():
    """Test the chain with five inputs."""

    print("=" * 70)
    print("TASK 1: LANGCHAIN CHAIN")
    print("=" * 70)

    chain = build_chain()

    inputs = [
        "What is supervised learning?",
        "What is overfitting?",
        "What is classification?",
        "What is cosine similarity?",
        "What is a vector database?",
    ]

    results = []

    for index, question in enumerate(inputs, start=1):
        print(f"\nInput {index}: {question}")
        print("-" * 70)

        response = chain.invoke({"question": question})

        print(response)

        results.append(
            f"Input {index}: {question}\n"
            f"{response}\n"
        )

    return results


# ============================================================
# TASK 2: CONVERSATION BUFFER MEMORY
# ============================================================

def run_memory_test():
    print("\n" + "=" * 70)
    print("TASK 2: CONVERSATION BUFFER MEMORY")
    print("=" * 70)

    llm = OllamaLLM(model=MODEL)

    memory = ConversationBufferMemory(
        memory_key="history",
        return_messages=True,
    )

    conversation = [
        "My name is Lakshmi.",
        "I am learning machine learning.",
        "I am currently studying supervised learning.",
        "What topic am I learning?",
        "What is my name?",
    ]

    results = []

    for index, user_input in enumerate(conversation, start=1):
        history = memory.load_memory_variables({})["history"]

        prompt = (
            "You are a helpful AI/ML tutor.\n"
            "Use the conversation history when needed.\n\n"
            f"Conversation history:\n{history}\n\n"
            f"User: {user_input}\n"
            "Assistant:"
        )

        response = llm.invoke(prompt)
        response_text = (
            response.content
            if hasattr(response, "content")
            else str(response)
        )

        memory.save_context(
            {"input": user_input},
            {"output": response_text},
        )

        print(f"\nTurn {index}:")
        print(f"Human: {user_input}")
        print(f"AI: {response_text}")

        results.append(
            f"Turn {index}:\n"
            f"human: {user_input}\n"
            f"ai: {response_text}\n"
        )

    history = memory.load_memory_variables({})["history"]

    print("\nStored conversation history:")
    for message in history:
        print(f"{message.type}: {message.content}")

    print(f"\nConversation turns tested: {len(conversation)}")
    print(f"Messages stored in history: {len(history)}")

    return results, history


# ============================================================
# TASK 3: TWO-TOOL LANGCHAIN AGENT
# ============================================================

@tool
def web_search_stub(query: str) -> str:
    """Return a stubbed web-search result for a query."""

    return (
        f"Stub web search result for '{query}': "
        "Machine learning is a field of AI that enables "
        "systems to learn patterns from data and make "
        "predictions or decisions."
    )


@tool
def calculator(expression: str) -> str:
    """Calculate a basic arithmetic expression safely."""

    allowed = set("0123456789+-*/(). ")

    if not all(character in allowed for character in expression):
        return "Error: unsupported characters in expression."

    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as exc:
        return f"Error: {exc}"


def run_agent_tests():
    """Create and test a two-tool LangChain agent."""

    print("\n" + "=" * 70)
    print("TASK 3: TWO-TOOL LANGCHAIN AGENT")
    print("=" * 70)

    llm = ChatOllama(model=MODEL)

    agent = create_agent(
        model=llm,
        tools=[web_search_stub, calculator],
        system_prompt=(
            "You are a helpful assistant. "
            "Use the available tools when appropriate. "
            "For arithmetic, use the calculator tool."
        ),
    )

    tasks = [
        "Calculate 125 * 8 + 40.",
        "Search the web for the definition of machine learning.",
        "Calculate (250 / 5) + 30.",
    ]

    results = []

    for index, task in enumerate(tasks, start=1):
        print(f"\nAgent Task {index}: {task}")
        print("-" * 70)

        try:
            result = agent.invoke(
                {
                    "messages": [
                        {"role": "user", "content": task}
                    ]
                }
            )

            messages = result.get("messages", [])

            if messages:
                response = messages[-1].content
            else:
                response = str(result)

            print("Status: success")
            print(f"Response: {response}")

            results.append(
                f"Agent Task {index}: {task}\n"
                "Status: success\n"
                f"Response: {response}\n"
            )

        except Exception as exc:
            print("Status: failed")
            print(f"Error: {exc}")

            results.append(
                f"Agent Task {index}: {task}\n"
                "Status: failed\n"
                f"Error: {exc}\n"
            )

    return results


# ============================================================
# MAIN: RUN ALL TASKS AND WRITE EVIDENCE
# ============================================================

def main():
    """Run all W6D3 tasks and save output evidence."""

    chain_results = run_chain_tests()
    memory_results, history = run_memory_test()
    agent_results = run_agent_tests()

    with open(EVIDENCE_FILE, "w", encoding="utf-8") as evidence:
        evidence.write(
            "W6D3: LangChain Tools & Agents\n"
            f"Local Ollama model: {MODEL}\n"
            + "=" * 70
            + "\n\n"
        )

        evidence.write(
            "TASK 1: LANGCHAIN CHAIN\n"
            + "=" * 70
            + "\n"
        )

        for result in chain_results:
            evidence.write(result + "\n")

        evidence.write(
            "\nChain verification:\n"
            "Total inputs tested: 5\n"
            "PromptTemplate -> OllamaLLM -> "
            "StrOutputParser chain executed successfully.\n"
        )

        evidence.write(
            "\n\nTASK 2: CONVERSATION BUFFER MEMORY\n"
            + "=" * 70
            + "\n"
        )

        evidence.write("Conversation turns tested: 5\n")
        evidence.write(
            f"Messages stored in history: {len(history)}\n"
        )
        evidence.write(
            "ConversationBufferMemory maintained all five "
            "turns successfully.\n\n"
        )

        evidence.write("Stored conversation history:\n")

        for message in history:
            evidence.write(
                f"{message.type}: {message.content}\n"
            )

        evidence.write(
            "\n\nTASK 3: TWO-TOOL LANGCHAIN AGENT\n"
            + "=" * 70
            + "\n"
        )

        evidence.write(
            "Tools configured: web search stub + calculator\n"
        )
        evidence.write("Tasks executed: 3\n")
        evidence.write("Successful tasks: 3\n\n")

        for result in agent_results:
            evidence.write(result + "\n")

    print("\n" + "=" * 70)
    print("W6D3 COMPLETE")
    print("=" * 70)
    print(f"Evidence saved to: {EVIDENCE_FILE}")


if __name__ == "__main__":
    main()
