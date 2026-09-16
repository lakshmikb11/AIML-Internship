"""
W6D1: LangChain Fundamentals - Chains & Prompts

Tasks:
1. Build and test a LangChain chain:
   PromptTemplate -> Ollama LLM -> OutputParser
2. Add ConversationBufferMemory and verify 5 turns.
3. Build a LangChain agent with two tools:
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

    llm = OllamaLLM(model=MODEL)
    parser = StrOutputParser()

    chain = prompt | llm | parser

    return chain


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
            {
                "input": question,
                "response": response,
            }
        )

    print("\nChain verification:")
    print(f"Total inputs tested: {len(results)}")
    print("PromptTemplate -> OllamaLLM -> StrOutputParser chain executed successfully.")

    return results


# ============================================================
# TASK 2: CONVERSATION BUFFER MEMORY
# ============================================================

def run_memory_tests():
    """Test ConversationBufferMemory across five conversation turns."""

    print("\n" + "=" * 70)
    print("TASK 2: CONVERSATION BUFFER MEMORY")
    print("=" * 70)

    memory = ConversationBufferMemory(
        memory_key="history",
        return_messages=True,
    )

    llm = OllamaLLM(model=MODEL)

    conversation = [
        "My name is Lakshmi.",
        "I am learning machine learning.",
        "I am currently studying supervised learning.",
        "What topic am I learning?",
        "What is my name?",
    ]

    for turn, user_input in enumerate(conversation, start=1):

        print(f"\nTurn {turn}")
        print(f"User: {user_input}")

        memory_variables = memory.load_memory_variables({})

        history_messages = memory_variables["history"]

        history_text = ""

        for message in history_messages:
            if hasattr(message, "content"):
                if message.type == "human":
                    history_text += f"User: {message.content}\n"
                elif message.type == "ai":
                    history_text += f"Assistant: {message.content}\n"

        prompt = f"""
You are a helpful AI/ML tutor.

Use the conversation history to answer the user's latest message.

Conversation history:
{history_text}

Current user message:
{user_input}

Answer accurately and briefly.
"""

        response = llm.invoke(prompt)

        print(f"Assistant: {response}")

        memory.save_context(
            {"input": user_input},
            {"output": response},
        )

    final_history = memory.load_memory_variables({})["history"]

    print("\nMemory verification:")
    print(f"Total conversation turns: {len(conversation)}")
    print(f"Messages stored in history: {len(final_history)}")

    if len(final_history) == 10:
        print("ConversationBufferMemory maintained all five turns successfully.")
    else:
        print("Memory verification failed.")

    return final_history


# ============================================================
# TASK 3: LANGCHAIN AGENT WITH TWO TOOLS
# ============================================================

@tool
def web_search_stub(query: str) -> str:
    """Simulate a web search and return a stub result."""

    return (
        f"Web search stub result for '{query}': "
        "Machine learning is a field of AI that enables systems "
        "to learn patterns from data and make predictions or decisions. "
        "This is a simulated web search result for the internship exercise."
    )


@tool
def calculator(expression: str) -> str:
    """Calculate a basic mathematical expression."""

    allowed_characters = "0123456789+-*/(). "

    if not all(character in allowed_characters for character in expression):
        return "Error: expression contains unsupported characters."

    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as exc:
        return f"Calculation error: {exc}"


def run_agent_tests():
    """Build an agent with two tools and run three tasks."""

    print("\n" + "=" * 70)
    print("TASK 3: LANGCHAIN AGENT WITH TWO TOOLS")
    print("=" * 70)

    # ChatOllama is required for LangChain tool binding.
    llm = ChatOllama(model=MODEL)

    tools = [
        web_search_stub,
        calculator,
    ]

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=(
            "You are a helpful AI assistant. "
            "Use the calculator tool for mathematical calculations. "
            "Use the web search stub for web-search requests. "
            "Give concise and accurate answers."
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
                        {
                            "role": "user",
                            "content": task,
                        }
                    ]
                }
            )

            messages = result.get("messages", [])

            if messages:
                final_message = messages[-1]

                if hasattr(final_message, "content"):
                    response = final_message.content
                else:
                    response = str(final_message)
            else:
                response = str(result)

            print(f"Agent response: {response}")

            results.append(
                {
                    "task": task,
                    "response": response,
                    "status": "success",
                }
            )

        except Exception as exc:
            print(f"Agent execution error: {exc}")

            results.append(
                {
                    "task": task,
                    "response": str(exc),
                    "status": "error",
                }
            )

    successful_tasks = sum(
        1 for result in results if result["status"] == "success"
    )

    print("\nAgent verification:")
    print("Tools configured: web search stub + calculator")
    print(f"Tasks executed: {len(tasks)}")
    print(f"Successful tasks: {successful_tasks}")

    return results


# ============================================================
# MANUAL VERIFICATION
# ============================================================

def main():

    print("W6D1: LangChain Fundamentals - Chains & Prompts")
    print(f"Local Ollama model: {MODEL}")

    chain_results = run_chain_tests()

    memory_results = run_memory_tests()

    agent_results = run_agent_tests()

    print("\n" + "=" * 70)
    print("MANUAL VERIFICATION")
    print("=" * 70)

    print(f"1. Chain tested with {len(chain_results)} inputs.")
    print("2. ConversationBufferMemory tested across 5 turns.")
    print(f"3. Conversation history contains {len(memory_results)} messages.")
    print(f"4. Agent tested with {len(agent_results)} tasks.")
    print("5. Agent configured with two tools: web-search stub and calculator.")

    successful_agent_tasks = sum(
        1 for result in agent_results if result["status"] == "success"
    )

    if (
        len(chain_results) == 5
        and len(memory_results) == 10
        and len(agent_results) == 3
        and successful_agent_tasks == 3
    ):
        print("6. W6D1 LangChain fundamentals test completed successfully.")
    else:
        print("6. W6D1 test completed with issues that need review.")


if __name__ == "__main__":
    main()