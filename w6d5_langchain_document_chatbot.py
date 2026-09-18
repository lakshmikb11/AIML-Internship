from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM, ChatOllama
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.tools import tool
from langchain.agents import create_agent


MODEL = "llama3.2:3b"


def task1_chain():
    print("\n" + "=" * 60)
    print("W6D5 TASK 1 - LANGCHAIN CHAIN")
    print("=" * 60)

    prompt = PromptTemplate(
        input_variables=["question"],
        template=(
            "Answer the following machine learning question in a "
            "simple and beginner-friendly way:\n\n{question}"
        ),
    )

    llm = OllamaLLM(model=MODEL)
    parser = StrOutputParser()

    chain = prompt | llm | parser

    questions = [
        "What is supervised learning?",
        "What is overfitting?",
        "What is classification?",
        "What is regression?",
        "What is a machine learning model?",
    ]

    successful = 0

    for i, question in enumerate(questions, start=1):
        answer = chain.invoke({"question": question})

        print(f"\nInput {i}: {question}")
        print(f"Output: {answer}")

        assert isinstance(answer, str)
        assert answer.strip()
        successful += 1

    print(f"\n5 inputs tested successfully: {successful}/5")
    print("Task 1 status: SUCCESS")


def task2_memory():
    print("\n" + "=" * 60)
    print("W6D5 TASK 2 - CONVERSATION MEMORY")
    print("=" * 60)

    llm = OllamaLLM(model=MODEL)

    memory = ConversationBufferMemory(
        memory_key="history",
        return_messages=True,
    )

    prompt = PromptTemplate(
        input_variables=["history", "question"],
        template=(
            "You are a helpful AI tutor.\n"
            "Conversation history:\n{history}\n\n"
            "User question: {question}\n"
            "Answer clearly and briefly."
        ),
    )

    chain = prompt | llm | StrOutputParser()

    turns = [
        "My name is Lakshmi.",
        "I am learning machine learning.",
        "What is supervised learning?",
        "What is overfitting?",
        "What topic did I say I am learning?",
    ]

    for i, question in enumerate(turns, start=1):
        history_messages = memory.load_memory_variables({}).get(
            "history", []
        )

        history_text = "\n".join(
            f"{message.type}: {message.content}"
            for message in history_messages
        )

        answer = chain.invoke(
            {
                "history": history_text,
                "question": question,
            }
        )

        memory.save_context(
            {"input": question},
            {"output": answer},
        )

        print(f"\nTurn {i}")
        print(f"User: {question}")
        print(f"Assistant: {answer}")

    final_history = memory.load_memory_variables({})["history"]

    print("\nConversation history verification")
    print("-" * 60)
    print(f"Turns completed: {len(turns)}")
    print(f"Messages stored: {len(final_history)}")

    assert len(final_history) == 10

    history_text = " ".join(
        message.content for message in final_history
    ).lower()

    assert "lakshmi" in history_text
    assert "machine learning" in history_text

    print("Memory verification: SUCCESS")
    print("Task 2 status: SUCCESS")


@tool
def web_search_stub(query: str) -> str:
    """Return a simulated web search result for a query."""
    return (
        f"Web search stub result for '{query}': "
        "Machine learning is a field of AI that learns patterns from data."
    )


@tool
def calculator(expression: str) -> str:
    """Calculate a basic arithmetic expression."""
    try:
        allowed = set("0123456789+-*/(). ")
        if not all(char in allowed for char in expression):
            return "Invalid expression"

        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as exc:
        return f"Calculation error: {exc}"


def task3_agent():
    print("\n" + "=" * 60)
    print("W6D5 TASK 3 - TWO-TOOL AGENT")
    print("=" * 60)

    tools = [web_search_stub, calculator]

    agent = create_agent(
        model=ChatOllama(model=MODEL),
        tools=tools,
        system_prompt=(
            "You are a helpful assistant. "
            "Use the calculator tool for arithmetic calculations. "
            "Use the web search stub for web search requests. "
            "Give concise answers."
        ),
    )

    tasks = [
        (
            "Calculate 125 * 8 + 40 using the calculator tool.",
            "1040",
        ),
        (
            "Use the web search tool to find a short definition of machine learning.",
            "machine learning",
        ),
        (
            "Calculate (250 / 5) + 30 using the calculator tool.",
            "80",
        ),
    ]

    for i, (task, expected) in enumerate(tasks, start=1):
        result = agent.invoke(
            {
                "messages": [
                    {"role": "user", "content": task}
                ]
            }
        )

        messages = result.get("messages", [])
        final_message = messages[-1].content if messages else ""

        print(f"\nAgent Task {i}")
        print(f"Input: {task}")
        print(f"Output: {final_message}")

        assert final_message
        assert expected.lower() in final_message.lower()

    print("\n3 agent tasks tested successfully: 3/3")
    print("Tools used: web search stub + calculator")
    print("Task 3 status: SUCCESS")


def main():
    print("\n" + "#" * 60)
    print("# W6D5: WEEK 6 PROJECT - DOCUMENT CHATBOT WITH LANGCHAIN")
    print("#" * 60)

    task1_chain()
    task2_memory()
    task3_agent()

    print("\n" + "#" * 60)
    print("# W6D5 ALL TASKS COMPLETED SUCCESSFULLY")
    print("#" * 60)


if __name__ == "__main__":
    main()
