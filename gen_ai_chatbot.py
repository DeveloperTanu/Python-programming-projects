import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    API_KEY = input("Enter your Groq API key: ").strip()

llm = ChatGroq(
    api_key=API_KEY,
    model="llama-3.3-70b-versatile",
    temperature=0.7
)

messages = [
    SystemMessage(
        content="You are a helpful, intelligent and conversational AI assistant."
    )
]

print("\n" + "=" * 45)
print("          TERMINAL AI CHATBOT")
print("=" * 45)
print("Commands: exit | clear")
print("=" * 45)

while True:
    try:
        user_input = input("\nYou: ").strip()

        if not user_input:
            continue

        if user_input.lower() == "exit":
            print("\nAI: Goodbye!")
            break

        if user_input.lower() == "clear":
            messages = [
                SystemMessage(
                    content="You are a helpful, intelligent and conversational AI assistant."
                )
            ]
            print("\nAI: Conversation history cleared.")
            continue

        messages.append(HumanMessage(content=user_input))

        print("\nAI: ", end="", flush=True)

        response = llm.invoke(messages)

        messages.append(AIMessage(content=response.content))

        print(response.content)

    except KeyboardInterrupt:
        print("\n\nAI: Chatbot stopped.")
        break

    except Exception as error:
        if messages and isinstance(messages[-1], HumanMessage):
            messages.pop()

        print(f"\nError: {error}")
