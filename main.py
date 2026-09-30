import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found. Check your .env file.")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=api_key,
    temperature=0.3,
)

# The system message tells Gemini who it is and how to behave
history = [
    SystemMessage(content=(
        "You are a friendly student assistant. Help with studying, "
        "explaining concepts simply, planning assignments, and staying "
        "organized. Keep answers clear and concise."
    ))
]

print("Student Assistant is ready! Type 'quit' to exit.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ("quit", "exit"):
        print("Goodbye, good luck with your studies!")
        break

    history.append(HumanMessage(content=user_input))
    response = llm.invoke(history)
    history.append(AIMessage(content=response.content))

    print(f"\nAssistant: {response.content}\n")