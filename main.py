import os
import json
from datetime import date
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from datetime import date, timedelta

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found. Check your .env file.")

DEADLINES_FILE = "deadlines.json"


def load_deadlines():
    if not os.path.exists(DEADLINES_FILE):
        return []
    with open(DEADLINES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------- TOOLS ----------
@tool
def save_deadline(title: str, due_date: str) -> str:
    """Save a deadline, exam, or assignment. due_date must be in YYYY-MM-DD format."""
    deadlines = load_deadlines()
    deadlines.append({"title": title, "due_date": due_date})
    with open(DEADLINES_FILE, "w", encoding="utf-8") as f:
        json.dump(deadlines, f, indent=2)
    return f"Saved: {title} on {due_date}"


@tool
def list_deadlines() -> str:
    """List all saved deadlines, exams, and assignments."""
    deadlines = load_deadlines()
    if not deadlines:
        return "No deadlines saved yet."
    return "\n".join(f"- {d['title']} (due {d['due_date']})" for d in deadlines)


tools = [save_deadline, list_deadlines]
tools_by_name = {t.name: t for t in tools}

# ---------- MODEL ----------
llm = ChatGroq(
        model="openai/gpt-oss-20b",
    api_key=api_key,
    temperature=0.3,
    max_retries=1,
)
llm_with_tools = llm.bind_tools(tools)

today = date.today()
calendar = "\n".join(
    f"{(today + timedelta(days=i)).strftime('%A')}: "
    f"{(today + timedelta(days=i)).isoformat()}"
    for i in range(15)
)

history = [
    SystemMessage(content=(
        "You are a friendly student assistant. Help with studying, explaining "
        "concepts simply, and staying organized. Use your tools to save and "
        "list the student's deadlines when appropriate.\n"
        f"Today is {today.strftime('%A')}, {today.isoformat()}.\n"
        f"Upcoming dates:\n{calendar}\n"
        "When the student mentions a day like 'Monday' or 'next Friday', look "
        "up the exact date in the calendar above. Never calculate dates yourself."
    ))
]

def get_text(message):
    """Return the reply as plain text, whatever format the model used."""
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(
        part.get("text", "") if isinstance(part, dict) else str(part)
        for part in content
    )


print("Student Assistant is ready! Type 'quit' to exit.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ("quit", "exit"):
        print("Goodbye, good luck with your studies!")
        break

    history.append(HumanMessage(content=user_input))

    # The agent loop: keep going until the model stops asking for tools
    while True:
        response = llm_with_tools.invoke(history)
        history.append(response)

        if not response.tool_calls:
            break

        for call in response.tool_calls:
            print(f"  [using tool: {call['name']}]")
            result = tools_by_name[call["name"]].invoke(call["args"])
            history.append(ToolMessage(content=result, tool_call_id=call["id"]))

    print(f"\nAssistant: {get_text(response)}\n")