from agent import new_history, run_turn

history = new_history()

print("Student Assistant is ready! Type 'quit' to exit.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ("quit", "exit"):
        print("Goodbye, good luck with your studies!")
        break

    reply, tools_used = run_turn(history, user_input)
    for name in tools_used:
        print(f"  [using tool: {name}]")
    print(f"\nAssistant: {reply}\n")