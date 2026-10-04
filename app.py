import streamlit as st
from agent import new_history, run_turn, load_deadlines

st.set_page_config(page_title="Student Assistant", page_icon="🎓")
st.title("🎓 Student Assistant")

# Streamlit re-runs this whole file on every interaction,
# so anything we want to keep must live in session_state.
if "history" not in st.session_state:
    st.session_state.history = new_history()   # what the model sees
    st.session_state.messages = []             # what the screen shows

# Handle new input first, so the sidebar shows fresh data
prompt = st.chat_input("Ask me anything, or tell me about a deadline...")
if prompt:
    with st.spinner("Thinking..."):
        reply, tools_used = run_turn(st.session_state.history, prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append(
        {"role": "assistant", "content": reply, "tools": tools_used}
    )

# Sidebar: live deadlines + reset button
with st.sidebar:
    st.header("📅 Your deadlines")
    deadlines = sorted(load_deadlines(), key=lambda d: d["due_date"])
    if deadlines:
        for d in deadlines:
            st.write(f"**{d['title']}**  \n{d['due_date']}")
    else:
        st.caption("No deadlines saved yet.")
    if st.button("Clear chat"):
        st.session_state.history = new_history()
        st.session_state.messages = []
        st.rerun()

# Draw the conversation
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("tools"):
            st.caption("🔧 Used: " + ", ".join(m["tools"]))