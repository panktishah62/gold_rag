import streamlit as st
from app.graph import build_graph

st.set_page_config(page_title="Gold RAG Assistant", layout="centered")

st.title("💰 Gold Market & Manufacturing Assistant")

# Build agent
@st.cache_resource
def load_graph():
    return build_graph()

graph = load_graph()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input
user_input = st.chat_input("Ask about gold pricing, manufacturing, exports...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = graph.invoke({"question": user_input})
                answer = result.get("answer", "No answer returned.")
                confidence = result.get("confidence")

                if confidence is not None:
                    answer += f"\n\n**Confidence:** {confidence}"

            except Exception as e:
                answer = f"Error: {str(e)}"

            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})