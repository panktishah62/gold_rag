import streamlit as st
import requests

# Backend URL
BACKEND_URL = "http://localhost:8000/chat"

st.set_page_config(page_title="Gold RAG Assistant", layout="centered")

st.title("💰 Gold Market & Manufacturing Assistant")

# Store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_input = st.chat_input("Ask about gold pricing, manufacturing, exports...")

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Call backend
    try:
        response = requests.post(
            BACKEND_URL,
            json={"question": user_input},
            timeout=60,
        )

        if response.status_code == 200:
            payload = response.json()
            answer = payload.get("answer", "No answer returned.")
            confidence = payload.get("confidence", None)
            if confidence is not None:
                answer = f"{answer}\n\n**Confidence:** {confidence}"
        else:
            answer = f"Backend error: {response.status_code}"

    except Exception as e:
        answer = f"Connection error: {str(e)}"

    # Show assistant response
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)

