"""
Zenic — Streamlit UI
Deployment: Streamlit Community Cloud (1 GiB RAM limit — keep models lean)
"""
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from zenic.agent.graph import app as zenic_app
from zenic.agent.state import ZenicState

st.set_page_config(page_title="Zenic", page_icon="💪", layout="centered")
st.title("Zenic — Your AI Health & Nutrition Guide")

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_profile" not in st.session_state:
    st.session_state.user_profile = {}
if "pdf_path" not in st.session_state:
    st.session_state.pdf_path = None

# Chat history display
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# PDF download sidebar
if st.session_state.pdf_path and os.path.exists(st.session_state.pdf_path):
    with st.sidebar:
        st.subheader("Your Plan")
        with open(st.session_state.pdf_path, "rb") as f:
            st.download_button("Download PDF", f, file_name="zenic_plan.pdf", mime="application/pdf")

# Chat input
if prompt := st.chat_input("Ask about nutrition, get a meal plan, workout split, or weekly summary..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    initial_state: ZenicState = {
        "messages": [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
        "user_profile": st.session_state.user_profile,
        "intent": "",
        "profile_complete": False,
        "missing_fields": [],
        "awaiting_input": False,
        "retrieved_context": [],
        "tool_results": {},
        "plan_data": {},
        "safety_flag": False,
        "safety_reason": "",
    }

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            final_state = zenic_app.invoke(initial_state)

        # Extract last assistant message — state messages are LangChain objects after invoke
        assistant_msgs = [m for m in final_state.get("messages", []) if getattr(m, "type", None) == "ai"]
        reply = assistant_msgs[-1].content if assistant_msgs else "Sorry, I couldn't generate a response."
        st.write(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})

    # Update profile and PDF path from state
    if final_state.get("user_profile"):
        st.session_state.user_profile = final_state["user_profile"]
    pdf_path = (final_state.get("tool_results") or {}).get("pdf_path")
    if pdf_path:
        st.session_state.pdf_path = pdf_path
        st.rerun()
