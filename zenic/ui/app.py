"""
Zenic — Streamlit UI
Deployment: Hugging Face Spaces (Docker runtime, 16 GiB RAM)
"""
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from zenic.agent.graph import app as zenic_app
from zenic.agent.state import ZenicState

st.set_page_config(page_title="Zenic", page_icon="💪", layout="centered")
st.title("Zenic — Your AI Health & Nutrition Guide")

# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_profile" not in st.session_state:
    st.session_state.user_profile = {}
if "pdf_path" not in st.session_state:
    st.session_state.pdf_path = None
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

# ---------------------------------------------------------------------------
# Sidebar — profile display + PDF download
# ---------------------------------------------------------------------------
with st.sidebar:
    st.subheader("Your Profile")
    profile = st.session_state.user_profile
    if profile:
        field_labels = {
            "age": "Age",
            "weight_kg": "Weight (kg)",
            "height_cm": "Height (cm)",
            "gender": "Gender",
            "activity_level": "Activity level",
            "goal": "Goal",
            "dietary_restrictions": "Diet",
            "experience_level": "Experience",
            "available_days": "Days / week",
            "equipment": "Equipment",
        }
        for key, label in field_labels.items():
            val = profile.get(key)
            if val is not None and val != "":
                st.markdown(f"**{label}:** {val}")
    else:
        st.caption("Profile builds as you chat — Zenic will ask for the details it needs.")

    st.divider()

    if st.session_state.pdf_path and os.path.exists(st.session_state.pdf_path):
        st.subheader("Your Plan")
        with open(st.session_state.pdf_path, "rb") as f:
            st.download_button(
                "Download PDF",
                f,
                file_name="zenic_plan.pdf",
                mime="application/pdf",
            )

# ---------------------------------------------------------------------------
# Chat history display
# ---------------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ---------------------------------------------------------------------------
# Sample prompt buttons — shown only when chat is empty
# ---------------------------------------------------------------------------
SAMPLE_PROMPTS = [
    "What does the ISSN recommend for protein intake for athletes?",
    "Calculate my TDEE — I'm 28, 75 kg, 178 cm, moderately active",
    "Give me a 7-day meal plan for muscle gain",
    "Best barbell exercises for back hypertrophy?",
    "How has my calorie intake trended this week?",
]

if not st.session_state.messages:
    st.markdown("**Try one of these to get started:**")
    col1, col2 = st.columns(2)
    for i, prompt_text in enumerate(SAMPLE_PROMPTS):
        target_col = col1 if i % 2 == 0 else col2
        if target_col.button(prompt_text, key=f"sample_{i}", use_container_width=True):
            st.session_state.pending_prompt = prompt_text
            st.rerun()

# ---------------------------------------------------------------------------
# Chat input + message processing
# ---------------------------------------------------------------------------
typed_prompt = st.chat_input("Ask about nutrition, get a meal plan, workout split, or weekly summary...")

# Pick up either a typed message or one injected from a sample button click
prompt = typed_prompt or st.session_state.pending_prompt
if st.session_state.pending_prompt:
    st.session_state.pending_prompt = None

if prompt:
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

    # Persist profile and PDF path from final state
    if final_state.get("user_profile"):
        st.session_state.user_profile = final_state["user_profile"]
    pdf_path = (final_state.get("tool_results") or {}).get("pdf_path")
    if pdf_path:
        st.session_state.pdf_path = pdf_path
        st.rerun()
