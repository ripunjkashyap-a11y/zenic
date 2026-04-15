"""Probe which Gemini models are currently responding on the free tier."""
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI

CANDIDATES = [
    "gemini-flash-lite-latest",
    "gemini-2.0-flash-lite",
    "gemini-2.5-flash-preview-04-17",
    "gemini-2.0-flash",
    "gemini-1.5-flash-8b",
    "gemini-1.5-flash",
    "gemini-3.1-flash-lite-preview",
]

for model in CANDIDATES:
    try:
        llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=os.environ.get("GOOGLE_API_KEY"),
            temperature=0,
        )
        resp = llm.invoke("Say OK")
        print(f"OK   {model}: {resp.content[:60]}")
    except Exception as e:
        print(f"ERR  {model}: {type(e).__name__}: {str(e)[:100]}")
