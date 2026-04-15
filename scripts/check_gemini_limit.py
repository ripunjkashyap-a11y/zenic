import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

def generate():
    # Use GOOGLE_API_KEY as the env variable since that's what's in our .env
    client = genai.Client(
        api_key=os.environ.get("GOOGLE_API_KEY"),
    )

    model = "gemma-4-31b-it"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="""What is the capital of France? Answer in one word."""),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig()

    print(f"Connecting to {model}...")
    try:
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            if text := chunk.text:
                print(text, end="")
        print("\n\nSuccess: Model is responding.")
    except Exception as e:
        print(f"\nERROR: {e}")

if __name__ == "__main__":
    generate()
