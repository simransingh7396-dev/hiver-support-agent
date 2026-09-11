import os
from dotenv import load_dotenv
from google import genai

# Load the API key from .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY not found. Make sure you created a .env file "
        "with a line like: GEMINI_API_KEY=your_key_here"
    )

client = genai.Client(api_key=api_key)

response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents="Say hello in one short sentence, and confirm you're working correctly."
)

print("SUCCESS! Gemini responded:")
print(response.text)