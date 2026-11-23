import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

# Step 1: Load variables from the .env file
load_dotenv()

# Step 2: Read the API key securely from the environment
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("Gemini API key not found!")

# Step 3: Initialize Gemini client (30 second timeout per request)
client = genai.Client(
    api_key=api_key,
    http_options=types.HttpOptions(timeout=30000),
)

# Step 4: Make first LLM call (retry only on temporary server errors)
MODEL = "gemini-3.5-flash-lite"
MAX_ATTEMPTS = 6
PROMPT = "What is Artificial Intelligence? Explain in simple words."

response = None

for attempt in range(MAX_ATTEMPTS):
    try:
        print(f"Trying {MODEL} (attempt {attempt + 1}/{MAX_ATTEMPTS})...")
        response = client.models.generate_content(model=MODEL, contents=PROMPT)
        break
    except errors.ServerError as e:
        wait = 2 ** attempt  # 1s, 2s, 4s, 8s, 16s, 32s
        print(f"Server busy ({e.code}), retrying in {wait}s...")
        time.sleep(wait)
    except errors.ClientError as e:
        # 4xx errors won't fix themselves, so show the real message and stop
        raise SystemExit(f"Request rejected: {e.code} {e.message}")

if response is None:
    raise RuntimeError("Gemini is still overloaded. Try again in a few minutes.")

# Step 5: Display response
print("\nLLM Response:")
print(response.text)