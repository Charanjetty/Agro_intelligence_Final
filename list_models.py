
from google import genai
import os

API_KEY = "AIzaSyDii357S9SM7hxYYe_YfHq1BxNq_IFUoQU"

def list_models():
    print("--- Listing Available Models ---")
    try:
        client = genai.Client(api_key=API_KEY)
        # Try to list models
        # Note: The new SDK might use a pager or different syntax. 
        # Checking if client.models.list() is iterable directly
        for m in client.models.list():
             print(f"Model: {m.name}")
    except Exception as e:
        print(f"List Error: {e}")

if __name__ == "__main__":
    list_models()
