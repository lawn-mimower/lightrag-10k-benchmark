import google.genai as genai
from dotenv import load_dotenv

GEMINI_API_KEY = load_dotenv("GEMINI_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY)

def test_gemini():
    try:
        print("Sending request to Gemini 3 Flash...")
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents="gimme a 5 line poem"
        )
        
        print("-" * 20)
        print(f"Response: {response.text}")
        print("-" * 20)
        print("API Key is working correctly.")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_gemini()