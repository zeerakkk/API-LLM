# Import required libraries 
# FastAPI: web framework for building APIs
# Depends, Header, HTTPException: used for dependency injection and error handling
# ollama: client library to interact with local Ollama LLM models
# os and dotenv: for environment variable management
from fastapi import FastAPI, Depends, HTTPException, Header
import ollama
import os
from dotenv import load_dotenv


# Load environment variables from .env file 
# This allows the API key to be securely stored instead of hard-coded
load_dotenv()


# Initialize API key and usage credits 
# The .env file should contain a line like: API_KEY=secretkey
API_KEY_CREDITS = {os.getenv("API_KEY"): 5}
print(API_KEY_CREDITS)  # For debugging purposes — displays loaded API key and credits


# Initialize FastAPI application 
# This creates an instance of the FastAPI app, which will handle routes and requests
app = FastAPI()


# Every request to the /generate endpoint must include an 'x-api-key' header.
# The function checks if the key exists and still has credits available.
# If not, it raises an HTTP 401 Unauthorized error.
def verify_api_key(x_api_key: str = Header(None)):
    credits = API_KEY_CREDITS.get(x_api_key, 0)
    if credits <= 0:
        raise HTTPException(status_code=401, detail="Invalid API Key, or no credits")

    return x_api_key


# Main Endpoint: /generate 
# Method: POST
# Description:
#   - Receives a text prompt from the user.
#   - Verifies the API key via the dependency function.
#   - Sends the prompt to a locally hosted Ollama model ("gemma:2b").
#   - Returns the model’s generated text response.
# The user's available credits are reduced by 1 for each request.
@app.post("/generate")
def generate(prompt: str, x_api_key: str = Depends(verify_api_key)):
    # Deduct one credit from the user's balance
    API_KEY_CREDITS[x_api_key] -= 1

    # Call the Ollama chat endpoint to generate a model response
    response = ollama.chat(
        model="gemma:2b",
        messages=[{"role": "user", "content": prompt}]
    )

    # Return only the generated text content as a JSON response
    return {"response": response["message"]["content"]}


