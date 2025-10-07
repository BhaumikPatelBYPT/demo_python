# Day 13: Custom FastAPI Project & Review
# ----------------------------------------
# This file is for your custom API or data science project using FastAPI and/or Hugging Face.
# Add clear comments to explain each step and help others (and yourself) understand your code.
# ----------------------------------------
# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline

# 1️⃣ Initialize FastAPI app
app = FastAPI(title="Sentiment Analysis API")

# 2️⃣ Allow frontend requests (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # for all domains (you can restrict later)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3️⃣ Define request model using Pydantic
class TextRequest(BaseModel):
    text: str

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=True)

# 4️⃣ Load Hugging Face sentiment model
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)
# 5️⃣ Routes
@app.get("/")
def home():
    return {"message": "Welcome to Sentiment Analysis API"}

@app.post("/analyze")
def analyze_sentiment(request: TextRequest):
    """
    POST /analyze
    Input: {"text": "I love this movie!"}
    Output: {"label": "POSITIVE", "score": 0.99}
    """
    result = sentiment_analyzer(request.text)[0]
    return {"label": result["label"], "score": result["score"]}