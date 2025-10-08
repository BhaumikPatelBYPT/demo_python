# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
import os

app = FastAPI(title="Sentiment API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lightweight model
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="sshleifer/tiny-distilbert-base-uncased-finetuned-sst-2-english"
)

@app.get("/")
def root():
    return {"message": "Sentiment API Running"}

@app.post("/sentiment")
async def analyze(data: dict):
    text = data.get("text", "")
    result = sentiment_analyzer(text)
    return {"sentiment": result[0]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
