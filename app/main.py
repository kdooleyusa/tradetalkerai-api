from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import os

app = FastAPI()

AUDIO_DIR = os.getenv("AUDIO_DIR", "./storage/audio")

# Make sure audio folder exists
os.makedirs(AUDIO_DIR, exist_ok=True)

app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")

@app.get("/")
def root():
    return {"status": "TradeTalkerAI API running"}

@app.get("/health")
def health():
    return {"ok": True}
