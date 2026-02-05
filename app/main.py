from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
import os
import uuid

from tts import generate_tts_mp3

app = FastAPI()

AUDIO_DIR = os.getenv("AUDIO_DIR", "./storage/audio")
os.makedirs(AUDIO_DIR, exist_ok=True)

app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")


@app.get("/")
def root():
    return {"status": "TradeTalkerAI API running"}


@app.post("/v1/analyze")
async def analyze(
    image: UploadFile = File(...),
    mode: str = Form("brief"),
):
    # For now we use a fake transcript just to test TTS
    transcript = "Trade Talker A I is running. This is your test audio."

    analysis_id = f"test_{uuid.uuid4().hex[:8]}"

    mp3_path, audio_url = await generate_tts_mp3(
        transcript=transcript,
        analysis_id=analysis_id,
        out_dir=AUDIO_DIR,
    )

    return {
        "transcript": transcript,
        "audio_url": audio_url
    }
