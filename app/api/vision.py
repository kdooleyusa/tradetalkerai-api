import hashlib, uuid
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from ..core.redisq import queue, redis_conn
from ..core.s3 import put_png_bytes
from ..vision.preprocess import preprocess_to_png_bytes
from ..vision.schema import Mode

router = APIRouter(prefix="/vision", tags=["vision"])

class UploadResp(BaseModel):
    image_id: str
    sha256: str

class JobResp(BaseModel):
    job_id: str

@router.post("/upload", response_model=UploadResp)
async def upload_chart(file: UploadFile = File(...)):
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "Empty upload")

    sha = hashlib.sha256(raw).hexdigest()
    image_id = uuid.uuid4().hex

    png = preprocess_to_png_bytes(raw)
    key = put_png_bytes(png, prefix="charts/")

    # store mapping in redis
    if not redis_conn:
        raise HTTPException(500, "REDIS_URL not configured")
    redis_conn.hset(f"img:{image_id}", mapping={"sha256": sha, "s3_key": key})

    return UploadResp(image_id=image_id, sha256=sha)

@router.post("/analyze/{image_id}", response_model=JobResp)
def analyze(image_id: str, mode: Mode = "f8"):
    if not queue or not redis_conn:
        raise HTTPException(500, "Redis/Queue not configured")

    meta = redis_conn.hgetall(f"img:{image_id}")
    if not meta:
        raise HTTPException(404, "Unknown image_id")

    # enqueue worker job
    job = queue.enqueue("app.worker.run_vision_job", image_id, mode)
    return JobResp(job_id=job.id)

@router.get("/result/{job_id}")
def result(job_id: str):
    if not redis_conn:
        raise HTTPException(500, "Redis not configured")

    key = f"job:{job_id}"
    data = redis_conn.get(key)
    if not data:
        return {"status": "running"}
    return {"status": "done", "result": data.decode("utf-8")}
