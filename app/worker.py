import json
from rq import Worker
from .core.redisq import redis_conn, queue
from .vision.pipeline import analyze_image_key

def run_vision_job(image_id: str, mode: str):
    meta = redis_conn.hgetall(f"img:{image_id}")
    s3_key = meta.get(b"s3_key", b"").decode("utf-8")
    res = analyze_image_key(image_id=image_id, s3_key=s3_key, mode=mode)  # VisionResult

    # store result for polling
    payload = res.model_dump()
    redis_conn.set(f"job:{_current_job_id()}", json.dumps(payload), ex=3600)
    return payload

def _current_job_id() -> str:
    # RQ sets current_job in worker process context
    from rq import get_current_job
    job = get_current_job()
    return job.id if job else "unknown"

if __name__ == "__main__":
    if not redis_conn or not queue:
        raise RuntimeError("REDIS_URL not configured")
    Worker([queue], connection=redis_conn).work()
