import os
from pydantic import BaseModel

class Settings(BaseModel):
    redis_url: str = os.getenv("REDIS_URL", "")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    vision_model: str = os.getenv("VISION_MODEL", "gpt-4o-mini")

    s3_endpoint: str = os.getenv("S3_ENDPOINT", "")
    s3_bucket: str = os.getenv("S3_BUCKET", "")
    s3_access_key: str = os.getenv("S3_ACCESS_KEY", "")
    s3_secret_key: str = os.getenv("S3_SECRET_KEY", "")
    s3_region: str = os.getenv("S3_REGION", "auto")

settings = Settings()
