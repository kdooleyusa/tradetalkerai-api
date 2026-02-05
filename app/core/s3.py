import io, uuid
import boto3
from botocore.client import Config
from .config import settings

def _client():
    return boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint,
        aws_access_key_id=settings.s3_access_key,
        aws_secret_access_key=settings.s3_secret_key,
        region_name=settings.s3_region,
        config=Config(signature_version="s3v4"),
    )

def put_png_bytes(data: bytes, prefix: str = "charts/") -> str:
    key = f"{prefix}{uuid.uuid4().hex}.png"
    _client().put_object(
        Bucket=settings.s3_bucket,
        Key=key,
        Body=data,
        ContentType="image/png",
    )
    return key

def presigned_get_url(key: str, expires_sec: int = 600) -> str:
    return _client().generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": settings.s3_bucket, "Key": key},
        ExpiresIn=expires_sec,
    )
