from redis import Redis
from rq import Queue
from .config import settings

redis_conn = Redis.from_url(settings.redis_url) if settings.redis_url else None
queue = Queue("vision", connection=redis_conn) if redis_conn else None
