from pydantic import BaseModel


class CacheStatsResponse(BaseModel):
    hit_count: int
    miss_count: int
    total_count: int
    hit_rate: float
