from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.config.settings import get_settings

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check():
    settings = get_settings()
    return {
        "status": "ok",
        "app": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
    }

@router.get("/health/db")
async def health_check_db(db: AsyncSession = Depends(get_db)):
    try:
        result = await db.execute(text("SELECT 1"))
        row = result.scalar()

        if row == 1:
            return {"status": "ok", "db": True}
        else:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "status": "unhealthy",
                    "db": False,
                    "error": "Database health check failed",
                },
                
            )

    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "db": False,
                "error": str(e),
            },
        )

# Health checks usually should not require normal user authentication, because infrastructure like Kubernetes/load balancers need to call them automatically. But detailed health checks should not expose secrets.
# Liveness probe asks: is the process alive? Readiness probe asks: is the app ready to receive traffic? Database health belongs more to readiness, because if DB is down, the app may be alive but not ready.