"""FastAPI application entrypoint for the Multi-Drone SLAM Tracker API."""

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from api.dependencies import init_tracker_service
from api.routes import frames, drones, status, pointcloud

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and initialise the tracker service before serving requests."""
    logger.info("Loading SLAM3R models...")
    init_tracker_service()
    logger.info("TrackerService ready.")
    yield
    # Nothing to clean up — cleanup thread is daemonised and dies with the process.


app = FastAPI(
    title="Multi-Drone SLAM Tracker API",
    description="Real-time multi-drone SLAM tracking with per-drone map management",
    version="2.0.0",
    lifespan=lifespan,
)


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logging.exception("Unhandled exception while processing request")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error. See server logs for details."},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logging.warning("Validation error while processing request: %s", exc)
    return JSONResponse(
        status_code=422,
        content={"error": "Validation error", "detail": exc.errors()},
    )


app.include_router(frames.router)
app.include_router(pointcloud.router)
app.include_router(drones.router)
app.include_router(status.router)


@app.get("/", tags=["root"])
def root() -> dict:
    return {
        "message": "Multi-Drone SLAM Tracker API",
        "info": "Use /api/* endpoints for tracking, status, and point cloud retrieval.",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)