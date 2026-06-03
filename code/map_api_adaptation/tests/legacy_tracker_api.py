"""
FastAPI server for multi-drone SLAM tracking.

Endpoints:
    POST /api/process_frame - Process frame from drone
    GET /api/pointcloud/{drone_id} - Get point cloud for drone
    DELETE /api/drones/{drone_id} - Clear drone's map
    GET /api/status - Get status of all drone maps
    GET /api/health - Health check

Usage:
    uvicorn tracker_api:app --host 0.0.0.0 --port 8000
    
API Examples:
    # Process frame
    curl -X POST http://localhost:8000/api/process_frame \
        -H "Content-Type: application/json" \
        -d '{"drone_id": "drone_1", "frame_idx": 0, ...}'
    
    # Get pointcloud
    curl http://localhost:8000/api/pointcloud/drone_1
    
    # Clear drone data
    curl -X DELETE http://localhost:8000/api/drones/drone_1
    
    # Get status
    curl http://localhost:8000/api/status
"""

from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
import numpy as np
import json
import io
from datetime import datetime

from services.tracker_service import TrackerService, DepthMatrixGenerator

# Initialize FastAPI app
app = FastAPI(
    title="Multi-Drone SLAM Tracker API",
    description="Real-time multi-drone SLAM tracking with per-drone map management",
    version="1.0.0"
)

# Initialize tracker service
tracker_service = TrackerService(
    stale_threshold_hours=6,
    cleanup_interval=300  # 5 minutes
)


# ============================================================================
# Pydantic Models for API requests/responses
# ============================================================================

class FrameData(BaseModel):
    """Frame data submission."""
    drone_id: str
    frame_idx: int
    registered_pcd: List[List[List[float]]]  # (H, W, 3)
    registered_conf: List[List[float]]  # (H, W)
    rgb_img: List[List[List[float]]]  # (H, W, 3)


class FrameResponse(BaseModel):
    """Response from frame processing."""
    drone_id: str
    frame_idx: int
    location: List[float]  # [x, y, z]
    depth_matrix: List[List[float]]  # 8x8 matrix
    frame_count: int
    timestamp: str


class PointCloudResponse(BaseModel):
    """Point cloud response."""
    drone_id: str
    point_count: int
    has_colors: bool
    timestamp: str


class DroneStatus(BaseModel):
    """Status of a single drone."""
    drone_id: str
    created_at: str
    last_updated: str
    frame_count: int
    max_points: int
    is_stale: bool
    location: Optional[List[float]]


class StatusResponse(BaseModel):
    """Overall system status."""
    timestamp: str
    total_drones: int
    drones: Dict[str, dict]


class ClearResponse(BaseModel):
    """Response from clearing drone data."""
    drone_id: str
    success: bool
    message: str


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: str
    timestamp: str


# ============================================================================
# Health & Status Endpoints
# ============================================================================

@app.get("/api/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "Multi-Drone SLAM Tracker"
    }


@app.get("/api/status", response_model=StatusResponse)
def get_status():
    """Get status of all active drone maps."""
    status = tracker_service.get_status()
    return StatusResponse(**status)


# ============================================================================
# Frame Processing Endpoint
# ============================================================================

@app.post("/api/process_frame", response_model=FrameResponse)
def process_frame(frame_data: FrameData):
    """
    Process a frame from a drone and return drone location + depth matrix.
    
    Args:
        frame_data: Frame data including drone_id, frame_idx, point cloud, confidence, and RGB image
        
    Returns:
        Drone location [x, y, z] and 8x8 depth matrix
    """
    try:
        # Convert lists to numpy arrays
        registered_pcd = np.array(frame_data.registered_pcd, dtype=np.float32)
        registered_conf = np.array(frame_data.registered_conf, dtype=np.float32)
        rgb_img = np.array(frame_data.rgb_img, dtype=np.float32)
        
        # Validate shapes
        if registered_pcd.shape[:2] != registered_conf.shape:
            raise ValueError("Point cloud and confidence map dimensions must match")
        if registered_pcd.shape[:2] != rgb_img.shape[:2]:
            raise ValueError("Point cloud and RGB image dimensions must match")
        
        # Process frame through tracker service
        result = tracker_service.process_frame(
            drone_id=frame_data.drone_id,
            frame_idx=frame_data.frame_idx,
            registered_pcd=registered_pcd,
            registered_conf=registered_conf,
            rgb_img=rgb_img
        )
        
        return FrameResponse(**result)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing frame: {str(e)}")


# ============================================================================
# Point Cloud Endpoint
# ============================================================================

@app.get("/api/pointcloud/{drone_id}")
def get_pointcloud(drone_id: str, conf_threshold: float = 12.0, format: str = "npy"):
    """
    Get point cloud for a drone's map.
    
    Args:
        drone_id: Unique drone identifier
        conf_threshold: Confidence threshold for filtering points
        format: Output format - 'npy' or 'ply'
        
    Returns:
        Point cloud in requested format (binary)
    """
    try:
        points, colors = tracker_service.get_pointcloud(drone_id, conf_threshold)
        
        if points is None:
            raise HTTPException(status_code=404, detail=f"No map found for drone {drone_id}")
        
        if format.lower() == "npy":
            return _return_pointcloud_npy(points, colors, drone_id)
        elif format.lower() == "ply":
            return _return_pointcloud_ply(points, colors, drone_id)
        else:
            raise HTTPException(status_code=400, detail="Format must be 'npy' or 'ply'")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving point cloud: {str(e)}")


def _return_pointcloud_npy(points: np.ndarray, colors: np.ndarray, drone_id: str):
    """Return point cloud as numpy arrays."""
    buffer = io.BytesIO()
    np.savez_compressed(buffer, points=points, colors=colors)
    buffer.seek(0)
    
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename=pointcloud_{drone_id}.npz"}
    )


def _return_pointcloud_ply(points: np.ndarray, colors: np.ndarray, drone_id: str):
    """Return point cloud as PLY file."""
    try:
        import open3d as o3d
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Ensure colors are in [0, 1] range
        if colors.max() > 1.0:
            colors = colors / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        buffer = io.BytesIO()
        o3d.io.write_point_cloud(buffer.name if hasattr(buffer, 'name') else "temp.ply", pcd)
        
        # Since open3d writes to file, we need to read it back
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp:
            o3d.io.write_point_cloud(tmp.name, pcd)
            with open(tmp.name, 'rb') as f:
                ply_data = f.read()
        
        return StreamingResponse(
            iter([ply_data]),
            media_type="model/vnd.ply",
            headers={"Content-Disposition": f"attachment; filename=pointcloud_{drone_id}.ply"}
        )
    except ImportError:
        raise HTTPException(status_code=500, detail="PLY format requires open3d. Use 'npy' instead.")


# ============================================================================
# Drone Management Endpoints
# ============================================================================

@app.delete("/api/drones/{drone_id}", response_model=ClearResponse)
def clear_drone(drone_id: str):
    """
    Clear all data for a drone completely.
    
    Args:
        drone_id: Unique drone identifier
        
    Returns:
        Confirmation of clearing
    """
    try:
        success = tracker_service.clear_drone_map(drone_id)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"No map found for drone {drone_id}")
        
        return ClearResponse(
            drone_id=drone_id,
            success=True,
            message=f"Successfully cleared all data for drone {drone_id}"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing drone data: {str(e)}")


@app.get("/api/drones/{drone_id}")
def get_drone_status(drone_id: str):
    """Get status of a specific drone."""
    try:
        status = tracker_service.get_status()
        
        if drone_id not in status['drones']:
            raise HTTPException(status_code=404, detail=f"No map found for drone {drone_id}")
        
        return {
            "drone_id": drone_id,
            "status": status['drones'][drone_id],
            "timestamp": status['timestamp']
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving drone status: {str(e)}")


# ============================================================================
# Depth Matrix Endpoint (utility)
# ============================================================================

@app.post("/api/depth_matrix")
def compute_depth_matrix(frame_data: FrameData):
    """
    Compute 8x8 depth matrix for a frame without storing it.
    
    Useful for testing or one-off depth matrix generation.
    
    Args:
        frame_data: Frame data
        
    Returns:
        8x8 depth matrix and metadata
    """
    try:
        registered_pcd = np.array(frame_data.registered_pcd, dtype=np.float32)
        depth_matrix = DepthMatrixGenerator.generate_from_pcd(registered_pcd)
        
        return {
            "depth_matrix": depth_matrix.tolist(),
            "shape": depth_matrix.shape,
            "min": float(np.min(depth_matrix)),
            "max": float(np.max(depth_matrix)),
            "mean": float(np.mean(depth_matrix[depth_matrix > 0]))
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error computing depth matrix: {str(e)}")


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP Error",
            "detail": exc.detail,
            "timestamp": datetime.now().isoformat()
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler."""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        },
    )


# ============================================================================
# Info Endpoint
# ============================================================================

@app.get("/api/info")
def get_api_info():
    """Get API information and available endpoints."""
    return {
        "name": "Multi-Drone SLAM Tracker API",
        "version": "1.0.0",
        "endpoints": {
            "POST /api/process_frame": "Process frame from drone",
            "GET /api/pointcloud/{drone_id}": "Get point cloud for drone",
            "GET /api/drones/{drone_id}": "Get drone status",
            "DELETE /api/drones/{drone_id}": "Clear drone's data",
            "GET /api/status": "Get all drones status",
            "POST /api/depth_matrix": "Compute depth matrix (utility)",
            "GET /api/health": "Health check",
            "GET /api/info": "This endpoint"
        },
        "features": {
            "multi_drone_support": True,
            "auto_cleanup_hours": 6,
            "depth_matrix_size": "8x8",
            "confidence_filtering": True,
            "point_cloud_formats": ["npy", "ply"]
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
