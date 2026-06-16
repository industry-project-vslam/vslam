# Mapping Service — Examples

Practical usage examples using `curl`. All requests assume the service is running at `http://localhost:8000`.

---

## 1. Health check

Verify the service is up before sending any requests.

```bash
curl http://localhost:8000/api/health
```

```json
{
  "status": "healthy",
  "timestamp": "2023-10-27T10:00:00.000000",
  "service": "Multi-Drone SLAM Tracker"
}
```

---

## 2. Run reconstruction from a server-side folder

The most common workflow. Frames are already on the server (e.g. mounted storage or a previous transfer).

```bash
curl -X POST http://localhost:8000/api/process_source \
  -H "Content-Type: application/json" \
  -d '{
    "drone_id": "drone001",
    "source_path": "/data/flights/mission_01/frames",
    "device": "cuda"
  }'
```

```json
{
  "drone_id": "drone001",
  "save_dir": "/results/drone001",
  "frame_count": 100,
  "last_frame": 99,
  "position": [0.1, 0.2, 0.3],
  "forward": [0.0, 0.0, 1.0],
  "valid": true,
  "skipped_frames": [],
  "message": "Reconstruction complete for drone drone001"
}
```

---

## 3. Run reconstruction with tuned parameters

Increase the localisation window and lower the L2W confidence threshold for challenging scenes with sparse visual features.

```bash
curl -X POST http://localhost:8000/api/process_source \
  -H "Content-Type: application/json" \
  -d '{
    "drone_id": "drone001",
    "source_path": "/data/flights/mission_02/frames",
    "win_r": 5,
    "conf_thres_l2w": 8.0,
    "conf_thres_i2p": 1.2,
    "num_scene_frame": 15,
    "frame_timeout_secs": 90.0,
    "device": "cuda"
  }'
```

---

## 4. Upload frames from a client

Use when frames are generated or captured on the client machine rather than the server.

```bash
curl -X POST http://localhost:8000/api/upload_frames/drone001 \
  -F "files=@/local/frames/frame_0001.jpg" \
  -F "files=@/local/frames/frame_0002.jpg" \
  -F "files=@/local/frames/frame_0003.jpg" \
  -F "device=cuda"
```

To upload all frames from a directory:

```bash
curl -X POST http://localhost:8000/api/upload_frames/drone001 \
  $(ls /local/frames/*.jpg | xargs -I{} echo "-F files=@{}") \
  -F "device=cuda"
```

---

## 5. Multi-drone stitching

Assign a shared `map_id` to register multiple drones into the same global map.

```bash
# Drone A
curl -X POST http://localhost:8000/api/process_source \
  -H "Content-Type: application/json" \
  -d '{
    "drone_id": "drone001",
    "source_path": "/data/mission/sector_a",
    "map_id": "mission_2023_10_27",
    "device": "cuda"
  }'

# Drone B
curl -X POST http://localhost:8000/api/process_source \
  -H "Content-Type: application/json" \
  -d '{
    "drone_id": "drone002",
    "source_path": "/data/mission/sector_b",
    "map_id": "mission_2023_10_27",
    "device": "cuda"
  }'
```

---

## 6. Set a per-frame timeout

Useful when processing time is variable and you want to bound total job duration.

```bash
# Set a 45-second timeout
curl -X POST http://localhost:8000/api/set_frame_timeout/drone001 \
  -H "Content-Type: application/json" \
  -d '{"timeout_secs": 45.0}'

# Disable the timeout
curl -X POST http://localhost:8000/api/set_frame_timeout/drone001 \
  -H "Content-Type: application/json" \
  -d '{"timeout_secs": null}'
```

---

## 7. Check skipped frames after reconstruction

After a job completes, verify whether any frames were dropped.

```bash
curl http://localhost:8000/api/skipped_frames/drone001
```

```json
{
  "drone_id": "drone001",
  "skipped_frames": [10, 15, 22]
}
```

If `skipped_frames` is non-empty, consider re-running with a higher `frame_timeout_secs` or switching to `device: "cpu"` for slower but more reliable processing.

---

## 8. Cancel a running job

```bash
curl -X POST http://localhost:8000/api/cancel/drone001
```

Poll `/api/drones/drone001` until `processing_status` returns `"idle"` to confirm the job has fully stopped.

---

## 9. Download the point cloud

```bash
curl -o drone001_map.ply http://localhost:8000/api/pointcloud/drone001
```

The `.ply` file can be opened in tools such as [CloudCompare](https://www.cloudcompare.org/) or [Open3D](http://www.open3d.org/).

---

## 10. Retrieve camera trajectories

```bash
curl http://localhost:8000/api/pointcloud/drone001/trajectories
```

Returns a JSON object containing per-frame camera positions and orientations. Useful for validating localisation quality or visualising the flight path.

---

## 11. Remove a drone from the registry

Clears the drone's entry without deleting output files on disk.

```bash
curl -X DELETE http://localhost:8000/api/drones/drone001
```