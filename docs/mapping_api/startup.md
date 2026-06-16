# Mapping Service — Startup

## Prerequisites

- Docker and Docker Compose installed and running.

---

## Starting the Service

```bash
cd ./code/map_api_adaptation
docker compose up
```

The service exposes the API on **`http://localhost:8000`**.

Startup may take a minute or two while the container initialises. Wait until you see the service ready log output before sending requests.

---

## Verifying the Service is Ready

```bash
curl http://localhost:8000/api/health
```

Expected response:

```json
{
  "status": "healthy",
  "timestamp": "2023-10-27T10:00:00.000000",
  "service": "Multi-Drone SLAM Tracker"
}
```

If the request fails or times out, the container is still starting. Retry after a few seconds.

---

## Stopping the Service

```bash
docker compose down
```

Output files written to disk during a session are retained after shutdown.