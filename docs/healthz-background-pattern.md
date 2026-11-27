# Healthz & License Check Pattern

All Vexarr products (Lynxx, BuildX, HostX) must follow this pattern for health and license endpoints.

## Problem

Health endpoints (`/healthz`, `/status`) are called constantly by:
- Load balancers (every 5-10s per backend)
- Kubernetes liveness/readiness probes
- Monitoring systems (Prometheus, Grafana, uptime checks)
- Orchestrators and service mesh sidecars

If these endpoints do real work per request (DB queries, license API calls, hash computation), they become the bottleneck under load.

**Measured impact:** Lynxx gateway healthz with inline DB queries: **1,300 req/s**. After caching: **18,000+ req/s** — a 14x improvement.

## Architecture

```
                    ┌─────────────────────┐
                    │  Background Task     │
                    │  (runs every N sec)  │
                    │                      │
                    │  - DB queries        │
                    │  - License check     │
                    │  - Hash computation  │
                    │  - Metric collection │
                    └──────────┬───────────┘
                               │ write
                               ▼
                    ┌─────────────────────┐
                    │  Shared Cache        │
                    │  Arc<RwLock<State>>  │
                    └──────────┬───────────┘
                               │ read (O(1))
                               ▼
┌──────────┐       ┌─────────────────────┐
│  Client  │──────▶│  GET /healthz       │
│          │◀──────│  (returns cached)    │
└──────────┘       └─────────────────────┘
```

## Implementation (Rust)

### 1. Define the cache

```rust
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Clone)]
pub struct StatusCache {
    cached: Arc<RwLock<StatusResponse>>,
}

impl StatusCache {
    pub fn new(initial: StatusResponse) -> Self {
        Self { cached: Arc::new(RwLock::new(initial)) }
    }

    pub async fn get(&self) -> StatusResponse {
        self.cached.read().await.clone()
    }

    async fn update(&self, response: StatusResponse) {
        *self.cached.write().await = response;
    }
}
```

### 2. Background updater

```rust
pub fn spawn_status_updater(
    cache: StatusCache,
    repo: Arc<dyn Repository>,
    interval_ms: u64,
) {
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(
            Duration::from_millis(interval_ms)
        );
        ticker.set_missed_tick_behavior(MissedTickBehavior::Skip);

        loop {
            ticker.tick().await;
            let status = build_status(&repo).await;
            cache.update(status).await;
        }
    });
}
```

### 3. Endpoint handler

```rust
// BEFORE (slow — 2 DB queries per request)
pub async fn healthz(State(s): State<AppState>) -> Result<Json<Status>> {
    let plugins = s.repo.list_plugins().await?;
    let consumers = s.repo.list_consumers().await?;
    Ok(Json(Status { ... }))
}

// AFTER (fast — pure cache read)
pub async fn healthz(State(s): State<AppState>) -> Json<Status> {
    Json(s.status_cache.get().await)
}
```

### 4. Wire up at startup

```rust
let initial = build_status(&repo).await;
let cache = StatusCache::new(initial);
spawn_status_updater(cache.clone(), repo.clone(), 5_000);
// pass cache into your app state / router
```

## Implementation (Python / FastAPI)

```python
import asyncio
from datetime import datetime

class StatusCache:
    def __init__(self):
        self._status = {"status": "starting"}
        self._lock = asyncio.Lock()

    async def get(self) -> dict:
        return self._status

    async def update(self, status: dict):
        async with self._lock:
            self._status = status

async def status_updater(cache: StatusCache, db, interval: float = 5.0):
    while True:
        status = await build_status(db)
        await cache.update(status)
        await asyncio.sleep(interval)

# At startup:
cache = StatusCache()
asyncio.create_task(status_updater(cache, db))

@app.get("/healthz")
async def healthz():
    return await cache.get()
```

## Rules

1. `/healthz` must NEVER make DB calls, network requests, or compute hashes
2. `/healthz` must be infallible — return last known state even if background task fails
3. Background task interval: **5 seconds** default, configurable via config
4. License checks: cached in background, refreshed every 60-300s
5. `StatusResponse` must include `updated_at` so callers know cache freshness
6. Use `MissedTickBehavior::Skip` (Rust) or equivalent — don't queue up if a tick is slow
7. On DB query failure in background task, keep the last successful state (don't overwrite with error)

## Checklist for New Services

- [ ] Health endpoint returns cached state only
- [ ] Background task refreshes status at configurable interval
- [ ] License validation runs in background, not per-request
- [ ] Cache is populated with initial state before server starts accepting traffic
- [ ] Graceful degradation: stale cache is better than error response
