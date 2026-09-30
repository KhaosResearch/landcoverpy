# Architecture: Systems, Concurrency & Storage Lifecycle

This document describes the underlying software architecture, distributed synchronization mechanisms, and storage optimization strategies in **LandCoverPy**.

---

## 1. System Topology

LandCoverPy separates computation from storage to support large-scale distributed deployments.

```mermaid
graph LR
    subgraph COMPUTE ["Compute Layer"]
        W1["Worker 1 (Node A)"]
        W2["Worker 2 (Node B / Helper)"]
    end

    subgraph METADATA ["Metadata & Locking (MongoDB)"]
        TC["tile_claims<br/>(Atomic Locks & TTL)"]
        P["products<br/>(Capture Footprints & Dates)"]
        C["composites<br/>(Band Catalogs)"]
    end

    subgraph STORAGE ["Object Storage (MinIO / S3)"]
        B_RAW["s2-products<br/>(Ephemeral Raw Granules)"]
        B_COMP["s2-composites<br/>(11 DEFLATE Bands)"]
        B_MODELS["models<br/>(Trained ML Weights)"]
        B_MAPS["classification-maps<br/>(Final GeoTIFFs)"]
    end

    W1 & W2 <-->|"Lock / Release"| TC
    W1 & W2 -->|"Catalog Queries"| P & C
    W1 & W2 -->|"Temporary Staging"| B_RAW
    W1 & W2 -->|"Store Composites"| B_COMP
    W1 & W2 -->|"Save Models & Maps"| B_MODELS & B_MAPS

    style COMPUTE fill:#e1f5fe,stroke:#0288d1
    style METADATA fill:#fff3e0,stroke:#f57c00
    style STORAGE fill:#e8f5e9,stroke:#388e3c
```

---

## 2. Distributed Synchronization: Atomic Tile Claims

When scaling to hundreds of tiles across multiple physical machines or virtual workers, workers coordinate via MongoDB's `tile_claims` collection to prevent redundant downloads and race conditions:

```mermaid
stateDiagram-v2
    [*] --> Unclaimed: Tile is queued
    Unclaimed --> Processing: Worker acquires claim (atomic upsert)
    
    state Processing {
        [*] --> Downloading
        Downloading --> Compositing
        Compositing --> Uploading
        Uploading --> Purging
    }

    Processing --> Completed: Pipeline succeeds -> release claim
    Processing --> DeadlockRecovery: Worker crashes (TTL > 7200s expires)
    DeadlockRecovery --> Processing: Reclaimed by another worker
    Completed --> [*]
```

### Claim Logic Details
1. **Atomic Upsert:** A worker queries `tile_claims` for `{tile: X, season: Y}`. If the record doesn't exist or is older than the Time-To-Live (TTL = 2 hours), the worker atomically updates the document with its `worker_id` and timestamps.
2. **Crash Recovery (TTL):** If a worker runs out of memory or crashes mid-tile, other workers can safely reclaim the tile after 7,200 seconds without manual intervention.
3. **Multi-Worker Strategies:**
   - **Forward & Reverse:** Node A processes alphabetically ($A \rightarrow Z$); Node B processes in reverse ($Z \rightarrow A$ with `REVERSE_TILES=true`). When they meet in the middle, claims prevent duplicate work.
   - **Worker Partitioning:** `process_composites_and_cleanup.py` supports deterministic modulo sharding via `--num-workers N --worker-id I`.

---

## 3. Storage Optimization & Automatic Purge Lifecycle

Processing 540 tiles across 4 seasons generates terabytes of intermediate satellite data. LandCoverPy implements a strict storage lifecycle to run reliably on limited disk:

```mermaid
sequenceDiagram
    autonumber
    participant W as Worker
    participant S3 as MinIO (S3)
    participant DB as MongoDB

    W->>S3: Download raw Sentinel-2 granules into s2-products
    W->>DB: Register capture metadata in products collection
    W->>W: Compute 11 cloud-masked median bands
    W->>S3: Upload compressed GeoTIFFs to s2-composites
    W->>DB: Register composite record in composites collection
    
    rect rgb(255, 235, 238)
    Note over W, S3: Immediate Batch Purge Phase
    W->>S3: Delete raw granules via S3 DeleteObject (chunks of 1,000)
    W->>DB: Unset heavy internal indexes ($unset: indexes, intermediateProducts)
    end
    Note over DB: Raw product count, capture dates, and footprints are PRESERVED for scientific auditing.
```

### Key Technical Implementations:
- **Batch S3 Deletion:** Objects are removed using MinIO's `remove_objects` with `DeleteObject` in batches of 1,000, avoiding thousands of individual HTTP requests.
- **Preserved Metadata:** `$unset` removes bulky pixel metadata arrays in Mongo while leaving the core product records (footprint polygon, capture date, tile ID) intact. This enables paper-level spatial audits without retaining raw raster files.
- **GeoTIFF Compression & Horizontal Predictor:** Composite bands and final maps are written with DEFLATE compression:
  - `predictor=3` for floating-point raster arrays (spectral indices).
  - `predictor=2` for integer arrays (reflectance bands and class masks).
  - This reduces disk footprint by **$\sim 45\text{–}60\%$** compared to standard GeoTIFFs.

---

## 4. Low-Level Resource Management

- **GDAL Cache Control:** `GDAL_CACHEMAX=512` caps the internal block cache to prevent unconstrained memory growth during windowed raster slicing.
- **Connection Recycling:** To prevent socket exhaustion (`FIN_WAIT`) in environments like Docker on WSL2, workers automatically recycle their HTTP session pools via exponential backoff and `_reset_global_session`.
