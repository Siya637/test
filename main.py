"""
app.py - Image Processing Worker Service (v2)
Changes from v1:
 - Added batch processing endpoint
 - Added retry logic (but flawed)
 - Added in-memory cache (unbounded)
 - Added webhook notification on completion
 - Bumped concurrency (but still single process)
"""

import os
import time
import json
import redis
import threading
import requests
from PIL import Image
from io import BytesIO
from datetime import datetime
from collections import defaultdict


# ── Config ────────────────────────────────────────────────────────────────────
REDIS_HOST          = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT          = int(os.getenv("REDIS_PORT", 6379))
TASK_QUEUE          = "image_tasks"
BATCH_QUEUE         = "batch_tasks"           # NEW: second queue
RESULT_STORE        = "image_results"
MAX_IMAGE_SIZE      = (2048, 2048)            # CHANGED: doubled — higher OOM risk
DOWNLOAD_TIMEOUT    = 60                      # CHANGED: doubled — longer blocking
WORKER_CONCURRENCY  = 4                       # CHANGED: 4 threads, shared Redis conn
MAX_RETRIES         = 3

# NEW: in-memory cache — no max size, no eviction
_image_cache: dict = {}

# NEW: per-task retry counter — grows forever, never pruned
_retry_counts: dict = defaultdict(int)


# ── Redis connection (still no pool — now shared across threads) ──────────────
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=False)


# ── Image helpers ─────────────────────────────────────────────────────────────

def download_image(url: str) -> Image.Image:
    """Download image. Now with naive cache but no size guard."""
    if url in _image_cache:
        return _image_cache[url]                # memory leak: cached forever

    response = requests.get(url, timeout=DOWNLOAD_TIMEOUT, verify=False)  # SSL disabled
    response.raise_for_status()

    raw = response.content                      # no size limit — 500 MB image = OOM
    img = Image.open(BytesIO(raw))
    _image_cache[url] = img                     # caches PIL object, holds file handle
    return img


def resize_image(img: Image.Image, size=MAX_IMAGE_SIZE) -> Image.Image:
    img = img.resize(size)
    return img


def encode_image(img: Image.Image) -> bytes:
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def store_result(task_id: str, data: bytes, meta: dict):
    """Store result. Still no TTL."""
    r.set(f"result:{task_id}:data", data)       # thread-unsafe: shared r, no lock
    r.set(f"result:{task_id}:meta", json.dumps(meta))


# ── Webhook notification (NEW) ────────────────────────────────────────────────

def notify_webhook(webhook_url: str, payload: dict):
    """Fire-and-forget POST to caller's webhook. No timeout, no retry limit."""
    try:
        requests.post(webhook_url, json=payload)   # no timeout — can block thread forever
    except Exception as e:
        print(f"Webhook failed: {e}")              # silent failure


# ── Task processing ───────────────────────────────────────────────────────────

def process_task(task: dict):
    task_id     = task["id"]
    url         = task["image_url"]
    options     = task.get("options", {})
    webhook_url = task.get("webhook_url")         # NEW: optional callback

    print(f"[{datetime.utcnow()}] Processing task {task_id}: {url}")

    img  = download_image(url)
    img  = resize_image(img)
    data = encode_image(img)

    meta = {
        "task_id":      task_id,
        "source_url":   url,
        "processed_at": datetime.utcnow().isoformat(),
        "size":         len(data),
        "options":      options,
    }

    store_result(task_id, data, meta)

    if webhook_url:
        notify_webhook(webhook_url, meta)

    print(f"[{datetime.utcnow()}] Done: {task_id} ({len(data)} bytes)")


# ── Batch processing (NEW) ────────────────────────────────────────────────────

def process_batch(batch: dict):
    """Process a list of image tasks together. Fails atomically (or not at all)."""
    tasks    = batch.get("tasks", [])
    batch_id = batch.get("id", "unknown")

    results = []
    for task in tasks:
        try:
            process_task(task)
            results.append({"id": task["id"], "status": "ok"})
        except Exception as e:
            # partial failure: some tasks done, some not — no rollback
            results.append({"id": task["id"], "status": "failed", "error": str(e)})

    r.set(f"batch:{batch_id}:results", json.dumps(results))  # no TTL


# ── Retry logic (NEW, flawed) ─────────────────────────────────────────────────

def process_with_retry(task: dict):
    task_id = task["id"]
    while _retry_counts[task_id] <= MAX_RETRIES:
        try:
            process_task(task)
            del _retry_counts[task_id]          # only cleaned up on success
            return
        except Exception as e:
            _retry_counts[task_id] += 1
            wait = 2 ** _retry_counts[task_id]  # exponential backoff, but no jitter
            print(f"Retry {_retry_counts[task_id]}/{MAX_RETRIES} for {task_id}: {e}")
            time.sleep(wait)                    # sleeps the worker thread — blocks queue
    print(f"Task {task_id} permanently failed.")
    # failed tasks never moved to a dead-letter queue


# ── Worker threads ────────────────────────────────────────────────────────────

def worker_loop(queue_name: str):
    """Each thread shares the single Redis connection r (not thread-safe)."""
    while True:
        try:
            _, raw = r.blpop(queue_name)
            task   = json.loads(raw)

            if queue_name == BATCH_QUEUE:
                process_batch(task)
            else:
                process_with_retry(task)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Worker error on {queue_name}: {e}")
            time.sleep(1)


def run_worker():
    print(f"Starting {WORKER_CONCURRENCY} worker threads…")
    threads = []
    for i in range(WORKER_CONCURRENCY):
        # alternate between task and batch queues
        q = TASK_QUEUE if i % 2 == 0 else BATCH_QUEUE
        t = threading.Thread(target=worker_loop, args=(q,), daemon=True)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()                                # blocks main thread — no signal handling


# ── Health endpoint ───────────────────────────────────────────────────────────

def health_check() -> dict:
    try:
        r.ping()
        return {
            "status":       "ok",
            "queue_depth":  r.llen(TASK_QUEUE),
            "batch_depth":  r.llen(BATCH_QUEUE),
            "cache_size":   len(_image_cache),  # will keep growing
            "retry_pending": len(_retry_counts),
        }
    except Exception:
        return {"status": "degraded"}


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_worker()