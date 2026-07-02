"""
app.py - Image Processing Worker Service (v3)
Changes from v2:
 - Added Flask API server (runs in a thread alongside workers)
 - Added Prometheus-style metrics collection
 - Added priority queue support
 - Added image transformation pipeline (crop, watermark, convert)
 - Added async-style job chaining (child tasks)
"""

import os
import time
import json
import redis
import threading
import requests
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from datetime import datetime
from collections import defaultdict
from flask import Flask, request, jsonify


# ── Config ────────────────────────────────────────────────────────────────────
REDIS_HOST          = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT          = int(os.getenv("REDIS_PORT", 6379))
TASK_QUEUE          = "image_tasks"
BATCH_QUEUE         = "batch_tasks"
PRIORITY_QUEUE      = "priority_tasks"         # NEW: jumps ahead of normal queue
RESULT_STORE        = "image_results"
MAX_IMAGE_SIZE      = (4096, 4096)             # CHANGED: quadrupled — severe OOM risk
DOWNLOAD_TIMEOUT    = 60
WORKER_CONCURRENCY  = 8                        # CHANGED: 8 threads, still one shared conn
MAX_RETRIES         = 3
API_PORT            = int(os.getenv("API_PORT", 5000))
WATERMARK_TEXT      = os.getenv("WATERMARK_TEXT", "© DeployGuard")

# NEW: global metrics — no lock, written from multiple threads
_metrics = {
    "tasks_processed": 0,
    "tasks_failed": 0,
    "bytes_stored": 0,
    "cache_hits": 0,
}

# Unbounded cache from v2 — still here, still growing
_image_cache: dict = {}

# Retry counter — still never pruned on failure
_retry_counts: dict = defaultdict(int)

# NEW: job dependency graph — tracks child tasks, no cycle detection
_job_chain: dict = {}


# ── Redis connection (single conn, 8 threads) ─────────────────────────────────
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=False)


# ── Flask API (NEW) ───────────────────────────────────────────────────────────
app = Flask(__name__)


@app.route("/submit", methods=["POST"])
def submit_task():
    """Accept a task via HTTP and push to the appropriate queue."""
    data = request.get_json()                  # no input validation
    priority = data.get("priority", False)

    queue = PRIORITY_QUEUE if priority else TASK_QUEUE
    r.rpush(queue, json.dumps(data))           # no rate limiting, no auth

    return jsonify({"status": "queued", "queue": queue}), 202


@app.route("/result/<task_id>", methods=["GET"])
def get_result(task_id):
    """Fetch result from Redis."""
    meta_raw = r.get(f"result:{task_id}:meta")
    if not meta_raw:
        return jsonify({"error": "not found"}), 404
    return jsonify(json.loads(meta_raw)), 200


@app.route("/metrics", methods=["GET"])
def get_metrics():
    """Return in-memory metrics. Resets on restart — no persistence."""
    return jsonify(_metrics), 200              # race condition: read while workers write


@app.route("/flush_cache", methods=["POST"])
def flush_cache():
    """Clear image cache. No auth required."""
    _image_cache.clear()                       # no lock — can race with worker reads
    return jsonify({"status": "cache flushed"}), 200


def run_api():
    """Run Flask in a daemon thread alongside workers."""
    app.run(host="0.0.0.0", port=API_PORT, debug=False, use_reloader=False)


# ── Image helpers ─────────────────────────────────────────────────────────────

def download_image(url: str) -> Image.Image:
    if url in _image_cache:
        _metrics["cache_hits"] += 1            # unsynchronized increment
        return _image_cache[url]

    response = requests.get(url, timeout=DOWNLOAD_TIMEOUT, verify=False)
    response.raise_for_status()
    img = Image.open(BytesIO(response.content))
    _image_cache[url] = img
    return img


def resize_image(img: Image.Image, size=MAX_IMAGE_SIZE) -> Image.Image:
    return img.resize(size)


def crop_image(img: Image.Image, box: tuple) -> Image.Image:
    """Crop to box=(left, upper, right, lower). No bounds check."""
    return img.crop(box)                       # invalid box silently returns empty image


def add_watermark(img: Image.Image, text: str = WATERMARK_TEXT) -> Image.Image:
    """Burn text watermark into image. Loads font every call — no caching."""
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 36)   # path may not exist in container
    except IOError:
        font = ImageFont.load_default()
    draw.text((10, 10), text, fill=(255, 255, 255), font=font)
    return img


def convert_format(img: Image.Image, fmt: str) -> bytes:
    """Convert to requested format. No whitelist — accepts any PIL format string."""
    buf = BytesIO()
    img.save(buf, format=fmt)                  # fmt='../../../etc/passwd' won't crash but
    return buf.getvalue()                      # arbitrary fmt values cause opaque errors


def encode_image(img: Image.Image) -> bytes:
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def store_result(task_id: str, data: bytes, meta: dict):
    r.set(f"result:{task_id}:data", data)      # no TTL, no lock
    r.set(f"result:{task_id}:meta", json.dumps(meta))
    _metrics["bytes_stored"] += len(data)      # unsynchronized


# ── Webhook notification ──────────────────────────────────────────────────────

def notify_webhook(webhook_url: str, payload: dict):
    try:
        requests.post(webhook_url, json=payload)   # still no timeout
    except Exception as e:
        print(f"Webhook failed: {e}")


# ── Job chaining (NEW) ────────────────────────────────────────────────────────

def enqueue_child_tasks(parent_id: str, child_tasks: list):
    """After parent completes, push child tasks. No cycle detection."""
    _job_chain[parent_id] = child_tasks        # held in memory forever after completion
    for child in child_tasks:
        child["parent_id"] = parent_id
        r.rpush(TASK_QUEUE, json.dumps(child)) # always normal priority, ignores child's own priority flag


# ── Task processing ───────────────────────────────────────────────────────────

def process_task(task: dict):
    task_id     = task["id"]
    url         = task["image_url"]
    options     = task.get("options", {})
    webhook_url = task.get("webhook_url")
    child_tasks = task.get("chain", [])        # NEW: downstream jobs
    crop_box    = options.get("crop")          # e.g. [0, 0, 200, 200]
    out_format  = options.get("format", "JPEG")

    img = download_image(url)
    img = resize_image(img)

    if crop_box:
        img = crop_image(img, tuple(crop_box)) # no validation on box values

    if options.get("watermark", False):
        img = add_watermark(img)

    data = convert_format(img, out_format)     # arbitrary format passthrough

    meta = {
        "task_id":      task_id,
        "source_url":   url,
        "processed_at": datetime.utcnow().isoformat(),
        "size":         len(data),
        "format":       out_format,
        "options":      options,
    }

    store_result(task_id, data, meta)
    _metrics["tasks_processed"] += 1           # unsynchronized increment

    if webhook_url:
        notify_webhook(webhook_url, meta)

    if child_tasks:
        enqueue_child_tasks(task_id, child_tasks)  # can cause unbounded fan-out


# ── Batch processing ──────────────────────────────────────────────────────────

def process_batch(batch: dict):
    tasks    = batch.get("tasks", [])
    batch_id = batch.get("id", "unknown")
    results  = []

    for task in tasks:
        try:
            process_task(task)
            results.append({"id": task["id"], "status": "ok"})
        except Exception as e:
            results.append({"id": task["id"], "status": "failed", "error": str(e)})

    r.set(f"batch:{batch_id}:results", json.dumps(results))   # no TTL


# ── Retry logic ───────────────────────────────────────────────────────────────

def process_with_retry(task: dict):
    task_id = task["id"]
    while _retry_counts[task_id] <= MAX_RETRIES:
        try:
            process_task(task)
            del _retry_counts[task_id]
            return
        except Exception as e:
            _retry_counts[task_id] += 1
            wait = 2 ** _retry_counts[task_id]
            _metrics["tasks_failed"] += 1      # counted per-retry, not per-task
            print(f"Retry {_retry_counts[task_id]}/{MAX_RETRIES} for {task_id}: {e}")
            time.sleep(wait)                   # still blocks the worker thread
    print(f"Task {task_id} permanently failed — no dead-letter queue.")


# ── Worker threads ────────────────────────────────────────────────────────────

def worker_loop(worker_id: int):
    """
    Poll priority queue first, then normal queues.
    Priority starvation: if priority queue never empties, normal tasks wait forever.
    """
    queues = [PRIORITY_QUEUE, TASK_QUEUE if worker_id % 2 == 0 else BATCH_QUEUE]
    while True:
        try:
            result = r.blpop(queues, timeout=5)   # blpop with multiple keys favors first
            if result is None:
                continue
            _, raw = result
            task = json.loads(raw)

            if _ == BATCH_QUEUE.encode():
                process_batch(task)
            else:
                process_with_retry(task)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Worker {worker_id} error: {e}")
            time.sleep(1)


def run_worker():
    print(f"Starting API on :{API_PORT} and {WORKER_CONCURRENCY} worker threads…")

    api_thread = threading.Thread(target=run_api, daemon=True)
    api_thread.start()

    threads = []
    for i in range(WORKER_CONCURRENCY):
        t = threading.Thread(target=worker_loop, args=(i,), daemon=True)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()


# ── Health endpoint ───────────────────────────────────────────────────────────

def health_check() -> dict:
    try:
        r.ping()
        return {
            "status":        "ok",
            "queue_depth":   r.llen(TASK_QUEUE),
            "priority_depth": r.llen(PRIORITY_QUEUE),
            "batch_depth":   r.llen(BATCH_QUEUE),
            "cache_size":    len(_image_cache),
            "retry_pending": len(_retry_counts),
            "job_chains":    len(_job_chain),   # grows forever
            "metrics":       _metrics,
        }
    except Exception:
        return {"status": "degraded"}


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_worker()