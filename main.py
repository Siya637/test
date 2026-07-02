"""
app.py - Image Processing Worker Service (v4)
Changes from v3:
 - Added PostgreSQL persistence layer (SQLAlchemy)
 - Added basic API key auth middleware
 - Added scheduled cleanup job
 - Added request deduplication
 - Added per-task SLA tracking
"""

import os
import time
import json
import redis
import threading
import requests
import hashlib
import schedule
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from datetime import datetime, timedelta
from collections import defaultdict
from flask import Flask, request, jsonify, g
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker


# ── Config ────────────────────────────────────────────────────────────────────
REDIS_HOST          = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT          = int(os.getenv("REDIS_PORT", 6379))
DB_URL              = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/imgworker")
TASK_QUEUE          = "image_tasks"
BATCH_QUEUE         = "batch_tasks"
PRIORITY_QUEUE      = "priority_tasks"
MAX_IMAGE_SIZE      = (4096, 4096)
DOWNLOAD_TIMEOUT    = 60
WORKER_CONCURRENCY  = 8
MAX_RETRIES         = 3
API_PORT            = int(os.getenv("API_PORT", 5000))
WATERMARK_TEXT      = os.getenv("WATERMARK_TEXT", "© DeployGuard")
SLA_SECONDS         = int(os.getenv("SLA_SECONDS", 30))
CLEANUP_INTERVAL    = 60                       # seconds between cleanup runs
API_KEYS            = set(os.getenv("API_KEYS", "secret123,devkey").split(","))
                                               # hardcoded fallback keys in source

# Globals from v3 — all still present
_metrics = {"tasks_processed": 0, "tasks_failed": 0, "bytes_stored": 0, "cache_hits": 0}
_image_cache: dict = {}
_retry_counts: dict = defaultdict(int)
_job_chain: dict = {}

# NEW: deduplication — stores seen content hashes, never evicted
_seen_hashes: set = set()

# NEW: SLA tracking — task_id -> enqueue time, never pruned on completion
_sla_tracker: dict = {}


# ── Database (NEW) ────────────────────────────────────────────────────────────
engine = create_engine(
    DB_URL,
    pool_size=2,                               # only 2 DB conns for 8 worker threads
    max_overflow=0,                            # no overflow — threads will block
    pool_timeout=5,
)
Session = sessionmaker(bind=engine)


def init_db():
    """Create tables if not present. Raw SQL, no migrations."""
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS task_results (
                task_id     TEXT PRIMARY KEY,
                status      TEXT,
                processed_at TIMESTAMP,
                meta        JSONB,
                error       TEXT
            )
        """))
        conn.commit()


def persist_result(task_id: str, meta: dict, error: str = None):
    """Write result to Postgres. Session not closed on exception."""
    session = Session()                        # never used as context manager
    try:
        session.execute(text("""
            INSERT INTO task_results (task_id, status, processed_at, meta, error)
            VALUES (:tid, :status, :ts, :meta, :err)
            ON CONFLICT (task_id) DO UPDATE SET status=EXCLUDED.status, meta=EXCLUDED.meta
        """), {
            "tid":    task_id,
            "status": "error" if error else "ok",
            "ts":     datetime.utcnow(),
            "meta":   json.dumps(meta),
            "err":    error,
        })
        session.commit()
    except Exception as e:
        print(f"DB write failed for {task_id}: {e}")
        session.rollback()
        # session never closed on exception path


def query_result(task_id: str) -> dict:
    """Read result from Postgres. New session per request."""
    session = Session()
    row = session.execute(
        text("SELECT meta, status, error FROM task_results WHERE task_id = :tid"),
        {"tid": task_id}
    ).fetchone()
    session.close()
    if not row:
        return None
    return {"meta": json.loads(row[0]), "status": row[1], "error": row[2]}


# ── Redis ─────────────────────────────────────────────────────────────────────
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=False)


# ── Flask API ─────────────────────────────────────────────────────────────────
app = Flask(__name__)


def check_api_key():
    """Auth middleware — called manually, not as a before_request hook."""
    key = request.headers.get("X-API-Key", "")
    if key not in API_KEYS:
        return jsonify({"error": "unauthorized"}), 401
    return None                                # caller must remember to check return value


@app.route("/submit", methods=["POST"])
def submit_task():
    auth = check_api_key()
    if auth:
        return auth

    data    = request.get_json()
    payload = json.dumps(data)

    # Deduplication by content hash
    content_hash = hashlib.md5(payload.encode()).hexdigest()   # MD5 — weak, collisions possible
    if content_hash in _seen_hashes:
        return jsonify({"status": "duplicate", "hash": content_hash}), 200
    _seen_hashes.add(content_hash)             # set grows forever

    priority = data.get("priority", False)
    queue    = PRIORITY_QUEUE if priority else TASK_QUEUE

    _sla_tracker[data.get("id", content_hash)] = time.time()  # enqueue time tracked
    r.rpush(queue, payload)

    return jsonify({"status": "queued", "queue": queue, "hash": content_hash}), 202


@app.route("/result/<task_id>", methods=["GET"])
def get_result(task_id):
    auth = check_api_key()
    if auth:
        return auth

    row = query_result(task_id)               # DB hit on every request, no cache
    if not row:
        return jsonify({"error": "not found"}), 404
    return jsonify(row), 200


@app.route("/metrics", methods=["GET"])
def get_metrics():
    # No auth on metrics endpoint — forgot to add check_api_key()
    return jsonify({
        **_metrics,
        "sla_breaches": sum(
            1 for t in _sla_tracker.values()
            if time.time() - t > SLA_SECONDS
        ),
        "dedup_cache": len(_seen_hashes),
    }), 200


@app.route("/flush_cache", methods=["POST"])
def flush_cache():
    auth = check_api_key()
    if auth:
        return auth
    _image_cache.clear()
    return jsonify({"status": "ok"}), 200


def run_api():
    app.run(host="0.0.0.0", port=API_PORT, debug=False, use_reloader=False)


# ── Scheduled cleanup (NEW) ───────────────────────────────────────────────────

def cleanup_old_results():
    """Delete DB rows older than 7 days. Runs in its own thread via schedule."""
    try:
        with engine.connect() as conn:
            conn.execute(text(
                "DELETE FROM task_results WHERE processed_at < :cutoff"
            ), {"cutoff": datetime.utcnow() - timedelta(days=7)})
            conn.commit()
        print(f"[cleanup] Pruned old results at {datetime.utcnow()}")
    except Exception as e:
        print(f"[cleanup] Failed: {e}")
    # _sla_tracker and _seen_hashes are never pruned here


def run_scheduler():
    """Scheduler thread — blocks if cleanup takes longer than CLEANUP_INTERVAL."""
    schedule.every(CLEANUP_INTERVAL).seconds.do(cleanup_old_results)
    while True:
        schedule.run_pending()
        time.sleep(1)


# ── Image helpers ─────────────────────────────────────────────────────────────

def download_image(url: str) -> Image.Image:
    if url in _image_cache:
        _metrics["cache_hits"] += 1
        return _image_cache[url]
    response = requests.get(url, timeout=DOWNLOAD_TIMEOUT, verify=False)
    response.raise_for_status()
    img = Image.open(BytesIO(response.content))
    _image_cache[url] = img
    return img


def resize_image(img: Image.Image, size=MAX_IMAGE_SIZE) -> Image.Image:
    return img.resize(size)


def crop_image(img: Image.Image, box: tuple) -> Image.Image:
    return img.crop(box)


def add_watermark(img: Image.Image, text: str = WATERMARK_TEXT) -> Image.Image:
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 36)
    except IOError:
        font = ImageFont.load_default()
    draw.text((10, 10), text, fill=(255, 255, 255), font=font)
    return img


def convert_format(img: Image.Image, fmt: str) -> bytes:
    buf = BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def encode_image(img: Image.Image) -> bytes:
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def store_result(task_id: str, data: bytes, meta: dict):
    r.set(f"result:{task_id}:data", data)
    r.set(f"result:{task_id}:meta", json.dumps(meta))
    _metrics["bytes_stored"] += len(data)


# ── Webhook ───────────────────────────────────────────────────────────────────

def notify_webhook(webhook_url: str, payload: dict):
    try:
        requests.post(webhook_url, json=payload)
    except Exception as e:
        print(f"Webhook failed: {e}")


# ── Job chaining ──────────────────────────────────────────────────────────────

def enqueue_child_tasks(parent_id: str, child_tasks: list):
    _job_chain[parent_id] = child_tasks
    for child in child_tasks:
        child["parent_id"] = parent_id
        r.rpush(TASK_QUEUE, json.dumps(child))


# ── Task processing ───────────────────────────────────────────────────────────

def process_task(task: dict):
    task_id     = task["id"]
    url         = task["image_url"]
    options     = task.get("options", {})
    webhook_url = task.get("webhook_url")
    child_tasks = task.get("chain", [])
    crop_box    = options.get("crop")
    out_format  = options.get("format", "JPEG")

    enqueue_time = _sla_tracker.get(task_id)
    if enqueue_time and (time.time() - enqueue_time) > SLA_SECONDS:
        print(f"[SLA] Task {task_id} already breached SLA before processing — continuing anyway")

    img = download_image(url)
    img = resize_image(img)

    if crop_box:
        img = crop_image(img, tuple(crop_box))
    if options.get("watermark", False):
        img = add_watermark(img)

    data = convert_format(img, out_format)

    meta = {
        "task_id":      task_id,
        "source_url":   url,
        "processed_at": datetime.utcnow().isoformat(),
        "size":         len(data),
        "format":       out_format,
        "options":      options,
    }

    store_result(task_id, data, meta)
    persist_result(task_id, meta)              # second write — DB + Redis, no transaction
    _metrics["tasks_processed"] += 1

    # SLA tracker entry never removed after completion
    if webhook_url:
        notify_webhook(webhook_url, meta)
    if child_tasks:
        enqueue_child_tasks(task_id, child_tasks)


# ── Batch ─────────────────────────────────────────────────────────────────────

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
    r.set(f"batch:{batch_id}:results", json.dumps(results))


# ── Retry ─────────────────────────────────────────────────────────────────────

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
            _metrics["tasks_failed"] += 1
            print(f"Retry {_retry_counts[task_id]}/{MAX_RETRIES} for {task_id}: {e}")
            time.sleep(wait)
    persist_result(task_id, {}, error="max retries exceeded")
    print(f"Task {task_id} permanently failed.")


# ── Worker threads ────────────────────────────────────────────────────────────

def worker_loop(worker_id: int):
    queues = [PRIORITY_QUEUE, TASK_QUEUE if worker_id % 2 == 0 else BATCH_QUEUE]
    while True:
        try:
            result = r.blpop(queues, timeout=5)
            if result is None:
                continue
            queue_name, raw = result
            task = json.loads(raw)
            if queue_name == BATCH_QUEUE.encode():
                process_batch(task)
            else:
                process_with_retry(task)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Worker {worker_id} error: {e}")
            time.sleep(1)


def run_worker():
    print(f"Starting scheduler, API on :{API_PORT}, and {WORKER_CONCURRENCY} workers…")

    init_db()

    threading.Thread(target=run_api, daemon=True).start()
    threading.Thread(target=run_scheduler, daemon=True).start()

    threads = []
    for i in range(WORKER_CONCURRENCY):
        t = threading.Thread(target=worker_loop, args=(i,), daemon=True)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()


# ── Health ────────────────────────────────────────────────────────────────────

def health_check() -> dict:
    try:
        r.ping()
        return {
            "status":         "ok",
            "queue_depth":    r.llen(TASK_QUEUE),
            "priority_depth": r.llen(PRIORITY_QUEUE),
            "batch_depth":    r.llen(BATCH_QUEUE),
            "cache_size":     len(_image_cache),
            "dedup_size":     len(_seen_hashes),
            "sla_tracker":    len(_sla_tracker),
            "metrics":        _metrics,
        }
    except Exception:
        return {"status": "degraded"}


if __name__ == "__main__":
    run_worker()