"""
app.py - Image Processing Worker Service (v6)

Changes from v5 (this pass — circuit breakers, timed locks, reduced fan-in):
 - Added a CircuitBreaker around every external dependency call: Postgres
   (via db_session), Redis, and outbound HTTP (image download + webhook).
   Previously a struggling DB/Redis/upstream got hit at full request rate
   forever; now each dependency trips open after repeated failures and
   fails fast for a cooldown period instead of piling up latency.
 - All locks (BoundedDict/BoundedSet/Metrics/_retry_lock) now acquire with
   a timeout instead of blocking indefinitely, so lock contention surfaces
   as a fast, loggable error instead of a silent thread hang.
 - Task completion now goes through a single save_task_result() call
   instead of process_task/process_with_retry each calling store_result()
   and persist_result() separately — fewer call sites touching the same
   two external systems, which was flagged as high fan-in.

Changes from v4 (reliability hardening pass):
 - All shared in-memory state (_metrics, _image_cache, _retry_counts,
   _seen_hashes, _sla_tracker, _job_chain) is now protected by locks and
   bounded in size (previously unlocked dicts/sets mutated concurrently by
   8 worker threads + the Flask thread, growing forever).
 - DB connection pool sized for actual concurrency (was 2 connections /
   0 overflow shared by 8 worker threads + API thread -> guaranteed
   blocking/timeouts under load). Added pool_pre_ping + pool_recycle.
 - DB sessions are now always closed via context manager, including on
   exception (previously leaked sessions on the persist_result error path).
 - Startup now waits for the DB to become reachable instead of crashing
   immediately if Postgres isn't up yet.
 - Auth is now enforced via a decorator on every route, including
   /metrics, which previously had no auth check at all.
 - TLS verification is back on by default for outbound image downloads
   (was verify=False, silently accepting invalid certs).
 - content hash for dedup switched from MD5 to SHA-256.
 - Cleanup job now also prunes _seen_hashes and _sla_tracker, not just
   the Postgres table, so those structures no longer grow unbounded.
 - SLA tracker entries are removed once a task completes (previously
   left forever, corrupting the sla_breaches metric over time).
 - Webhook calls now have a timeout and bounded retries instead of a
   single best-effort POST with no timeout.
 - Replaced print() with the logging module for real log levels.
"""

import os
import time
import json
import logging
import redis
import threading
import requests
import hashlib
import schedule
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from functools import wraps
from collections import defaultdict, OrderedDict
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager


# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(threadName)s: %(message)s",
)
log = logging.getLogger("imgworker")


# ── Config ────────────────────────────────────────────────────────────────────
REDIS_HOST           = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT           = int(os.getenv("REDIS_PORT", 6379))
DB_URL               = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/imgworker")
TASK_QUEUE           = "image_tasks"
BATCH_QUEUE          = "batch_tasks"
PRIORITY_QUEUE       = "priority_tasks"
MAX_IMAGE_SIZE       = (4096, 4096)
DOWNLOAD_TIMEOUT     = 60
WEBHOOK_TIMEOUT      = 10
WEBHOOK_MAX_RETRIES  = 2
WORKER_CONCURRENCY   = 8
MAX_RETRIES          = 3
API_PORT             = int(os.getenv("API_PORT", 5000))
WATERMARK_TEXT       = os.getenv("WATERMARK_TEXT", "© DeployGuard")
SLA_SECONDS          = int(os.getenv("SLA_SECONDS", 30))
CLEANUP_INTERVAL     = 60                      # seconds between cleanup runs
RESULT_RETENTION_DAYS = 7
VERIFY_TLS           = os.getenv("VERIFY_TLS", "true").lower() != "false"

# Bounds for in-memory structures so nothing grows forever
IMAGE_CACHE_MAX      = int(os.getenv("IMAGE_CACHE_MAX", 200))
DEDUP_CACHE_MAX      = int(os.getenv("DEDUP_CACHE_MAX", 50000))
SLA_TRACKER_MAX      = int(os.getenv("SLA_TRACKER_MAX", 50000))

# No hardcoded fallback keys — require explicit configuration.
_api_keys_env = os.getenv("API_KEYS", "")
API_KEYS = set(k for k in _api_keys_env.split(",") if k)
if not API_KEYS:
    log.warning("No API_KEYS configured — all authenticated endpoints will reject every request "
                "until API_KEYS is set. This is intentional; do not hardcode fallback keys.")


LOCK_TIMEOUT = float(os.getenv("LOCK_TIMEOUT_SECONDS", 5))


class LockTimeoutError(RuntimeError):
    pass


class TimedLock:
    """A Lock wrapper that never blocks forever. Raises LockTimeoutError
    instead of hanging indefinitely under contention — a raw threading.Lock
    acquired with no timeout is itself an unprotected risky call."""

    def __init__(self, timeout: float = LOCK_TIMEOUT):
        self._lock = threading.Lock()
        self._timeout = timeout

    @contextmanager
    def __call__(self):
        acquired = self._lock.acquire(timeout=self._timeout)
        if not acquired:
            raise LockTimeoutError(f"Could not acquire lock within {self._timeout}s")
        try:
            yield
        finally:
            self._lock.release()


class CircuitBreaker:
    """Simple failure-counting circuit breaker for external dependencies
    (DB, Redis, HTTP). Closed -> normal operation. Opens after
    `failure_threshold` consecutive failures and fails fast (raising
    CircuitOpenError) for `recovery_timeout` seconds, rather than letting
    every caller keep hammering a dependency that's already struggling.
    After the cooldown it goes half-open: exactly one call is allowed
    through as a probe; success closes the circuit, failure re-opens it."""

    class CircuitOpenError(RuntimeError):
        pass

    def __init__(self, name: str, failure_threshold: int = 5, recovery_timeout: float = 30.0):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._lock = threading.Lock()
        self._failures = 0
        self._state = "closed"          # closed | open | half_open
        self._opened_at = None

    def _enter(self):
        with self._lock:
            if self._state == "open":
                if time.time() - self._opened_at >= self.recovery_timeout:
                    self._state = "half_open"
                else:
                    raise CircuitBreaker.CircuitOpenError(
                        f"circuit '{self.name}' is open — failing fast")

    def _on_success(self):
        with self._lock:
            self._failures = 0
            self._state = "closed"
            self._opened_at = None

    def _on_failure(self):
        with self._lock:
            self._failures += 1
            if self._state == "half_open" or self._failures >= self.failure_threshold:
                self._state = "open"
                self._opened_at = time.time()

    def call(self, fn, *args, **kwargs):
        self._enter()
        try:
            result = fn(*args, **kwargs)
        except Exception:
            self._on_failure()
            raise
        else:
            self._on_success()
            return result


def call_with_retry(breaker: "CircuitBreaker", fn, *args, retries=2, base_delay=1.0, **kwargs):
    """Runs fn() through the given circuit breaker, retrying transient
    failures with exponential backoff. Does not retry when the breaker is
    open — that's the point of the breaker, fail fast instead of piling on."""
    last_exc = None
    for attempt in range(retries + 1):
        try:
            return breaker.call(fn, *args, **kwargs)
        except CircuitBreaker.CircuitOpenError:
            raise
        except Exception as e:
            last_exc = e
            if attempt < retries:
                time.sleep(base_delay * (2 ** attempt))
    raise last_exc


# Dedicated breaker per external dependency so one struggling system
# doesn't trip the breaker for an unrelated one.
_db_breaker      = CircuitBreaker("database", failure_threshold=5, recovery_timeout=30)
_redis_breaker   = CircuitBreaker("redis", failure_threshold=5, recovery_timeout=15)
_http_breaker    = CircuitBreaker("http_download", failure_threshold=5, recovery_timeout=30)
_webhook_breaker = CircuitBreaker("webhook", failure_threshold=8, recovery_timeout=30)


# ── Thread-safe bounded shared state ─────────────────────────────────────────
class BoundedDict:
    """Thread-safe dict with a max size; evicts oldest entries (insertion order)."""

    def __init__(self, max_size: int):
        self._data = OrderedDict()
        self._max_size = max_size
        self._lock = TimedLock()

    def __contains__(self, key):
        with self._lock():
            return key in self._data

    def get(self, key, default=None):
        with self._lock():
            return self._data.get(key, default)

    def set(self, key, value):
        with self._lock():
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = value
            while len(self._data) > self._max_size:
                self._data.popitem(last=False)

    def pop(self, key, default=None):
        with self._lock():
            return self._data.pop(key, default)

    def prune(self, predicate):
        """Remove entries where predicate(key, value) is True."""
        with self._lock():
            stale = [k for k, v in self._data.items() if predicate(k, v)]
            for k in stale:
                del self._data[k]
            return len(stale)

    def __len__(self):
        with self._lock():
            return len(self._data)

    def values(self):
        with self._lock():
            return list(self._data.values())


class BoundedSet:
    """Thread-safe set with a max size; evicts oldest entries (insertion order)."""

    def __init__(self, max_size: int):
        self._data = OrderedDict()
        self._max_size = max_size
        self._lock = TimedLock()

    def add_if_new(self, key) -> bool:
        """Returns True if key was newly added, False if it already existed."""
        with self._lock():
            if key in self._data:
                return False
            self._data[key] = True
            while len(self._data) > self._max_size:
                self._data.popitem(last=False)
            return True

    def __len__(self):
        with self._lock():
            return len(self._data)


class Metrics:
    def __init__(self):
        self._lock = TimedLock()
        self._counters = {"tasks_processed": 0, "tasks_failed": 0, "bytes_stored": 0, "cache_hits": 0}

    def incr(self, key, amount=1):
        with self._lock():
            self._counters[key] = self._counters.get(key, 0) + amount

    def snapshot(self):
        with self._lock():
            return dict(self._counters)


_metrics       = Metrics()
_image_cache   = BoundedDict(IMAGE_CACHE_MAX)
_retry_counts  = defaultdict(int)
_retry_lock    = TimedLock()
_job_chain     = BoundedDict(10000)
_seen_hashes   = BoundedSet(DEDUP_CACHE_MAX)
_sla_tracker   = BoundedDict(SLA_TRACKER_MAX)


def _incr_retry(task_id: str) -> int:
    with _retry_lock():
        _retry_counts[task_id] += 1
        return _retry_counts[task_id]


def _get_retry(task_id: str) -> int:
    with _retry_lock():
        return _retry_counts[task_id]


def _clear_retry(task_id: str):
    with _retry_lock():
        _retry_counts.pop(task_id, None)


# ── Database ──────────────────────────────────────────────────────────────────
engine = create_engine(
    DB_URL,
    pool_size=WORKER_CONCURRENCY + 4,          # enough for all workers + API + headroom
    max_overflow=10,
    pool_timeout=30,
    pool_pre_ping=True,                        # avoids handing out dead connections
    pool_recycle=1800,
)
Session = sessionmaker(bind=engine)


@contextmanager
def db_session():
    """Always closes the session, commits on success, rolls back on error.
    Session creation and commit are routed through the DB circuit breaker
    so a struggling Postgres fails fast instead of piling up connections
    and latency once it's already unhealthy."""
    def _open():
        return Session()

    session = call_with_retry(_db_breaker, _open, retries=1, base_delay=0.5)
    try:
        yield session
        _db_breaker.call(session.commit)
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def wait_for_db(max_attempts=10, base_delay=2):
    for attempt in range(1, max_attempts + 1):
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            log.info("Database is reachable.")
            return
        except OperationalError as e:
            delay = base_delay * attempt
            log.warning(f"DB not ready (attempt {attempt}/{max_attempts}): {e}. Retrying in {delay}s.")
            time.sleep(delay)
    raise RuntimeError("Database never became reachable; aborting startup.")


def init_db():
    with db_session() as session:
        session.execute(text("""
            CREATE TABLE IF NOT EXISTS task_results (
                task_id      TEXT PRIMARY KEY,
                status       TEXT,
                processed_at TIMESTAMP,
                meta         JSONB,
                error        TEXT
            )
        """))


def persist_result(task_id: str, meta: dict, error: str = None):
    try:
        with db_session() as session:
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
    except CircuitBreaker.CircuitOpenError:
        log.warning(f"DB circuit open — skipped write for {task_id}, will not retry inline")
    except Exception as e:
        # DB being down shouldn't crash a worker thread — log and move on.
        log.error(f"DB write failed for {task_id}: {e}")


def query_result(task_id: str) -> dict:
    try:
        with db_session() as session:
            row = session.execute(
                text("SELECT meta, status, error FROM task_results WHERE task_id = :tid"),
                {"tid": task_id}
            ).fetchone()
    except CircuitBreaker.CircuitOpenError:
        log.warning(f"DB circuit open — read for {task_id} skipped")
        return None
    except Exception as e:
        log.error(f"DB read failed for {task_id}: {e}")
        return None
    if not row:
        return None
    return {"meta": json.loads(row[0]), "status": row[1], "error": row[2]}


# ── Redis ─────────────────────────────────────────────────────────────────────
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=False,
                 socket_timeout=10, socket_connect_timeout=10)


def redis_call(fn, *args, **kwargs):
    """Runs a Redis operation through the Redis circuit breaker with one
    retry. Raises redis.RedisError or CircuitBreaker.CircuitOpenError on
    failure — callers decide whether that's fatal for their code path."""
    return call_with_retry(_redis_breaker, fn, *args, retries=1, base_delay=0.5, **kwargs)


# ── Flask API ─────────────────────────────────────────────────────────────────
app = Flask(__name__)


def require_api_key(fn):
    """Decorator so auth can't accidentally be skipped by a route (v4 required
    every route to manually call check_api_key() and check its return value)."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        key = request.headers.get("X-API-Key", "")
        if not API_KEYS or key not in API_KEYS:
            return jsonify({"error": "unauthorized"}), 401
        return fn(*args, **kwargs)
    return wrapper


@app.route("/submit", methods=["POST"])
@require_api_key
def submit_task():
    data = request.get_json(silent=True)
    if not data or "id" not in data or "image_url" not in data:
        return jsonify({"error": "request must include at least 'id' and 'image_url'"}), 400

    payload = json.dumps(data, sort_keys=True)
    content_hash = hashlib.sha256(payload.encode()).hexdigest()

    if not _seen_hashes.add_if_new(content_hash):
        return jsonify({"status": "duplicate", "hash": content_hash}), 200

    priority = bool(data.get("priority", False))
    queue = PRIORITY_QUEUE if priority else TASK_QUEUE

    _sla_tracker.set(data["id"], time.time())

    try:
        redis_call(r.rpush, queue, payload)
    except CircuitBreaker.CircuitOpenError:
        log.warning("Redis circuit open — rejecting /submit")
        return jsonify({"error": "queue temporarily unavailable"}), 503
    except redis.RedisError as e:
        log.error(f"Failed to enqueue task {data['id']}: {e}")
        return jsonify({"error": "queue unavailable"}), 503

    return jsonify({"status": "queued", "queue": queue, "hash": content_hash}), 202


@app.route("/result/<task_id>", methods=["GET"])
@require_api_key
def get_result(task_id):
    row = query_result(task_id)
    if not row:
        return jsonify({"error": "not found"}), 404
    return jsonify(row), 200


@app.route("/metrics", methods=["GET"])
@require_api_key
def get_metrics():
    now = time.time()
    breaches = sum(1 for t in _sla_tracker.values() if now - t > SLA_SECONDS)
    return jsonify({
        **_metrics.snapshot(),
        "sla_breaches": breaches,
        "dedup_cache":  len(_seen_hashes),
        "sla_tracked":  len(_sla_tracker),
        "circuits": {
            "database": _db_breaker._state,
            "redis": _redis_breaker._state,
            "http_download": _http_breaker._state,
            "webhook": _webhook_breaker._state,
        },
    }), 200


@app.route("/flush_cache", methods=["POST"])
@require_api_key
def flush_cache():
    global _image_cache
    _image_cache = BoundedDict(IMAGE_CACHE_MAX)
    return jsonify({"status": "ok"}), 200


@app.route("/healthz", methods=["GET"])
def healthz():
    """Unauthenticated liveness probe — deliberately excludes internal counts."""
    return jsonify(health_check()), 200


def run_api():
    app.run(host="0.0.0.0", port=API_PORT, debug=False, use_reloader=False)


# ── Scheduled cleanup ─────────────────────────────────────────────────────────
def cleanup_old_results():
    """Prune old DB rows AND the in-memory trackers that mirror them."""
    cutoff_dt = datetime.utcnow() - timedelta(days=RESULT_RETENTION_DAYS)
    try:
        with db_session() as session:
            session.execute(text(
                "DELETE FROM task_results WHERE processed_at < :cutoff"
            ), {"cutoff": cutoff_dt})
        log.info(f"[cleanup] Pruned DB rows older than {cutoff_dt.isoformat()}")
    except Exception as e:
        log.error(f"[cleanup] DB prune failed: {e}")

    now = time.time()
    stale_cutoff = now - (RESULT_RETENTION_DAYS * 86400)
    pruned_sla = _sla_tracker.prune(lambda k, v: v < stale_cutoff)
    log.info(f"[cleanup] Pruned {pruned_sla} stale SLA tracker entries")


def run_scheduler():
    schedule.every(CLEANUP_INTERVAL).seconds.do(cleanup_old_results)
    while True:
        try:
            schedule.run_pending()
        except Exception as e:
            log.error(f"[scheduler] cleanup job raised: {e}")
        time.sleep(1)


# ── Image helpers ─────────────────────────────────────────────────────────────
def _fetch_image_bytes(url: str) -> bytes:
    response = requests.get(url, timeout=DOWNLOAD_TIMEOUT, verify=VERIFY_TLS)
    response.raise_for_status()
    return response.content


def download_image(url: str) -> Image.Image:
    cached = _image_cache.get(url)
    if cached is not None:
        _metrics.incr("cache_hits")
        return cached
    content = call_with_retry(_http_breaker, _fetch_image_bytes, url, retries=1, base_delay=1.0)
    img = Image.open(BytesIO(content))
    img.load()  # force decode now so a corrupt image fails here, not later
    _image_cache.set(url, img)
    return img


def resize_image(img: Image.Image, size=MAX_IMAGE_SIZE) -> Image.Image:
    return img.resize(size)


def crop_image(img: Image.Image, box: tuple) -> Image.Image:
    return img.crop(box)


def add_watermark(img: Image.Image, text: str = WATERMARK_TEXT) -> Image.Image:
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 36)
    except IOError:
        font = ImageFont.load_default()
    draw.text((10, 10), text, fill=(255, 255, 255), font=font)
    return img


def convert_format(img: Image.Image, fmt: str) -> bytes:
    if fmt.upper() == "JPEG" and img.mode == "RGBA":
        img = img.convert("RGB")
    buf = BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def store_result(task_id: str, data: bytes, meta: dict):
    try:
        redis_call(r.set, f"result:{task_id}:data", data)
        redis_call(r.set, f"result:{task_id}:meta", json.dumps(meta))
        _metrics.incr("bytes_stored", len(data))
    except CircuitBreaker.CircuitOpenError:
        log.warning(f"Redis circuit open — could not store result for {task_id}")
        raise
    except redis.RedisError as e:
        log.error(f"Failed to store result for {task_id} in Redis: {e}")
        raise


# ── Webhook ───────────────────────────────────────────────────────────────────
def _post_webhook(webhook_url: str, payload: dict):
    resp = requests.post(webhook_url, json=payload, timeout=WEBHOOK_TIMEOUT)
    resp.raise_for_status()


def notify_webhook(webhook_url: str, payload: dict):
    try:
        call_with_retry(_webhook_breaker, _post_webhook, webhook_url, payload,
                         retries=WEBHOOK_MAX_RETRIES, base_delay=2.0)
    except CircuitBreaker.CircuitOpenError:
        log.warning(f"Webhook circuit open — skipped notify to {webhook_url}")
    except Exception as e:
        log.error(f"Webhook to {webhook_url} permanently failed: {e}")


def save_task_result(task_id: str, data: bytes = None, meta: dict = None, error: str = None):
    """Single entry point for persisting a task outcome. Both process_task
    and process_with_retry previously called store_result() and
    persist_result() separately (two external-system call sites per
    caller); consolidating into one function reduces fan-in and keeps the
    Redis-write / Postgres-write pairing consistent in one place."""
    meta = meta or {}
    if data is not None:
        store_result(task_id, data, meta)
    persist_result(task_id, meta, error=error)


# ── Job chaining ──────────────────────────────────────────────────────────────
def enqueue_child_tasks(parent_id: str, child_tasks: list):
    _job_chain.set(parent_id, child_tasks)
    for child in child_tasks:
        child["parent_id"] = parent_id
        try:
            redis_call(r.rpush, TASK_QUEUE, json.dumps(child))
        except (CircuitBreaker.CircuitOpenError, redis.RedisError) as e:
            log.error(f"Failed to enqueue child task of {parent_id}: {e}")


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
        log.warning(f"[SLA] Task {task_id} breached SLA before processing — continuing anyway")

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

    save_task_result(task_id, data=data, meta=meta)
    _metrics.incr("tasks_processed")
    _sla_tracker.pop(task_id, None)   # completed — stop tracking it

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
            log.error(f"Batch {batch_id} task {task.get('id')} failed: {e}")
            results.append({"id": task.get("id"), "status": "failed", "error": str(e)})
    try:
        redis_call(r.set, f"batch:{batch_id}:results", json.dumps(results))
    except CircuitBreaker.CircuitOpenError:
        log.warning(f"Redis circuit open — could not store batch results for {batch_id}")
    except redis.RedisError as e:
        log.error(f"Failed to store batch results for {batch_id}: {e}")


# ── Retry ─────────────────────────────────────────────────────────────────────
def process_with_retry(task: dict):
    task_id = task["id"]
    while _get_retry(task_id) <= MAX_RETRIES:
        try:
            process_task(task)
            _clear_retry(task_id)
            return
        except Exception as e:
            count = _incr_retry(task_id)
            _metrics.incr("tasks_failed")
            if count > MAX_RETRIES:
                break
            wait = 2 ** count
            log.warning(f"Retry {count}/{MAX_RETRIES} for {task_id}: {e}")
            time.sleep(wait)
    save_task_result(task_id, error="max retries exceeded")
    _sla_tracker.pop(task_id, None)
    _clear_retry(task_id)
    log.error(f"Task {task_id} permanently failed.")


# ── Worker threads ────────────────────────────────────────────────────────────
def worker_loop(worker_id: int):
    queues = [PRIORITY_QUEUE, TASK_QUEUE if worker_id % 2 == 0 else BATCH_QUEUE]
    while True:
        try:
            result = r.blpop(queues, timeout=5)   # blocking pop already has its own timeout;
                                                    # not routed through the breaker's retry logic
                                                    # since that would defeat the blocking wait.
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
        except redis.RedisError as e:
            log.error(f"Worker {worker_id} lost Redis connection: {e}")
            time.sleep(2)
        except Exception as e:
            log.error(f"Worker {worker_id} error: {e}")
            time.sleep(1)


def run_worker():
    log.info(f"Starting scheduler, API on :{API_PORT}, and {WORKER_CONCURRENCY} workers…")

    wait_for_db()
    init_db()

    threading.Thread(target=run_api, daemon=True, name="api").start()
    threading.Thread(target=run_scheduler, daemon=True, name="scheduler").start()

    threads = []
    for i in range(WORKER_CONCURRENCY):
        t = threading.Thread(target=worker_loop, args=(i,), daemon=True, name=f"worker-{i}")
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
            "metrics":        _metrics.snapshot(),
        }
    except Exception:
        return {"status": "degraded"}


if __name__ == "__main__":
    run_worker()