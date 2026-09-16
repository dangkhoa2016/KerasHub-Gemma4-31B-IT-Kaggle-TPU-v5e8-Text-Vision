from __future__ import annotations
import hmac, json, logging, os, re, threading, time, uuid
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable

from .. import __version__
from ..core.errors import (
    QueueFullError,
    StoreFullError,
    ValidationError,
    WorkerNotReadyError,
)
from ..core.validation import (
    parse_generate_payload,
    parse_image_binary,
    parse_image_payload,
)

logger = logging.getLogger("gemma4_server")
API_VERSION = __version__
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")

@dataclass
class Runtime:
    config: Any
    manager: Any
    restart_lock: threading.Lock = field(
        default_factory=threading.Lock
    )

def normalize_request_id(value):
    candidate = (
        value.strip() if isinstance(value, str) else ""
    )
    if candidate and _REQUEST_ID_RE.fullmatch(candidate):
        return candidate
    return f"req-{uuid.uuid4().hex[:24]}"

def build_info_payload(health):
    return {
        "service": "Gemma 4 31B Instruct",
        "api_version": API_VERSION,
        "architecture": (
            "single logical model sharded across 8 TPU devices"
        ),
        "state": health.get("state"),
        "worker_generation": health.get(
            "worker_generation"
        ),
        "runtime": health.get("runtime", {}),
        "capabilities": {
            "text": True,
            "vision": True,
            "audio": False,
            "async_jobs": True,
            "request_ids": True,
            "generation_mode": (
                "keras_hub_native_unvalidated"
            ),
        },
        "validation_notice": (
            "Development runtime until fresh Kaggle TPU "
            "evidence closes G2-G5."
        ),
    }

def parse_restart_options(data, config):
    wait = data.get("wait_for_jobs", True)
    if not isinstance(wait, bool):
        raise ValueError(
            "wait_for_jobs must be boolean"
        )
    try:
        timeout = float(
            data.get("timeout", config.shutdown_timeout)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "timeout must be numeric"
        ) from exc
    if timeout <= 0:
        raise ValueError("timeout must be > 0")
    return wait, min(timeout, 900.0)

def create_app(runtime: Runtime):
    from flask import Flask, g, jsonify, request
    from werkzeug.exceptions import (
        HTTPException,
        RequestEntityTooLarge,
    )

    config = runtime.config
    manager = runtime.manager
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = (
        config.max_request_bytes
    )

    @app.before_request
    def begin():
        g.request_id = normalize_request_id(
            request.headers.get("X-Request-ID")
        )
        g.request_started = time.monotonic()
        g.job_id = None

    @app.after_request
    def complete(response):
        response.headers["X-Request-ID"] = (
            g.request_id
        )
        logger.info(json.dumps({
            "event":"http_request",
            "request_id":g.request_id,
            "method":request.method,
            "path":request.path,
            "status":response.status_code,
            "elapsed_ms":round(
                (time.monotonic()-g.request_started)*1000,
                3,
            ),
            "job_id":g.job_id,
        }, separators=(",",":"), sort_keys=True))
        return response

    def provided_key():
        auth = request.headers.get(
            "Authorization",""
        )
        if auth.lower().startswith("bearer "):
            return auth[7:].strip()
        return request.headers.get(
            "X-API-Key",""
        ).strip()

    def valid_key():
        if not config.api_auth_required:
            return True
        value = provided_key()
        return bool(
            value
            and hmac.compare_digest(
                value, config.api_key
            )
        )

    def auth_required(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            if not valid_key():
                return jsonify(
                    {"error":"Unauthorized"}
                ), 401
            return fn(*args, **kwargs)
        return wrapped

    def json_body():
        if not request.is_json:
            raise ValidationError(
                "Content-Type must be application/json"
            )
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            raise ValidationError(
                "Request body must be a JSON object"
            )
        return data

    def image_body():
        if request.is_json:
            return parse_image_payload(
                json_body(), config
            )
        if request.mimetype == "multipart/form-data":
            upload = request.files.get("image")
            if upload is None:
                raise ValidationError(
                    "Multipart field 'image' is required"
                )
            binary = upload.stream.read(
                config.max_image_bytes + 1
            )
            return parse_image_binary(
                binary,
                request.form.to_dict(flat=True),
                config,
            )
        raise ValidationError(
            "Content-Type must be application/json "
            "or multipart/form-data"
        )

    def submit(
        factory: Callable[[],dict],
        async_mode: bool,
    ):
        try:
            payload = factory()
            job = manager.submit(
                payload, request_id=g.request_id
            )
            g.job_id = job.id
        except ValidationError as exc:
            return jsonify({
                "error":str(exc),
                "request_id":g.request_id,
            }), 400
        except QueueFullError as exc:
            return jsonify({
                "error":str(exc),
                "request_id":g.request_id,
            }), 429
        except (
            WorkerNotReadyError,
            StoreFullError,
        ) as exc:
            health = (
                getattr(exc, "health", None)
                or manager.health()
            )
            return jsonify({
                "error":str(exc),
                "request_id":g.request_id,
                "state":health.get("state"),
                "retry_after_seconds":30,
            }), 503, {"Retry-After":"30"}

        if async_mode:
            return jsonify({
                "job_id":job.id,
                "request_id":job.request_id,
                "status":job.status,
                "result_url":f"/result/{job.id}",
            }), 202

        if not job.done.wait(
            timeout=config.request_timeout
        ):
            return jsonify({
                "job_id":job.id,
                "request_id":job.request_id,
                "status":"processing",
                "result_url":f"/result/{job.id}",
            }), 202

        return (
            jsonify(job.public_dict()),
            500 if job.status == "failed" else 200,
        )

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_exc):
        return jsonify({
            "error":"Request body is too large",
            "request_id":g.request_id,
        }), 413

    @app.errorhandler(Exception)
    def unhandled(exc):
        if isinstance(exc, HTTPException):
            return jsonify({
                "error":exc.description,
                "request_id":g.request_id,
            }), exc.code
        logger.exception(
            "Unhandled error request_id=%s",
            g.request_id,
        )
        return jsonify({
            "error":"Internal server error",
            "request_id":g.request_id,
        }), 500

    @app.get("/")
    def index():
        return jsonify({
            "service":"Gemma 4 31B Instruct",
            "api_version":API_VERSION,
            "runtime_validation":"NOT_YET_PROVEN",
            "endpoints":[
                "/health/live",
                "/health/ready",
                "/info",
                "/generate",
                "/generate/async",
                "/generate/image",
                "/generate/image/async",
                "/result/<job_id>",
                "/restart",
            ],
        })

    @app.get("/health/live")
    def live():
        return jsonify({
            "status":"alive",
            "pid":os.getpid(),
        })

    @app.get("/health")
    @app.get("/health/ready")
    def ready():
        health = manager.health()
        return (
            jsonify(health),
            200 if health["ready"] else 503,
        )

    @app.get("/info")
    @auth_required
    def info():
        return jsonify(
            build_info_payload(manager.health())
        )

    @app.post("/generate")
    @auth_required
    def generate():
        return submit(
            lambda: parse_generate_payload(
                json_body(), config
            ),
            False,
        )

    @app.post("/generate/async")
    @auth_required
    def generate_async():
        return submit(
            lambda: parse_generate_payload(
                json_body(), config
            ),
            True,
        )

    @app.post("/generate/image")
    @auth_required
    def generate_image():
        return submit(image_body, False)

    @app.post("/generate/image/async")
    @auth_required
    def generate_image_async():
        return submit(image_body, True)

    @app.get("/result/<job_id>")
    @auth_required
    def result(job_id):
        g.job_id = job_id
        job = manager.store.get(job_id)
        if not job:
            return jsonify({
                "error":"Job not found or expired",
                "request_id":g.request_id,
            }), 404
        if job.status in {"queued","processing"}:
            return jsonify(
                job.public_dict(False)
            ), 202
        return (
            jsonify(job.public_dict()),
            500 if job.status == "failed" else 200,
        )

    @app.post("/restart")
    @auth_required
    def restart():
        provided = request.headers.get(
            "X-Restart-Secret",""
        ).strip()
        if (
            not provided
            or not hmac.compare_digest(
                provided, config.restart_secret
            )
        ):
            return jsonify(
                {"error":"Unauthorized"}
            ), 401

        data = request.get_json(
            silent=True
        ) or {}
        try:
            wait, timeout = parse_restart_options(
                data, config
            )
        except ValueError as exc:
            return jsonify(
                {"error":str(exc)}
            ), 400

        if not runtime.restart_lock.acquire(
            blocking=False
        ):
            return jsonify({
                "error":"Restart already in progress"
            }), 409

        def run_restart():
            try:
                manager.restart_worker(
                    wait, timeout
                )
            finally:
                runtime.restart_lock.release()

        threading.Thread(
            target=run_restart,
            daemon=False,
            name="http-restart",
        ).start()
        return jsonify({
            "status":"restarting",
            "pid":os.getpid(),
        }), 202

    return app
