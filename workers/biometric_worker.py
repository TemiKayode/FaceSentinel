"""
biometric_worker.py
===================
Consent-based, user-initiated biometric authentication microservice.

Architecture
------------
  User presents badge + face at a terminal
  -> Kafka  auth.requests  (transaction_id + user_claimed_id + face image)
  -> ArcFace  1:1 verification against the single enrolled template in Qdrant
  -> Kafka  auth.responses  (GRANTED | DENIED + confidence + audit fields)
  -> Kafka  auth.audit      (structured audit event for every decision)

Design principles
-----------------
* 1:1 verification ONLY - the worker never performs an open 1:N identity
  search.  It retrieves exactly one enrolled record (filtered by
  user_claimed_id) and computes a single similarity score.
* Explicit transaction boundary - every auth attempt has a unique
  transaction_id created by the calling terminal, not by this service.
* No passive ingestion - the worker is idle until a user-initiated
  auth.request arrives.
* Zero raw-image retention - the JPEG bytes are decoded in-process,
  used for inference, then zeroed before the frame is garbage-collected.
* Rate limiting per terminal - prevents credential-stuffing / replay
  attacks at the Kafka consumer level (in-process sliding window;
  back this with Redis for multi-pod consistency in production).
* Consent lifecycle - enrolled records carry revoked_at; any revoked
  user is denied immediately before vector comparison.
* Enrollment is a separate, audited service (not this worker).
  This worker is read-only against the enrolled_faces collection.
"""

from __future__ import annotations

import asyncio
import base64
import ctypes
import hashlib
import hmac as _hmac_module
import json
import logging
import os
import signal
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Event, Thread
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from confluent_kafka import Consumer, KafkaError, KafkaException, Producer
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroDeserializer, AvroSerializer
from confluent_kafka.serialization import MessageField, SerializationContext
from insightface.app import FaceAnalysis
from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

logger = logging.getLogger("auth.worker")

# ---------------------------------------------------------------------------
# Avro schemas (in production, fetch from Schema Registry)
# ---------------------------------------------------------------------------

AUTH_REQUEST_SCHEMA = """
{
  "type": "record",
  "name": "AuthRequest",
  "namespace": "com.company.biometric",
  "fields": [
    {"name": "transaction_id",       "type": "string"},
    {"name": "terminal_id",          "type": "string"},
    {"name": "user_claimed_id",      "type": "string"},
    {"name": "image_b64",            "type": "string",
     "doc": "Base64-encoded JPEG, max 512 KB, pre-cropped to face ROI by terminal"},
    {"name": "request_ts",           "type": "long",
     "doc": "Unix timestamp milliseconds (UTC) when user initiated the request"},
    {"name": "secondary_factor_token", "type": ["null", "string"], "default": null,
     "doc": "TOTP code, PIN hash, or hardware token response for ENHANCED/CRITICAL tier MFA"}
  ]
}
"""

AUTH_RESPONSE_SCHEMA = """
{
  "type": "record",
  "name": "AuthResponse",
  "namespace": "com.company.biometric",
  "fields": [
    {"name": "transaction_id",  "type": "string"},
    {"name": "terminal_id",     "type": "string"},
    {"name": "user_claimed_id", "type": "string"},
    {"name": "decision",        "type": {"type": "enum", "name": "Decision",
                                 "symbols": ["GRANTED", "DENIED"]}},
    {"name": "confidence",      "type": "float",   "default": 0.0},
    {"name": "denial_reason",   "type": ["null", "string"], "default": null},
    {"name": "processing_ms",   "type": "long"},
    {"name": "worker_region",   "type": "string"},
    {"name": "response_ts",     "type": "long"}
  ]
}
"""


# ===========================================================================
# Configuration
# ===========================================================================

@dataclass
class WorkerConfig:
    """
    All configuration is injected via environment variables so the container
    image is immutable across environments.
    """
    region: str                 = field(default_factory=lambda: os.environ["REGION"])

    # -- Kafka ----------------------------------------------------------------
    kafka_bootstrap: str        = field(default_factory=lambda: os.environ["KAFKA_BOOTSTRAP_SERVERS"])
    kafka_request_topic: str    = field(default_factory=lambda: os.getenv("KAFKA_REQUEST_TOPIC",  "auth.requests"))
    kafka_response_topic: str   = field(default_factory=lambda: os.getenv("KAFKA_RESPONSE_TOPIC", "auth.responses"))
    kafka_dlq_topic: str        = field(default_factory=lambda: os.getenv("KAFKA_DLQ_TOPIC",      "auth.requests.dlq"))
    kafka_audit_topic: str      = field(default_factory=lambda: os.getenv("KAFKA_AUDIT_TOPIC",    "auth.audit"))
    kafka_group_id: str         = field(default_factory=lambda: os.getenv("KAFKA_GROUP_ID",       "auth-worker"))
    kafka_security_protocol: str = field(default_factory=lambda: os.getenv("KAFKA_SECURITY_PROTOCOL", "SASL_SSL"))
    kafka_username: str         = field(default_factory=lambda: os.getenv("KAFKA_USERNAME", ""))
    kafka_password: str         = field(default_factory=lambda: os.getenv("KAFKA_PASSWORD", ""))
    schema_registry_url: str    = field(default_factory=lambda: os.environ["KAFKA_SCHEMA_REGISTRY_URL"])

    # -- Qdrant ---------------------------------------------------------------
    qdrant_host: str            = field(default_factory=lambda: os.environ["QDRANT_HOST"])
    qdrant_api_key: str         = field(default_factory=lambda: os.environ["QDRANT_API_KEY"])
    qdrant_https: bool          = field(default_factory=lambda: os.getenv("QDRANT_HTTPS", "false").lower() == "true")
    enrolled_collection: str    = field(default_factory=lambda: os.getenv("QDRANT_ENROLLED_COLLECTION", "enrolled_faces"))
    # Optional second collection for dual-index reads during model version migration.
    # Set QDRANT_ENROLLED_COLLECTION_V2 to the new collection name during rollover;
    # unset once all embeddings have been re-enrolled against the new model.
    enrolled_collection_v2: str = field(default_factory=lambda: os.getenv("QDRANT_ENROLLED_COLLECTION_V2", ""))
    # Per-tenant isolation: if TENANT_ID is set, appends _{tenant_id} to collection names.
    tenant_id: str              = field(default_factory=lambda: os.getenv("TENANT_ID", ""))

    # -- Model identity -------------------------------------------------------
    model_name: str             = field(default_factory=lambda: os.getenv("MODEL_NAME",    "buffalo_l"))
    model_version: str          = field(default_factory=lambda: os.getenv("MODEL_VERSION", "insightface-0.7.3"))

    # -- Inference ------------------------------------------------------------
    match_threshold: float      = field(default_factory=lambda: float(os.getenv("MATCH_THRESHOLD", "0.75")))
    det_score_min: float        = field(default_factory=lambda: float(os.getenv("DET_SCORE_MIN",   "0.85")))
    inference_timeout_s: float  = field(default_factory=lambda: float(os.getenv("INFERENCE_TIMEOUT_S", "10.0")))
    liveness_enabled: bool      = field(default_factory=lambda: os.getenv("LIVENESS_CHECK_ENABLED", "false").lower() == "true")
    model_det_size: Tuple[int, int] = (640, 640)
    max_image_bytes: int        = 524_288   # 512 KB hard cap on incoming image

    # -- Batching -------------------------------------------------------------
    batch_size: int             = field(default_factory=lambda: int(os.getenv("WORKER_BATCH_SIZE",       "16")))
    batch_timeout_ms: int       = field(default_factory=lambda: int(os.getenv("WORKER_BATCH_TIMEOUT_MS", "100")))

    # -- Rate limiting --------------------------------------------------------
    rate_limit_max: int         = field(default_factory=lambda: int(os.getenv("RATE_LIMIT_MAX",      "10")))
    rate_limit_window_s: int    = field(default_factory=lambda: int(os.getenv("RATE_LIMIT_WINDOW_S", "30")))

    # -- Health server --------------------------------------------------------
    health_port: int            = field(default_factory=lambda: int(os.getenv("HEALTH_PORT", "8080")))

    # -- Intelligence / security hardening -----------------------------------
    # HMAC-SHA256 key for tamper-evident audit event signing.
    # Set AUDIT_HMAC_SECRET to a random 32+ byte hex string in production.
    hmac_secret: str            = field(default_factory=lambda: os.getenv("AUDIT_HMAC_SECRET", ""))
    anomaly_detection_enabled: bool = field(
        default_factory=lambda: os.getenv("ANOMALY_DETECTION_ENABLED", "true").lower() == "true"
    )
    # InsightFace model weights root. None -> InsightFace default (~/.insightface).
    # In K8s, mount the PVC at /models/insightface and set INSIGHTFACE_ROOT=/models/insightface.
    model_root: Optional[str]   = field(default_factory=lambda: os.getenv("INSIGHTFACE_ROOT") or None)

    @property
    def effective_collection(self) -> str:
        """Collection name with optional tenant suffix."""
        if self.tenant_id:
            return f"{self.enrolled_collection}_{self.tenant_id}"
        return self.enrolled_collection

    @property
    def effective_collection_v2(self) -> str:
        """V2 migration collection name with optional tenant suffix."""
        if not self.enrolled_collection_v2:
            return ""
        if self.tenant_id:
            return f"{self.enrolled_collection_v2}_{self.tenant_id}"
        return self.enrolled_collection_v2

    def __post_init__(self) -> None:
        if not (0.0 < self.match_threshold <= 1.0):
            raise ValueError(f"MATCH_THRESHOLD must be in (0, 1], got {self.match_threshold}")
        if not (0.0 < self.det_score_min <= 1.0):
            raise ValueError(f"DET_SCORE_MIN must be in (0, 1], got {self.det_score_min}")
        if self.batch_size < 1:
            raise ValueError(f"WORKER_BATCH_SIZE must be >= 1, got {self.batch_size}")
        if self.rate_limit_max < 1:
            raise ValueError(f"RATE_LIMIT_MAX must be >= 1, got {self.rate_limit_max}")
        if self.inference_timeout_s <= 0:
            raise ValueError(f"INFERENCE_TIMEOUT_S must be > 0, got {self.inference_timeout_s}")
        if self.kafka_security_protocol == "SASL_SSL" and (not self.kafka_username or not self.kafka_password):
            raise ValueError(
                "KAFKA_USERNAME and KAFKA_PASSWORD must be set when KAFKA_SECURITY_PROTOCOL=SASL_SSL"
            )


# ===========================================================================
# Security tier system
# ===========================================================================

class SecurityTier(str, Enum):
    """
    Per-user security clearance level stored in the Qdrant enrollment payload.
    Controls match threshold, enrollment freshness limit, and MFA requirements.

    STANDARD  - offices, cafeterias, car parks
    ENHANCED  - server rooms, labs, sensitive data areas
    CRITICAL  - SCIFs, weapons storage, executive protection, TS/SCI access
    """
    STANDARD = "standard"
    ENHANCED = "enhanced"
    CRITICAL = "critical"

# Base cosine similarity thresholds by tier
_TIER_THRESHOLDS: Dict[str, float] = {
    "standard": 0.75,
    "enhanced": 0.82,
    "critical": 0.88,
}

# Elevated thresholds for users with twin_risk_flagged=True.
# Identical twins share ~99.7% of face geometry; raising the threshold to 0.96+
# forces the system to demand near-perfect pixel-level match, discriminating on
# the ~0.3% of micro-geometry (nose cartilage shape, fine skin texture) that
# differs. Combine with multi-angle enrollment averaging for best results.
_TWIN_RISK_THRESHOLDS: Dict[str, float] = {
    "standard": 0.88,
    "enhanced": 0.92,
    "critical": 0.96,
}

# Maximum enrollment age in days before re-enrollment is required (0 = unlimited)
_TIER_MAX_ENROLLMENT_AGE: Dict[str, int] = {
    "standard": 0,    # no expiry
    "enhanced": 180,  # 6 months
    "critical": 90,   # 3 months
}


# ===========================================================================
# Decision enum
# ===========================================================================

class Decision(str, Enum):
    GRANTED = "GRANTED"
    DENIED  = "DENIED"


# ===========================================================================
# Auth result value object
# ===========================================================================

@dataclass
class AuthResult:
    transaction_id:  str
    terminal_id:     str
    user_claimed_id: str
    decision:        Decision
    confidence:      float         = 0.0
    denial_reason:   Optional[str] = None
    processing_ms:   int           = 0
    worker_region:   str           = ""
    response_ts:     int           = field(default_factory=lambda: _now_ms())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transaction_id":  self.transaction_id,
            "terminal_id":     self.terminal_id,
            "user_claimed_id": self.user_claimed_id,
            "decision":        self.decision.value,
            "confidence":      round(self.confidence, 6),
            "denial_reason":   self.denial_reason,
            "processing_ms":   self.processing_ms,
            "worker_region":   self.worker_region,
            "response_ts":     self.response_ts,
        }


def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


# ===========================================================================
# Rate limiter (in-process sliding window)
# ===========================================================================

class TerminalRateLimiter:
    """
    Per-terminal sliding-window rate limiter.

    Note: in-process only. In a multi-pod deployment, back this with a Redis
    ZSET (ZADD + ZREMRANGEBYSCORE + ZCARD) for cluster-wide enforcement.
    """

    def __init__(self, max_attempts: int, window_seconds: int) -> None:
        self._max = max_attempts
        self._window = window_seconds
        self._buckets: Dict[str, Deque[float]] = defaultdict(deque)

    def is_allowed(self, terminal_id: str) -> bool:
        now = time.monotonic()
        window_start = now - self._window
        bucket = self._buckets[terminal_id]
        while bucket and bucket[0] < window_start:
            bucket.popleft()
        if len(bucket) >= self._max:
            logger.warning(
                "Rate limit exceeded for terminal '%s' (%d attempts in %ds window).",
                terminal_id, len(bucket), self._window,
            )
            return False
        bucket.append(now)
        return True


# ===========================================================================
# Behavioral anomaly detector
# ===========================================================================

class AnomalyDetector:
    """
    Lightweight behavioral anomaly detection for access patterns.

    Flags three threat classes:
    1. Off-hours access (00:00-05:00 UTC) -- unusual for most facilities.
    2. Physically-impossible rapid multi-terminal access -- the same identity
       appearing at two different terminals within 60 seconds suggests cloned
       credentials or a deepfake attack sourced from video of the target.
    3. Repeated denial bursts -- >3 DENIED decisions in 5 minutes on the same
       (user, terminal) pair indicates a sustained impersonation attempt.

    Returns a float anomaly score [0.0, 1.0].  Values >= 0.5 are flagged in
    the auth.audit record and should trigger a security alert downstream.
    """

    def __init__(self, enabled: bool) -> None:
        self._enabled = enabled
        # user_id -> (terminal_id, wall_clock_time)
        self._last_seen: Dict[str, Tuple[str, float]] = {}
        # "user:terminal" -> deque of recent DENIED timestamps
        self._recent_denials: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=100))

    def score(
        self,
        user_id: str,
        terminal_id: str,
        decision: "Decision",
        wall_ts: float,
    ) -> float:
        """Compute anomaly score; update internal state as a side-effect."""
        if not self._enabled:
            return 0.0

        anomaly = 0.0
        hour = datetime.utcfromtimestamp(wall_ts).hour

        # 1. Off-hours window
        if 0 <= hour < 5:
            anomaly += 0.3

        # 2. Physically-impossible multi-terminal access
        if user_id in self._last_seen:
            prev_terminal, prev_ts = self._last_seen[user_id]
            elapsed = wall_ts - prev_ts
            if prev_terminal != terminal_id and elapsed < 60:
                anomaly += 0.5
                logger.warning(
                    "Anomaly: user '%s' at '%s' only %.0fs after '%s' -- possible relay attack.",
                    user_id, terminal_id, elapsed, prev_terminal,
                )

        # 3. Repeated denial burst
        key = f"{user_id}:{terminal_id}"
        denials = self._recent_denials[key]
        cutoff = wall_ts - 300.0  # 5-minute window
        while denials and denials[0] < cutoff:
            denials.popleft()
        if decision == Decision.DENIED:
            denials.append(wall_ts)
        burst = len(denials)
        if burst >= 3:
            anomaly += 0.2 * min(burst / 5.0, 1.0)

        self._last_seen[user_id] = (terminal_id, wall_ts)
        return min(anomaly, 1.0)


# ===========================================================================
# Passive liveness checker (MiniFAS via DeepFace -- optional)
# ===========================================================================

class LivenessChecker:
    """
    Optional MiniFAS (Minimalist Face Anti-Spoofing) passive liveness detection.

    Detects whether a face is from a real live person or a spoof artifact
    (printed photo, screen replay, silicone mask). Requires deepface>=0.0.93.

    Enable with LIVENESS_CHECK_ENABLED=true.  If deepface is not installed,
    liveness checks are skipped with a warning.
    """

    def __init__(self) -> None:
        self._deepface: Any = None
        try:
            from deepface import DeepFace as _DF  # type: ignore
            self._deepface = _DF
            logger.info("LivenessChecker: MiniFAS (deepface) loaded.")
        except ImportError:
            logger.warning(
                "LivenessChecker: deepface not installed -- liveness checks skipped. "
                "pip install deepface to enable."
            )

    @property
    def available(self) -> bool:
        return self._deepface is not None

    def check(self, frame: np.ndarray) -> Tuple[bool, float]:
        """
        Run MiniFAS on a BGR frame.

        Returns
        -------
        (is_real, antispoof_score)
        is_real=False means the frame is a probable spoof artifact.
        Returns (True, 1.0) when deepface is not installed (fail-open).
        """
        if self._deepface is None:
            return True, 1.0

        faces = self._deepface.extract_faces(
            img_path=frame,
            anti_spoofing=True,
            enforce_detection=False,
        )
        if not faces:
            return True, 1.0
        score = float(faces[0].get("antispoof_score", 1.0))
        is_real = bool(faces[0].get("is_real", True))
        return is_real, score


# ===========================================================================
# ArcFace embedding model
# ===========================================================================

class EmbeddingModel:
    """
    Wraps InsightFace buffalo_l (ArcFace R100 + RetinaFace) on a single GPU.
    Thread-safe for read inference; model weights are loaded once at startup.
    """

    def __init__(self, cfg: WorkerConfig) -> None:
        self._cfg = cfg
        self._app: Optional[FaceAnalysis] = None
        self._ready = False

    def load(self) -> None:
        gpu_available = torch.cuda.is_available()
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if gpu_available
            else ["CPUExecutionProvider"]
        )
        kwargs: Dict[str, Any] = {"name": self._cfg.model_name, "providers": providers}
        if self._cfg.model_root:
            kwargs["root"] = self._cfg.model_root
        self._app = FaceAnalysis(**kwargs)
        ctx_id = 0 if gpu_available else -1  # 0=first GPU, -1=CPU
        self._app.prepare(ctx_id=ctx_id, det_size=self._cfg.model_det_size)
        self._ready = True
        device = "GPU" if gpu_available else "CPU"
        logger.info("ArcFace model '%s' loaded on %s.", self._cfg.model_name, device)

    @property
    def is_ready(self) -> bool:
        return self._ready

    def extract(self, frame: np.ndarray) -> Optional[Tuple[str, List[float]]]:
        """
        Run detection + ArcFace embedding on a single BGR frame.

        Returns (face_hash, unit_vector) if a qualifying face is found, else None.
        The face_hash is the SHA-256 prefix of raw pixel bytes for audit logging only.
        """
        faces = self._app.get(frame)
        if not faces:
            return None

        best = max(faces, key=lambda f: float(f.det_score))
        if float(best.det_score) < self._cfg.det_score_min:
            logger.debug(
                "Detection score %.3f below threshold %.3f - discarded.",
                float(best.det_score), self._cfg.det_score_min,
            )
            return None

        raw: np.ndarray = best.embedding.astype(np.float32)
        norm = np.linalg.norm(raw)
        if norm > 1e-10:
            raw /= norm

        face_hash = hashlib.sha256(frame.tobytes()).hexdigest()[:16]
        return face_hash, raw.tolist()


# ===========================================================================
# Qdrant 1:1 verifier
# ===========================================================================

class EnrolledFaceVerifier:
    """
    Performs 1:1 biometric verification against Qdrant.

    Supports:
    - Consent revocation: enrolled records with revoked_at are denied immediately.
    - Dual-index reads: if enrolled_collection_v2 is configured, the worker
      checks both collections during a model migration and takes the higher score.
    - Per-tenant isolation: collection names are suffixed with tenant_id.
    """

    def __init__(self, cfg: WorkerConfig) -> None:
        self._cfg = cfg
        self._client: Optional[AsyncQdrantClient] = None

    async def connect(self) -> None:
        self._client = AsyncQdrantClient(
            host=self._cfg.qdrant_host,
            api_key=self._cfg.qdrant_api_key,
            https=self._cfg.qdrant_https,
            timeout=5,
            check_compatibility=False,
        )
        # Ping to confirm connectivity (raises on failure)
        await self._client.get_collections()
        logger.info("Connected to Qdrant at %s.", self._cfg.qdrant_host)

    async def verify(
        self,
        user_claimed_id: str,
        presented_vector: List[float],
        secondary_factor_token: Optional[str] = None,
    ) -> Tuple[float, Decision, Optional[str]]:
        """
        1:1 verification against the primary (and optionally v2) collection.

        Returns (confidence_score, decision, denial_reason).
        """
        result = await self._verify_collection(
            self._cfg.effective_collection, user_claimed_id,
            presented_vector, secondary_factor_token,
        )

        # Dual-index: during model migration, also check v2 collection and
        # take whichever enrolled record gives a higher similarity score.
        if self._cfg.effective_collection_v2:
            try:
                v2 = await self._verify_collection(
                    self._cfg.effective_collection_v2, user_claimed_id,
                    presented_vector, secondary_factor_token,
                )
                if v2[0] > result[0]:
                    logger.debug(
                        "Dual-index: v2 score %.4f > v1 score %.4f for user '%s'.",
                        v2[0], result[0], user_claimed_id,
                    )
                    result = v2
            except Exception as exc:
                logger.warning("Dual-index v2 lookup failed (non-fatal): %s", exc)

        return result

    async def _verify_collection(
        self,
        collection_name: str,
        user_claimed_id: str,
        presented_vector: List[float],
        secondary_factor_token: Optional[str] = None,
    ) -> Tuple[float, Decision, Optional[str]]:
        records, _ = await self._client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(
                must=[FieldCondition(key="user_id", match=MatchValue(value=user_claimed_id))]
            ),
            with_vectors=True,
            limit=1,
        )

        if not records:
            logger.info("User '%s' not found in '%s'.", user_claimed_id, collection_name)
            return 0.0, Decision.DENIED, "USER_NOT_ENROLLED"

        record = records[0]
        payload = record.payload or {}

        # -- Consent revocation check ----------------------------------------
        if payload.get("revoked_at"):
            logger.info(
                "User '%s' consent revoked at %s.",
                user_claimed_id, payload["revoked_at"],
            )
            return 0.0, Decision.DENIED, "CONSENT_REVOKED"

        # -- Security tier + threshold selection -----------------------------
        tier_raw = payload.get("security_clearance_level", "standard")
        tier_thresholds = _TWIN_RISK_THRESHOLDS if payload.get("twin_risk_flagged") else _TIER_THRESHOLDS
        tier_threshold = tier_thresholds.get(tier_raw, self._cfg.match_threshold)
        # Always enforce the stricter of the global config and per-user tier threshold
        threshold = max(self._cfg.match_threshold, tier_threshold)

        # -- Enrollment freshness enforcement --------------------------------
        max_age = _TIER_MAX_ENROLLMENT_AGE.get(tier_raw, 0)
        if max_age > 0:
            enrolled_at_str = payload.get("enrolled_at", "")
            if enrolled_at_str:
                try:
                    enrolled_dt = datetime.fromisoformat(enrolled_at_str.replace("Z", "+00:00"))
                    age_days = (datetime.now(timezone.utc) - enrolled_dt).days
                    if age_days > max_age:
                        logger.warning(
                            "User '%s' enrollment %d days old (max %d for %s tier).",
                            user_claimed_id, age_days, max_age, tier_raw,
                        )
                        return 0.0, Decision.DENIED, f"ENROLLMENT_EXPIRED"
                except ValueError:
                    pass

        # -- Multi-factor authentication check -------------------------------
        if payload.get("require_secondary_factor"):
            if not secondary_factor_token:
                logger.info("User '%s' requires secondary factor but none provided.", user_claimed_id)
                return 0.0, Decision.DENIED, "SECONDARY_FACTOR_REQUIRED"
            # Validate against stored SHA-256 hash of the credential.
            # In production: replace with TOTP validation (pyotp.TOTP(secret).verify(token))
            expected_hash = payload.get("secondary_factor_hash", "")
            if expected_hash:
                provided_hash = hashlib.sha256(secondary_factor_token.encode()).hexdigest()
                if provided_hash != expected_hash:
                    logger.info("User '%s' secondary factor invalid.", user_claimed_id)
                    return 0.0, Decision.DENIED, "SECONDARY_FACTOR_INVALID"

        # -- Model version mismatch warning ----------------------------------
        enrolled_model = payload.get("model_name", "unknown")
        if enrolled_model != self._cfg.model_name:
            logger.warning(
                "Model mismatch for user '%s': enrolled='%s', worker='%s'. "
                "Re-enroll or configure QDRANT_ENROLLED_COLLECTION_V2.",
                user_claimed_id, enrolled_model, self._cfg.model_name,
            )

        # -- Cosine similarity (both vectors unit-normalised -> dot = cosine) -
        enrolled_vector: List[float] = record.vector  # type: ignore[assignment]
        pv = np.array(presented_vector, dtype=np.float32)
        ev = np.array(enrolled_vector,  dtype=np.float32)
        score = float(np.dot(pv, ev))

        twin_note = " [twin-risk threshold]" if payload.get("twin_risk_flagged") else ""
        logger.debug(
            "User '%s' score=%.4f threshold=%.4f tier=%s%s",
            user_claimed_id, score, threshold, tier_raw, twin_note,
        )

        if score >= threshold:
            return score, Decision.GRANTED, None
        return score, Decision.DENIED, "BELOW_THRESHOLD"


# ===========================================================================
# Kafka transport
# ===========================================================================

class AuthRequestConsumer:
    """Pulls auth request events from Kafka with at-least-once delivery."""

    def __init__(self, cfg: WorkerConfig, deserializer: AvroDeserializer) -> None:
        self._cfg = cfg
        self._deserializer = deserializer

        consumer_conf: Dict[str, Any] = {
            "bootstrap.servers":    cfg.kafka_bootstrap,
            "group.id":             cfg.kafka_group_id,
            "auto.offset.reset":    "latest",
            "enable.auto.commit":   False,
            "max.poll.interval.ms": 60_000,
            "session.timeout.ms":   30_000,
            "security.protocol":    cfg.kafka_security_protocol,
        }
        if cfg.kafka_security_protocol == "SASL_SSL":
            consumer_conf.update({
                "sasl.mechanism": os.getenv("KAFKA_SASL_MECHANISM", "SCRAM-SHA-512"),
                "sasl.username":  cfg.kafka_username,
                "sasl.password":  cfg.kafka_password,
            })
        self._consumer = Consumer(consumer_conf)

    def subscribe(self) -> None:
        self._consumer.subscribe([self._cfg.kafka_request_topic])
        logger.info("Subscribed to '%s'.", self._cfg.kafka_request_topic)

    def poll_batch(self) -> List[Tuple[Any, Dict]]:
        """Collect up to batch_size messages within the batch_timeout_ms window."""
        batch: List[Tuple[Any, Dict]] = []
        deadline = time.monotonic() + self._cfg.batch_timeout_ms / 1000.0

        while len(batch) < self._cfg.batch_size and time.monotonic() < deadline:
            msg = self._consumer.poll(timeout=0.02)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                raise KafkaException(msg.error())
            event = self._deserializer(
                msg.value(),
                SerializationContext(self._cfg.kafka_request_topic, MessageField.VALUE),
            )
            batch.append((msg, event))

        return batch

    def commit(self, messages: List[Tuple[Any, Dict]]) -> None:
        if messages:
            self._consumer.commit(message=messages[-1][0], asynchronous=False)

    def close(self) -> None:
        self._consumer.close()


class AuthResponseProducer:
    """Publishes auth decisions to the response, audit, and DLQ topics."""

    def __init__(self, cfg: WorkerConfig, serializer: AvroSerializer) -> None:
        self._cfg = cfg
        self._serializer = serializer
        producer_conf: Dict[str, Any] = {
            "bootstrap.servers":   cfg.kafka_bootstrap,
            "acks":                "all",
            "retries":             5,
            "delivery.timeout.ms": 10_000,
            "security.protocol":   cfg.kafka_security_protocol,
        }
        if cfg.kafka_security_protocol == "SASL_SSL":
            producer_conf.update({
                "sasl.mechanism": os.getenv("KAFKA_SASL_MECHANISM", "SCRAM-SHA-512"),
                "sasl.username":  cfg.kafka_username,
                "sasl.password":  cfg.kafka_password,
            })
        self._producer = Producer(producer_conf)

    def publish_result(self, result: AuthResult) -> None:
        payload = self._serializer(
            result.to_dict(),
            SerializationContext(self._cfg.kafka_response_topic, MessageField.VALUE),
        )
        self._producer.produce(
            topic=self._cfg.kafka_response_topic,
            key=result.transaction_id.encode(),
            value=payload,
            on_delivery=self._on_delivery,
        )
        self._producer.poll(0)

    def publish_audit(self, audit_event: Dict[str, Any], hmac_secret: str = "") -> None:
        """
        Emit a tamper-evident JSON audit record to auth.audit.

        If hmac_secret is non-empty, appends _sig: HMAC-SHA256 of the
        canonical JSON payload so consumers can verify the record was not
        modified in transit or in the log store.
        """
        if hmac_secret:
            canonical = json.dumps(
                {k: v for k, v in sorted(audit_event.items())},
                separators=(",", ":"), sort_keys=True,
            ).encode()
            sig = _hmac_module.new(hmac_secret.encode(), canonical, "sha256").hexdigest()
            audit_event = {**audit_event, "_sig": sig}
        self._producer.produce(
            topic=self._cfg.kafka_audit_topic,
            key=audit_event.get("transaction_id", "").encode(),
            value=json.dumps(audit_event).encode(),
            on_delivery=self._on_delivery,
        )
        self._producer.poll(0)

    def publish_dlq(self, transaction_id: str, raw_event: Dict, reason: str) -> None:
        dlq_payload = json.dumps({
            "transaction_id": transaction_id,
            "original_event": raw_event,
            "failure_reason": reason,
            "failed_at": _now_ms(),
        }).encode()
        self._producer.produce(
            topic=self._cfg.kafka_dlq_topic,
            key=transaction_id.encode() if transaction_id else None,
            value=dlq_payload,
            on_delivery=self._on_delivery,
        )
        self._producer.poll(0)

    @staticmethod
    def _on_delivery(err: Any, msg: Any) -> None:
        if err:
            logger.error("Delivery failure for %s: %s", msg.key(), err)

    def flush(self) -> None:
        self._producer.flush(timeout=5)


# ===========================================================================
# Frame helpers
# ===========================================================================

def decode_image(b64_payload: str, max_bytes: int) -> Optional[np.ndarray]:
    """
    Decode a Base64-encoded JPEG/PNG into a BGR numpy array.

    Returns None if the payload exceeds max_bytes (hard limit enforced before
    decoding to prevent decompression bombs) or if OpenCV cannot decode it.
    """
    raw: bytes = base64.b64decode(b64_payload)
    if len(raw) > max_bytes:
        logger.warning(
            "Image payload %d bytes exceeds cap %d - rejected.",
            len(raw), max_bytes,
        )
        _zero_bytes(raw)
        return None

    buf = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    _zero_bytes(raw)

    if frame is None:
        logger.warning("OpenCV could not decode image payload.")
    return frame


def _zero_bytes(buf: bytes) -> None:
    """Best-effort in-process memory zeroing of raw byte buffers."""
    try:
        ctypes.memset(id(buf) + 32, 0, len(buf))
    except Exception as exc:
        logger.debug("Memory zeroing skipped: %s", exc)


def _zero_frame(frame: np.ndarray) -> None:
    frame[:] = 0


# ===========================================================================
# Health HTTP server (K8s liveness / readiness / startup probes)
# ===========================================================================

class HealthServer:
    """
    Minimal HTTP server exposing /healthz, /readyz, and /startupz in a
    daemon thread so it does not block the main async event loop.

    /readyz now requires both Kafka connection (ready Event) AND a successful
    Qdrant connection (qdrant_ok Event) before returning 200.
    """

    def __init__(self, port: int, model: EmbeddingModel) -> None:
        self._port      = port
        self._model     = model
        self._ready     = Event()
        self._qdrant_ok = Event()

    def mark_ready(self) -> None:
        self._ready.set()

    def mark_qdrant_ok(self) -> None:
        self._qdrant_ok.set()

    def mark_qdrant_failed(self) -> None:
        self._qdrant_ok.clear()

    def start(self) -> None:
        model     = self._model
        ready     = self._ready
        qdrant_ok = self._qdrant_ok

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/healthz":
                    self._respond(200, b"ok")
                elif self.path == "/readyz":
                    if model.is_ready and ready.is_set() and qdrant_ok.is_set():
                        self._respond(200, b"ready")
                    else:
                        self._respond(503, b"not ready")
                elif self.path == "/startupz":
                    if model.is_ready:
                        self._respond(200, b"started")
                    else:
                        self._respond(503, b"loading model")
                else:
                    self._respond(404, b"not found")

            def _respond(self, code: int, body: bytes) -> None:
                self.send_response(code)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args: Any) -> None:
                pass

        server = HTTPServer(("0.0.0.0", self._port), _Handler)
        thread = Thread(target=server.serve_forever, daemon=True, name="health-server")
        thread.start()
        logger.info("Health server listening on :%d", self._port)


# ===========================================================================
# Telemetry bootstrap
# ===========================================================================

def _setup_telemetry(service_name: str) -> Tuple[Any, Any, Any, Any, Any]:
    """
    Initialise OpenTelemetry tracing + metrics.
    Returns (tracer, frames_counter, decisions_counter, latency_histogram, errors_counter).
    Set OTEL_SDK_DISABLED=true to skip exporter setup in local dev (no collector).
    """
    sdk_disabled = os.getenv("OTEL_SDK_DISABLED", "false").lower() == "true"

    if sdk_disabled:
        tracer = trace.get_tracer(service_name)
        meter  = metrics.get_meter(service_name)
    else:
        otlp_endpoint = os.getenv(
            "OTEL_EXPORTER_OTLP_ENDPOINT",
            "http://otel-collector.observability.svc.cluster.local:4317",
        )
        tracer_provider = TracerProvider()
        tracer_provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint))
        )
        trace.set_tracer_provider(tracer_provider)
        tracer = trace.get_tracer(service_name)

        meter_provider = MeterProvider()
        metrics.set_meter_provider(meter_provider)
        meter = metrics.get_meter(service_name)

    frames_counter    = meter.create_counter("auth_requests_total",  description="Auth requests received")
    decisions_counter = meter.create_counter("auth_decisions_total", description="Auth decisions by outcome")
    latency_hist      = meter.create_histogram("auth_latency_ms",    description="End-to-end auth latency in ms")
    errors_counter    = meter.create_counter("auth_errors_total",    description="Processing errors")

    return tracer, frames_counter, decisions_counter, latency_hist, errors_counter


# ===========================================================================
# Main worker
# ===========================================================================

class BiometricAuthWorker:
    """
    Event-driven authentication worker.

    Per-request lifecycle:
      1.  Validate event schema fields
      2.  Rate-limit check per terminal_id
      3.  Decode + validate image (size cap, OpenCV)
      4.  [Optional] MiniFAS passive liveness check
      5.  ArcFace inference (timeout-guarded, per-stage OTel span)
      6.  Zero raw frame buffer
      7.  Qdrant 1:1 lookup + consent revocation check (per-stage OTel span)
      8.  Cosine similarity -> GRANTED / DENIED
      9.  Publish decision to auth.responses
     10.  Publish structured audit event to auth.audit
     11.  Commit Kafka offset
    """

    def __init__(self) -> None:
        self._cfg      = WorkerConfig()
        self._model    = EmbeddingModel(self._cfg)
        self._verifier = EnrolledFaceVerifier(self._cfg)
        self._limiter  = TerminalRateLimiter(
            max_attempts=self._cfg.rate_limit_max,
            window_seconds=self._cfg.rate_limit_window_s,
        )
        self._liveness: Optional[LivenessChecker] = (
            LivenessChecker() if self._cfg.liveness_enabled else None
        )
        self._anomaly = AnomalyDetector(self._cfg.anomaly_detection_enabled)
        self._shutdown = asyncio.Event()
        self._health   = HealthServer(self._cfg.health_port, self._model)

        (
            self._tracer,
            self._req_counter,
            self._dec_counter,
            self._lat_hist,
            self._err_counter,
        ) = _setup_telemetry(f"auth-worker-{self._cfg.region}")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def startup(self) -> None:
        self._health.start()

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._model.load)

        await self._verifier.connect()
        self._health.mark_qdrant_ok()

        schema_client = SchemaRegistryClient({"url": self._cfg.schema_registry_url})
        deserializer  = AvroDeserializer(schema_client, AUTH_REQUEST_SCHEMA)
        serializer    = AvroSerializer(schema_client, AUTH_RESPONSE_SCHEMA)

        self._consumer = AuthRequestConsumer(self._cfg, deserializer)
        self._producer = AuthResponseProducer(self._cfg, serializer)
        self._consumer.subscribe()

        signal.signal(signal.SIGTERM, lambda *_: self._shutdown.set())
        signal.signal(signal.SIGINT,  lambda *_: self._shutdown.set())

        self._health.mark_ready()
        logger.info(
            "Auth worker online - region=%s  collection=%s  request_topic=%s",
            self._cfg.region,
            self._cfg.effective_collection,
            self._cfg.kafka_request_topic,
        )

    async def run(self) -> None:
        await self.startup()
        try:
            while not self._shutdown.is_set():
                await self._process_batch()
        finally:
            self._consumer.close()
            self._producer.flush()
            logger.info("Auth worker shut down cleanly.")

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    async def _process_batch(self) -> None:
        messages = self._consumer.poll_batch()
        if not messages:
            return

        loop = asyncio.get_running_loop()
        tasks = [
            loop.create_task(self._handle_request(raw_msg, event))
            for raw_msg, event in messages
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, BaseException):
                logger.error("Unhandled error in request handler: %s", res, exc_info=res)
                self._err_counter.add(1, {"reason": "unhandled_exception"})
        self._consumer.commit(messages)

    # ------------------------------------------------------------------
    # Per-request handler
    # ------------------------------------------------------------------

    async def _handle_request(self, raw_msg: Any, event: Dict) -> None:
        t0              = time.monotonic()
        wall_ts         = time.time()
        tx              = event.get("transaction_id", str(uuid.uuid4()))
        terminal        = event.get("terminal_id",    "unknown")
        user            = event.get("user_claimed_id", "")
        secondary_token = event.get("secondary_factor_token") or None
        face_hash: Optional[str] = None
        anomaly_score: float = 0.0

        with self._tracer.start_as_current_span("handle_auth_request") as span:
            span.set_attribute("transaction.id",  tx)
            span.set_attribute("terminal.id",     terminal)
            span.set_attribute("user.claimed_id", user)
            self._req_counter.add(1, {"terminal": terminal})

            # -- 1. Validate required fields ---------------------------------
            missing = [
                f for f in ("transaction_id", "terminal_id", "user_claimed_id", "image_b64")
                if not event.get(f)
            ]
            if missing:
                logger.error("Malformed event - missing %s  tx=%s", missing, tx)
                self._producer.publish_dlq(tx, event, f"missing_fields:{missing}")
                self._err_counter.add(1, {"reason": "malformed_event"})
                return

            # -- 2. Rate limiting --------------------------------------------
            if not self._limiter.is_allowed(terminal):
                result = self._make_denied(event, t0, reason="RATE_LIMITED")
                self._publish_and_audit(result, face_hash)
                return

            # -- 3. Decode image ---------------------------------------------
            frame = decode_image(event["image_b64"], self._cfg.max_image_bytes)
            if frame is None:
                result = self._make_denied(event, t0, reason="IMAGE_DECODE_FAILURE")
                self._publish_and_audit(result, face_hash)
                return

            # -- 4. Passive liveness check (MiniFAS, optional) ---------------
            if self._liveness is not None:
                try:
                    is_real, liveness_score = self._liveness.check(frame)
                    span.set_attribute("liveness.score", liveness_score)
                    if not is_real:
                        logger.info(
                            "Liveness FAILED tx=%s terminal=%s score=%.3f",
                            tx, terminal, liveness_score,
                        )
                        result = self._make_denied(event, t0, reason="LIVENESS_FAILED")
                        self._publish_and_audit(result, face_hash)
                        return
                except Exception as exc:
                    logger.warning("Liveness check error tx=%s: %s -- skipping", tx, exc)

            # -- 5. ArcFace inference (timeout-guarded, OTel child span) ----
            loop = asyncio.get_running_loop()
            try:
                with self._tracer.start_as_current_span("arcface_inference") as infer_span:
                    infer_result = await asyncio.wait_for(
                        loop.run_in_executor(None, self._model.extract, frame),
                        timeout=self._cfg.inference_timeout_s,
                    )
                    infer_span.set_attribute("face.detected", infer_result is not None)
            except asyncio.TimeoutError:
                logger.error(
                    "Inference timed out after %.1fs  tx=%s",
                    self._cfg.inference_timeout_s, tx,
                )
                self._err_counter.add(1, {"reason": "inference_timeout"})
                result = self._make_denied(event, t0, reason="INFERENCE_TIMEOUT")
                self._publish_and_audit(result, face_hash)
                return
            except Exception as exc:
                logger.error("Inference error tx=%s: %s", tx, exc)
                self._err_counter.add(1, {"reason": "inference_error"})
                result = self._make_denied(event, t0, reason="INFERENCE_ERROR")
                self._publish_and_audit(result, face_hash)
                return
            finally:
                # -- 6. Zero raw frame immediately post-inference -----------
                _zero_frame(frame)

            if infer_result is None:
                result = self._make_denied(event, t0, reason="NO_FACE_DETECTED")
                self._publish_and_audit(result, face_hash)
                return

            face_hash, presented_vector = infer_result
            span.set_attribute("face.hash", face_hash)

            # -- 7. Qdrant 1:1 lookup + decision (OTel child span) ----------
            try:
                with self._tracer.start_as_current_span("qdrant_vector_search") as search_span:
                    score, decision, denial_reason = await self._verifier.verify(
                        user_claimed_id=user,
                        presented_vector=presented_vector,
                        secondary_factor_token=secondary_token,
                    )
                    search_span.set_attribute("auth.decision", decision.value)
                    search_span.set_attribute("auth.score",    score)
            except Exception as exc:
                logger.error("Qdrant verify error tx=%s: %s", tx, exc)
                self._health.mark_qdrant_failed()
                self._err_counter.add(1, {"reason": "qdrant_error"})
                result = self._make_denied(event, t0, reason="VERIFICATION_SERVICE_ERROR")
                self._publish_and_audit(result, face_hash)
                return

            elapsed_ms = int((time.monotonic() - t0) * 1000)
            result = AuthResult(
                transaction_id=tx,
                terminal_id=terminal,
                user_claimed_id=user,
                decision=decision,
                confidence=score,
                denial_reason=denial_reason,
                processing_ms=elapsed_ms,
                worker_region=self._cfg.region,
            )

            # -- 8+9. Publish response + audit event (with anomaly scoring) --
            anomaly_score = self._anomaly.score(user, terminal, decision, wall_ts)
            if anomaly_score >= 0.5:
                logger.warning(
                    "Anomaly score %.2f for tx=%s user=%s terminal=%s decision=%s",
                    anomaly_score, tx, user, terminal, decision.value,
                )
            self._publish_and_audit(result, face_hash, anomaly_score)

            # -- 10. Telemetry -----------------------------------------------
            self._dec_counter.add(1, {"decision": decision.value, "terminal": terminal})
            self._lat_hist.record(elapsed_ms, {"decision": decision.value})
            span.set_attribute("auth.decision",   decision.value)
            span.set_attribute("auth.score",      score)
            span.set_attribute("auth.latency_ms", elapsed_ms)

            logger.info(
                "tx=%s terminal=%s user=%s -> %s (score=%.4f, %dms)",
                tx, terminal, user, decision.value, score, elapsed_ms,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_denied(
        self,
        event: Dict,
        t0: float,
        reason: str,
    ) -> AuthResult:
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        return AuthResult(
            transaction_id=event.get("transaction_id", "unknown"),
            terminal_id=event.get("terminal_id",    "unknown"),
            user_claimed_id=event.get("user_claimed_id", "unknown"),
            decision=Decision.DENIED,
            confidence=0.0,
            denial_reason=reason,
            processing_ms=elapsed_ms,
            worker_region=self._cfg.region,
        )

    def _publish_and_audit(
        self,
        result: AuthResult,
        face_hash: Optional[str] = None,
        anomaly_score: float = 0.0,
    ) -> None:
        """Publish decision to auth.responses and HMAC-signed event to auth.audit."""
        try:
            self._producer.publish_result(result)
        except Exception as exc:
            logger.error("Failed to publish result tx=%s: %s", result.transaction_id, exc)

        try:
            self._producer.publish_audit(
                {
                    "transaction_id":  result.transaction_id,
                    "terminal_id":     result.terminal_id,
                    "user_claimed_id": result.user_claimed_id,
                    "decision":        result.decision.value,
                    "confidence":      round(result.confidence, 6),
                    "denial_reason":   result.denial_reason,
                    "face_hash":       face_hash,
                    "anomaly_score":   round(anomaly_score, 4),
                    "processing_ms":   result.processing_ms,
                    "worker_region":   result.worker_region,
                    "model_name":      self._cfg.model_name,
                    "model_version":   self._cfg.model_version,
                    "tenant_id":       self._cfg.tenant_id or None,
                    "event_ts":        _now_ms(),
                },
                hmac_secret=self._cfg.hmac_secret,
            )
        except Exception as exc:
            logger.error("Failed to publish audit tx=%s: %s", result.transaction_id, exc)


# ===========================================================================
# Entrypoint
# ===========================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
    )
    worker = BiometricAuthWorker()
    asyncio.run(worker.run())
