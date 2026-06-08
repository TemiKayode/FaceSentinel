"""
02_enroll_face.py
=================
Enroll a person's face into the Qdrant `enrolled_faces` collection.

This script acts as a minimal stand-in for the separate Enrollment Service.
It extracts an ArcFace embedding from a local image and upserts it directly
into Qdrant - bypassing Kafka (enrollment is a privileged, audited operation
separate from the authentication worker).

Quality gates applied at enrollment time
-----------------------------------------
  - Face count: exactly one face must be visible
  - Sharpness: Laplacian variance must exceed MIN_BLUR_SCORE (default 100)
  - Pose: |yaw| and |pitch| must not exceed MAX_POSE_ANGLE degrees (default 30)
  - Liveness: optional MiniFAS check via deepface (set LIVENESS_CHECK_ENABLED=true)

Consent lifecycle
-----------------
  Enroll  : creates record with consent_version, consent_method, consent_ts
  Revoke  : sets revoked_at; worker denies with CONSENT_REVOKED; audit trail intact
  Forget  : hard-deletes record from Qdrant (right-to-be-forgotten / GDPR Art 17)
            Note: Kafka auth.audit records are append-only and cannot be deleted.

Usage
-----
    # Enroll
    python scripts/02_enroll_face.py \\
        --user-id  "emp-001" \\
        --name     "Alice Smith" \\
        --image    "./test_faces/alice.jpg" \\
        --consent-version "1.0" \\
        --consent-method  "written" \\
        --metadata '{"department": "Engineering"}'

    # List enrolled users
    python scripts/02_enroll_face.py --list

    # Delete (hard delete, right-to-be-forgotten)
    python scripts/02_enroll_face.py --forget "emp-001"

    # Revoke consent (soft delete -- worker will deny, record kept for audit)
    python scripts/02_enroll_face.py --revoke "emp-001"

    # Per-tenant (collection becomes enrolled_faces_acme)
    python scripts/02_enroll_face.py --tenant acme --user-id emp-001 ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# -- Configuration (reads from env with local-dev defaults) --------------------
QDRANT_HOST    = os.getenv("QDRANT_HOST",       "localhost")
QDRANT_PORT    = int(os.getenv("QDRANT_PORT",   "6333"))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY",    "local-dev-key-change-in-prod")
_BASE_COLLECTION = os.getenv("QDRANT_ENROLLED_COLLECTION", "enrolled_faces")
EMBEDDING_DIM  = 512
DET_SCORE_MIN  = float(os.getenv("DET_SCORE_MIN",    "0.75"))
MIN_BLUR_SCORE = float(os.getenv("MIN_BLUR_SCORE",   "100.0"))
MAX_POSE_ANGLE = float(os.getenv("MAX_POSE_ANGLE",   "30.0"))
MODEL_NAME     = os.getenv("MODEL_NAME",    "buffalo_l")
MODEL_VERSION  = os.getenv("MODEL_VERSION", "insightface-0.7.3")
LIVENESS_ENABLED = os.getenv("LIVENESS_CHECK_ENABLED", "false").lower() == "true"


def _collection(tenant: Optional[str]) -> str:
    if tenant:
        return f"{_BASE_COLLECTION}_{tenant}"
    return _BASE_COLLECTION


# -- Qdrant client -------------------------------------------------------------

def get_client():
    from qdrant_client import QdrantClient  # type: ignore
    return QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
        api_key=QDRANT_API_KEY,
        https=False,
        timeout=10,
        check_compatibility=False,
    )


def ensure_collection(client, collection: str) -> None:
    from qdrant_client.models import Distance, VectorParams  # type: ignore
    existing = {c.name for c in client.get_collections().collections}
    if collection not in existing:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        print(f"  Created collection '{collection}'.")
    else:
        count = client.count(collection).count
        print(f"  Collection '{collection}' exists -- {count} enrolled face(s).")


# -- Liveness check (optional MiniFAS via DeepFace) ----------------------------

def _check_liveness(img) -> bool:
    """
    Run MiniFAS passive liveness check via DeepFace.
    Returns True (real) / False (spoof). Raises ImportError if deepface missing.
    """
    from deepface import DeepFace  # type: ignore
    faces = DeepFace.extract_faces(
        img_path=img,
        anti_spoofing=True,
        enforce_detection=False,
    )
    if not faces:
        return True
    is_real = bool(faces[0].get("is_real", True))
    score = float(faces[0].get("antispoof_score", 1.0))
    print(f"  Liveness score: {score:.3f}  is_real={is_real}")
    return is_real


# -- Quality gates + ArcFace embedding -----------------------------------------

def extract_embedding(image_path: str) -> Optional[List[float]]:
    """
    Run quality gates then extract an L2-normalised 512-D ArcFace embedding.

    Quality gates (in order):
      1. Exactly one face detected
      2. Detection confidence >= DET_SCORE_MIN
      3. Sharpness (Laplacian variance) >= MIN_BLUR_SCORE
      4. Pose |yaw| and |pitch| <= MAX_POSE_ANGLE
      5. [Optional] MiniFAS liveness check
    """
    import cv2  # type: ignore
    from insightface.app import FaceAnalysis  # type: ignore

    print(f"  Loading ArcFace model (first run downloads ~300 MB) ...")
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))

    img = cv2.imread(image_path)
    if img is None:
        print(f"  ERROR: Could not read '{image_path}'")
        return None

    faces = app.get(img)

    # Gate 1: face count
    if len(faces) == 0:
        print("  ERROR: No face detected. Use a clear, well-lit portrait.")
        return None
    if len(faces) > 1:
        print(f"  ERROR: {len(faces)} faces detected. Enrollment requires a single-person portrait.")
        return None

    best = faces[0]

    # Gate 2: detection confidence
    det_score = float(best.det_score)
    print(f"  Detection score:  {det_score:.4f}")
    if det_score < DET_SCORE_MIN:
        print(f"  ERROR: Detection score below {DET_SCORE_MIN}. Use a clearer image.")
        return None

    # Gate 3: sharpness
    import cv2  # already imported above; keep for reader clarity
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    print(f"  Sharpness score:  {blur_score:.1f}  (min {MIN_BLUR_SCORE})")
    if blur_score < MIN_BLUR_SCORE:
        print(f"  ERROR: Image too blurry (score={blur_score:.1f}). Use a sharper photo.")
        return None

    # Gate 4: pose (frontal check)
    if hasattr(best, "pose") and best.pose is not None:
        pitch = float(best.pose[0])
        yaw   = float(best.pose[1])
        print(f"  Pose:             pitch={pitch:.1f}  yaw={yaw:.1f}  (max +/-{MAX_POSE_ANGLE})")
        if abs(yaw) > MAX_POSE_ANGLE or abs(pitch) > MAX_POSE_ANGLE:
            print(
                f"  ERROR: Face not frontal enough "
                f"(yaw={yaw:.1f} pitch={pitch:.1f}). Use a forward-facing portrait."
            )
            return None

    # Gate 5: liveness (optional)
    if LIVENESS_ENABLED:
        try:
            is_real = _check_liveness(img)
            if not is_real:
                print("  ERROR: Liveness check FAILED. Enrollment rejected (possible printed photo or screen).")
                return None
        except ImportError:
            print("  WARNING: deepface not installed -- liveness check skipped.")
            print("           pip install deepface  to enable MiniFAS liveness.")

    # Extract embedding
    vec = best.embedding.astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm > 1e-10:
        vec /= norm
    return vec.tolist()


# -- Enroll --------------------------------------------------------------------

def enroll(
    user_id: str,
    name: str,
    image_path: str,
    metadata: Optional[Dict[str, Any]],
    consent_version: str,
    consent_method: str,
    consented_by: Optional[str],
    tenant: Optional[str],
    security_clearance: str = "standard",
    twin_risk_flagged: bool = False,
    require_secondary_factor: bool = False,
    secondary_factor_hash: Optional[str] = None,
    extra_images: Optional[List[str]] = None,
) -> None:
    from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue, PointStruct  # type: ignore

    collection = _collection(tenant)
    image_path = str(Path(image_path).resolve())
    if not Path(image_path).is_file():
        print(f"ERROR: Image not found: {image_path}")
        sys.exit(1)

    print(f"\nEnrolling user_id='{user_id}'  collection='{collection}'  tier='{security_clearance}' ...")

    # Multi-angle enrollment: average embeddings from up to 3 images for a more
    # distinctive, stable template. Identical twins have different micro-geometry
    # that becomes more measurable when the template is averaged across angles.
    all_images = [image_path] + (extra_images or [])
    all_vectors: List[List[float]] = []
    for img_path in all_images[:3]:
        resolved = str(Path(img_path).resolve())
        if not Path(resolved).is_file():
            print(f"  WARNING: Image not found, skipping: {resolved}")
            continue
        vec = extract_embedding(resolved)
        if vec is None:
            print(f"  WARNING: Quality gate failed for {resolved}, skipping.")
            continue
        all_vectors.append(vec)

    if not all_vectors:
        print("ERROR: No images passed quality gates.")
        sys.exit(1)

    if len(all_vectors) > 1:
        print(f"  Averaging {len(all_vectors)} angle(s) for a composite template.")
        avg = np.mean(np.array(all_vectors, dtype=np.float32), axis=0)
        norm = np.linalg.norm(avg)
        vector = (avg / norm).tolist() if norm > 1e-10 else all_vectors[0]
    else:
        vector = all_vectors[0]

    client = get_client()
    ensure_collection(client, collection)

    existing, _ = client.scroll(
        collection_name=collection,
        scroll_filter=Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]),
        limit=1,
    )
    if existing:
        print(f"  WARNING: user_id '{user_id}' already enrolled -- overwriting.")
        client.delete(
            collection_name=collection,
            points_selector=FilterSelector(
                filter=Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))])
            ),
        )

    now_iso = datetime.now(timezone.utc).isoformat()
    payload: Dict[str, Any] = {
        "user_id":                  user_id,
        "name":                     name,
        "enrolled_at":              now_iso,
        "image_sha256":             hashlib.sha256(Path(image_path).read_bytes()).hexdigest(),
        "enrollment_image_count":   len(all_vectors),
        # Model version tags -- used to detect stale embeddings during upgrades
        "model_name":               MODEL_NAME,
        "model_version":            MODEL_VERSION,
        # Security clearance (controls threshold and enrollment freshness limit in worker)
        "security_clearance_level": security_clearance,
        # Twin risk flag: raises match threshold to 0.88-0.96 depending on tier.
        # Set for any user with a known near-identical sibling/twin enrolled in the system.
        "twin_risk_flagged":        twin_risk_flagged,
        # MFA requirement
        "require_secondary_factor": require_secondary_factor,
        # Consent record
        "consent_version":          consent_version,
        "consent_method":           consent_method,
        "consent_ts":               now_iso,
    }
    if secondary_factor_hash:
        payload["secondary_factor_hash"] = secondary_factor_hash
    if consented_by:
        payload["consented_by"] = consented_by
    if tenant:
        payload["tenant_id"] = tenant
    if metadata:
        payload.update(metadata)

    point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"enrolled/{user_id}"))
    client.upsert(
        collection_name=collection,
        points=[PointStruct(id=point_id, vector=vector, payload=payload)],
    )

    print(f"\n  OK  Enrolled: user_id='{user_id}'  name='{name}'  point_id={point_id}")
    print(f"      model={MODEL_NAME}  version={MODEL_VERSION}  angles={len(all_vectors)}")
    print(f"      tier={security_clearance}  twin_risk={twin_risk_flagged}  mfa={require_secondary_factor}")
    print(f"      consent_version={consent_version}  method={consent_method}")


# -- Revoke consent (soft delete) ----------------------------------------------

def revoke_consent(user_id: str, tenant: Optional[str]) -> None:
    """
    Soft-revoke consent: sets revoked_at on the enrolled record.

    The worker will deny this user with CONSENT_REVOKED without removing the
    record, which preserves the audit trail. Use --forget for hard deletion.
    """
    from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue  # type: ignore

    collection = _collection(tenant)
    client = get_client()

    existing, _ = client.scroll(
        collection_name=collection,
        scroll_filter=Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]),
        limit=1,
    )
    if not existing:
        print(f"  ERROR: user_id '{user_id}' not found in '{collection}'.")
        sys.exit(1)

    if existing[0].payload and existing[0].payload.get("revoked_at"):
        print(f"  WARNING: user_id '{user_id}' already revoked at {existing[0].payload['revoked_at']}.")
        return

    revoked_at = datetime.now(timezone.utc).isoformat()
    client.set_payload(
        collection_name=collection,
        payload={"revoked_at": revoked_at},
        points=FilterSelector(
            filter=Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))])
        ),
    )
    print(f"  OK  Consent revoked for user_id='{user_id}' at {revoked_at}.")
    print(f"      The worker will now deny this user with reason: CONSENT_REVOKED.")
    print(f"      Use --forget to hard-delete the record (GDPR right-to-be-forgotten).")


# -- Forget (hard delete, right-to-be-forgotten) --------------------------------

def forget_user(user_id: str, tenant: Optional[str]) -> None:
    """
    GDPR Art. 17 right-to-be-forgotten: hard-delete the enrolled record.

    NOTE: Kafka auth.audit records are append-only and cannot be deleted by
    this script. Purge those separately via your Kafka admin tooling if required.
    """
    from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue  # type: ignore

    collection = _collection(tenant)
    client = get_client()
    client.delete(
        collection_name=collection,
        points_selector=FilterSelector(
            filter=Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))])
        ),
    )
    print(f"  OK  Hard-deleted enrollment for user_id='{user_id}' from '{collection}'.")
    print(f"      NOTE: auth.audit Kafka records are append-only and were NOT deleted.")


# -- List ----------------------------------------------------------------------

def list_enrolled(tenant: Optional[str]) -> None:
    collection = _collection(tenant)
    client = get_client()
    ensure_collection(client, collection)
    records, _ = client.scroll(
        collection_name=collection,
        with_payload=True,
        with_vectors=False,
        limit=200,
    )
    if not records:
        print(f"  No enrolled faces in '{collection}'.")
        return

    print(f"\n  {'user_id':<20}  {'name':<25}  {'model':<15}  {'status':<10}  {'enrolled_at'}")
    print(f"  {'-'*20}  {'-'*25}  {'-'*15}  {'-'*10}  {'-'*30}")
    for r in records:
        p = r.payload or {}
        status = "REVOKED" if p.get("revoked_at") else "active"
        model  = p.get("model_name", "?")
        print(
            f"  {p.get('user_id','?'):<20}  {p.get('name','?'):<25}  "
            f"{model:<15}  {status:<10}  {p.get('enrolled_at','?')}"
        )


# -- Delete (legacy alias for forget) ------------------------------------------

def delete_user(user_id: str, tenant: Optional[str]) -> None:
    forget_user(user_id, tenant)


# -- CLI -----------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        prog="enroll_face",
        description="Enroll, manage, or delete faces in the Qdrant enrolled_faces collection.",
    )
    p.add_argument("--tenant",          metavar="ID",   default=None,
                   help="Tenant ID for per-tenant collection isolation (optional).")
    p.add_argument("--user-id",         metavar="ID",   help="Unique employee / user ID.")
    p.add_argument("--name",            metavar="NAME", help="Full name (stored in payload).")
    p.add_argument("--image",           metavar="PATH", help="Path to portrait JPEG/PNG.")
    p.add_argument("--consent-version",  metavar="VER",  default="1.0",
                   help="Consent form version (default: 1.0).")
    p.add_argument("--consent-method",   metavar="METHOD", default="written",
                   choices=["written", "digital", "verbal"],
                   help="How consent was obtained (default: written).")
    p.add_argument("--consented-by",     metavar="ADMIN", default=None,
                   help="ID of admin who witnessed consent (optional).")
    p.add_argument("--security-clearance", metavar="TIER", default="standard",
                   choices=["standard", "enhanced", "critical"],
                   help="Security tier controlling match threshold and re-enrollment interval "
                        "(standard=0.75 / enhanced=0.82 / critical=0.88).")
    p.add_argument("--twin-risk",        action="store_true",
                   help="Flag user as having a near-identical twin/sibling in the system. "
                        "Raises match threshold to 0.88-0.96 depending on tier.")
    p.add_argument("--require-secondary-factor", action="store_true",
                   help="Require a secondary factor (TOTP/PIN) alongside face at auth time.")
    p.add_argument("--secondary-factor-hash", metavar="SHA256", default=None,
                   help="SHA-256 hex digest of the user's PIN/token. "
                        "Generate with: python -c \"import hashlib; print(hashlib.sha256(b'PIN').hexdigest())\"")
    p.add_argument("--extra-image",      metavar="PATH", action="append", dest="extra_images",
                   help="Additional angle image for multi-angle enrollment (up to 2 extra). "
                        "Embeddings are averaged for a more distinctive composite template.")
    p.add_argument("--metadata",         metavar="JSON", default=None,
                   help='Extra JSON fields, e.g. \'{"dept": "HR"}\'')
    p.add_argument("--list",            action="store_true",
                   help="List all enrolled users.")
    p.add_argument("--delete",          metavar="USER_ID",
                   help="Hard-delete an enrolled user (alias for --forget).")
    p.add_argument("--revoke",          metavar="USER_ID",
                   help="Soft-revoke consent (worker denies; record kept for audit).")
    p.add_argument("--forget",          metavar="USER_ID",
                   help="Hard-delete (GDPR right-to-be-forgotten).")
    args = p.parse_args()

    if args.list:
        list_enrolled(args.tenant)
    elif args.revoke:
        revoke_consent(args.revoke, args.tenant)
    elif args.forget:
        forget_user(args.forget, args.tenant)
    elif args.delete:
        delete_user(args.delete, args.tenant)
    elif args.user_id and args.image:
        try:
            meta = json.loads(args.metadata) if args.metadata else None
        except json.JSONDecodeError as exc:
            print(f"ERROR: --metadata is not valid JSON: {exc}")
            sys.exit(1)
        enroll(
            user_id=args.user_id,
            name=args.name or args.user_id,
            image_path=args.image,
            metadata=meta,
            consent_version=args.consent_version,
            consent_method=args.consent_method,
            consented_by=args.consented_by,
            tenant=args.tenant,
            security_clearance=args.security_clearance,
            twin_risk_flagged=args.twin_risk,
            require_secondary_factor=args.require_secondary_factor,
            secondary_factor_hash=args.secondary_factor_hash,
            extra_images=args.extra_images,
        )
    else:
        p.print_help()


if __name__ == "__main__":
    main()
