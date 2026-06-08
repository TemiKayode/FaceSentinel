"""
face_indexer.py
===============
Production-ready local face embedding & vector database indexing CLI.

Architecture
------------
  Detection/Alignment  ->  DeepFace (ArcFace, 512-D embeddings)
  Vector Storage       ->  Qdrant local on-disk (no server / Docker needed)

All operations are fully offline once model weights are downloaded on
the first run (~300 MB cached to ~/.deepface/).

Usage
-----
  # Register a face
  python face_indexer.py --mode register --name "Alice" --input alice.jpg

  # Search for an unknown face
  python face_indexer.py --mode search --input unknown.jpg --threshold 0.70

  # List every registered identity
  python face_indexer.py --mode list

  # Remove an identity by UUID
  python face_indexer.py --mode delete --id <uuid>
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointIdsList,
    PointStruct,
    VectorParams,
)

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("face_indexer")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EMBEDDING_DIM: int = 512          # ArcFace output dimensionality
ARCFACE_MODEL: str = "ArcFace"    # DeepFace model name
DEFAULT_DETECTOR: str = "retinaface"
DEFAULT_THRESHOLD: float = 0.65   # cosine similarity floor
DEFAULT_LIMIT: int = 10           # max candidates before threshold filter


# ===========================================================================
# Helper - unit-normalise a vector
# ===========================================================================

def _l2_normalise(vec: List[float]) -> List[float]:
    """Return the L2-normalised form of vec for consistent cosine similarity."""
    arr = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm < 1e-10:
        return vec
    return (arr / norm).tolist()


# ===========================================================================
# LocalFaceIndexer
# ===========================================================================

class LocalFaceIndexer:
    """
    Manages a local on-disk face identity index backed by Qdrant.

    Each registered identity is stored as a 512-dimensional ArcFace
    embedding alongside a JSON payload containing the person's name and
    any caller-supplied metadata.

    Parameters
    ----------
    collection_name : str
        Name of the Qdrant collection to use or create.
    db_path : str
        Filesystem path where Qdrant stores its segment files.
        Created automatically if absent.
    detector_backend : str
        DeepFace-compatible face detector:
        "retinaface" (most accurate), "opencv" (fastest),
        "mtcnn", "ssd", or "mediapipe".
    """

    def __init__(
        self,
        collection_name: str = "faces",
        db_path: str = "./face_db",
        detector_backend: str = DEFAULT_DETECTOR,
    ) -> None:
        self.collection_name = collection_name
        self.detector_backend = detector_backend

        # Ensure the storage directory exists before Qdrant touches it
        Path(db_path).mkdir(parents=True, exist_ok=True)
        log.info("Connecting to local Qdrant at '%s' ...", db_path)

        self._client = QdrantClient(path=db_path)
        self._ensure_collection()

    # ------------------------------------------------------------------
    # Collection bootstrap
    # ------------------------------------------------------------------

    def _ensure_collection(self) -> None:
        """Create the Qdrant collection if it does not already exist."""
        existing_names = [
            c.name for c in self._client.get_collections().collections
        ]
        if self.collection_name not in existing_names:
            log.info(
                "Collection '%s' not found - creating (dim=%d, metric=COSINE) ...",
                self.collection_name,
                EMBEDDING_DIM,
            )
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIM,
                    distance=Distance.COSINE,
                ),
            )
            log.info("Collection '%s' ready.", self.collection_name)
        else:
            count = self._client.count(self.collection_name).count
            log.info(
                "Collection '%s' found - %d face(s) currently indexed.",
                self.collection_name,
                count,
            )

    # ------------------------------------------------------------------
    # Embedding extraction (internal)
    # ------------------------------------------------------------------

    def _extract_embedding(self, img_path: str) -> Optional[List[float]]:
        """
        Run ArcFace inference on a local image and return a unit-normalised
        512-D embedding vector.

        Parameters
        ----------
        img_path : str
            Absolute path to a JPEG or PNG portrait image.

        Returns
        -------
        list[float] or None
            512-element float list, or None if no face was detected or
            if inference encountered an unrecoverable error.
        """
        # Lazy import: keeps argparse --help instant; avoids 4-second
        # TensorFlow init when the user is just checking syntax.
        from deepface import DeepFace  # type: ignore

        try:
            representations = DeepFace.represent(
                img_path=img_path,
                model_name=ARCFACE_MODEL,
                detector_backend=self.detector_backend,
                enforce_detection=True,   # raises ValueError if no face found
                align=True,               # canonical face alignment improves accuracy
            )
        except ValueError as exc:
            # DeepFace raises ValueError specifically for "Face could not be detected"
            log.warning("No face detected in '%s': %s", img_path, exc)
            return None
        except Exception as exc:
            log.error(
                "Unexpected error during embedding extraction for '%s': %s",
                img_path, exc,
            )
            return None

        if not representations:
            log.warning("DeepFace returned an empty list for '%s'.", img_path)
            return None

        # When multiple faces are present take the one with the largest
        # bounding-box area (most prominent face in the frame).
        best = max(
            representations,
            key=lambda r: (
                r.get("facial_area", {}).get("w", 0)
                * r.get("facial_area", {}).get("h", 0)
            ),
        )

        raw_vec: List[float] = best["embedding"]
        log.debug("Raw embedding length: %d", len(raw_vec))
        return _l2_normalise(raw_vec)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_face(
        self,
        name: str,
        img_path: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Extract an ArcFace embedding from a local image and store the
        resulting identity record in the Qdrant index.

        Parameters
        ----------
        name : str
            Human-readable label for this identity (e.g. "Alice Smith").
            Stored verbatim in the Qdrant payload and returned during search.
        img_path : str
            Path to the source portrait image (JPEG / PNG).
        metadata : dict, optional
            Arbitrary extra fields to store alongside the embedding
            (e.g. ``{"department": "Engineering", "employee_id": "E-042"}``).
            All values must be JSON-serialisable.

        Returns
        -------
        str
            UUID-v4 string assigned to the newly created Qdrant point,
            or ``None`` if registration was skipped due to an error.

        Notes
        -----
        If the image contains multiple faces the largest (most prominent)
        face is selected for registration. If no face is detected at all
        the call logs a warning and returns ``None`` without modifying
        the index.
        """
        resolved = Path(img_path).resolve()
        if not resolved.is_file():
            log.error("Image file not found: '%s'", resolved)
            return None

        log.info("Registering '%s' from '%s' ...", name, resolved)

        embedding = self._extract_embedding(str(resolved))
        if embedding is None:
            log.warning(
                "Registration aborted for '%s' - no usable face found in '%s'.",
                name, resolved,
            )
            return None

        point_id = str(uuid.uuid4())
        payload: Dict[str, Any] = {
            "name": name,
            "source_image": str(resolved),
        }
        if metadata:
            # Shallow merge; name / source_image take precedence over
            # any colliding metadata keys
            merged = {**metadata, "name": name, "source_image": str(resolved)}
            payload = merged

        self._client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload,
                )
            ],
        )

        log.info("Registered '%s' successfully -> point ID: %s", name, point_id)
        return point_id

    def search_face(
        self,
        query_img_path: str,
        threshold: float = DEFAULT_THRESHOLD,
        limit: int = DEFAULT_LIMIT,
    ) -> List[Dict[str, Any]]:
        """
        Identify an unknown face by querying the index for the nearest
        stored embeddings above a similarity threshold.

        Parameters
        ----------
        query_img_path : str
            Path to the query image. Must contain exactly one visible face.
        threshold : float
            Minimum cosine similarity for a result to be returned.
            Values range from 0 (completely dissimilar) to 1 (identical).
            Recommended operating range: 0.60 - 0.75.
        limit : int
            Upper bound on the number of candidates Qdrant retrieves
            before the threshold filter is applied.

        Returns
        -------
        list[dict]
            Sorted list (highest score first) of match dicts, each
            containing at minimum: ``id``, ``score``, ``name``.
            Any additional metadata fields stored at registration are
            also included. Returns an empty list if no face is found or
            no stored identity clears the threshold.
        """
        resolved = Path(query_img_path).resolve()
        if not resolved.is_file():
            log.error("Query image not found: '%s'", resolved)
            return []

        log.info(
            "Searching for face in '%s' (threshold=%.2f, limit=%d) ...",
            resolved, threshold, limit,
        )

        embedding = self._extract_embedding(str(resolved))
        if embedding is None:
            log.warning(
                "Search aborted - no usable face detected in '%s'.", resolved
            )
            return []

        scored_points = self._client.search(
            collection_name=self.collection_name,
            query_vector=embedding,
            limit=limit,
            score_threshold=threshold,
            with_payload=True,
        )

        results: List[Dict[str, Any]] = []
        for hit in scored_points:
            entry: Dict[str, Any] = {
                "id": str(hit.id),
                "score": round(float(hit.score), 6),
            }
            if hit.payload:
                entry.update(hit.payload)
            results.append(entry)

        # Qdrant already returns results sorted by score, but make it explicit
        results.sort(key=lambda r: r["score"], reverse=True)

        if results:
            log.info(
                "%d match(es) found above threshold %.2f. "
                "Best: '%s' (score=%.4f)",
                len(results),
                threshold,
                results[0].get("name", "<unknown>"),
                results[0]["score"],
            )
        else:
            log.info("No matches found above threshold %.2f.", threshold)

        return results

    def list_registered(self) -> List[Dict[str, Any]]:
        """
        Return a compact summary of every registered identity in the index.

        Returns
        -------
        list[dict]
            Each dict contains ``id`` and ``name`` (plus any metadata
            fields stored at registration). Sorted alphabetically by name.
        """
        # scroll() pages through all points; 1000 per page is safe for
        # local deployments - raise the limit or implement pagination for
        # very large collections.
        records, _next = self._client.scroll(
            collection_name=self.collection_name,
            with_payload=True,
            with_vectors=False,
            limit=1000,
        )

        summaries: List[Dict[str, Any]] = []
        for rec in records:
            entry: Dict[str, Any] = {"id": str(rec.id)}
            if rec.payload:
                entry.update(rec.payload)
            summaries.append(entry)

        summaries.sort(key=lambda r: str(r.get("name", "")))
        return summaries

    def delete_face(self, point_id: str) -> bool:
        """
        Permanently remove an identity record from the index by its UUID.

        Parameters
        ----------
        point_id : str
            The UUID-v4 string assigned at registration time.

        Returns
        -------
        bool
            ``True`` if the operation was acknowledged by Qdrant,
            ``False`` if an error occurred.
        """
        try:
            self._client.delete(
                collection_name=self.collection_name,
                points_selector=PointIdsList(points=[point_id]),
            )
            log.info("Deleted identity point ID=%s.", point_id)
            return True
        except Exception as exc:
            log.error("Failed to delete point ID=%s: %s", point_id, exc)
            return False

    def collection_stats(self) -> Dict[str, Any]:
        """
        Return basic statistics about the active collection.

        Returns
        -------
        dict
            Keys: ``collection``, ``total_faces``, ``vector_dim``,
            ``distance_metric``.
        """
        info = self._client.get_collection(self.collection_name)
        count = self._client.count(self.collection_name).count
        return {
            "collection": self.collection_name,
            "total_faces": count,
            "vector_dim": EMBEDDING_DIM,
            "distance_metric": "Cosine",
            "status": str(info.status),
        }


# ===========================================================================
# CLI - argument parser
# ===========================================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="face_indexer",
        description=(
            "Local face embedding & vector index CLI.\n"
            "Runs fully offline using DeepFace (ArcFace) + Qdrant on-disk."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes
-----
  register   Extract and store a face embedding from a local image.
  search     Identify an unknown face against the index.
  list       Print all registered identities (name + UUID).
  delete     Remove an identity from the index by UUID.
  stats      Print collection statistics.

Examples
--------
  python face_indexer.py --mode register \\
      --name "Alice Smith" --input ./photos/alice.jpg \\
      --metadata '{"department": "Engineering"}'

  python face_indexer.py --mode search \\
      --input ./photos/unknown.jpg --threshold 0.70

  python face_indexer.py --mode list

  python face_indexer.py --mode delete --id 550e8400-e29b-41d4-a716-446655440000

  python face_indexer.py --mode stats
""",
    )

    p.add_argument(
        "--mode", "-m",
        required=True,
        choices=["register", "search", "list", "delete", "stats"],
        metavar="MODE",
        help=(
            "Execution mode: register | search | list | delete | stats"
        ),
    )
    p.add_argument(
        "--input", "-i",
        metavar="PATH",
        help="Path to the input image (required for register / search).",
    )
    p.add_argument(
        "--name", "-n",
        metavar="NAME",
        help="Person's full name (required for register mode).",
    )
    p.add_argument(
        "--metadata",
        metavar="JSON",
        default=None,
        help=(
            "Optional JSON string of extra fields to store alongside the "
            "embedding. Example: '{\"department\": \"HR\", \"site\": \"NYC\"}'"
        ),
    )
    p.add_argument(
        "--threshold", "-t",
        type=float,
        default=DEFAULT_THRESHOLD,
        metavar="FLOAT",
        help=(
            "Cosine similarity floor for search results "
            f"(0-1, default: {DEFAULT_THRESHOLD}). "
            "Raise to reduce false positives; lower to increase recall."
        ),
    )
    p.add_argument(
        "--limit", "-l",
        type=int,
        default=DEFAULT_LIMIT,
        metavar="INT",
        help=f"Max candidates retrieved before threshold filtering (default: {DEFAULT_LIMIT}).",
    )
    p.add_argument(
        "--id",
        metavar="UUID",
        help="Point UUID to remove (required for delete mode).",
    )
    p.add_argument(
        "--collection",
        default="faces",
        metavar="NAME",
        help='Qdrant collection name (default: "faces").',
    )
    p.add_argument(
        "--db-path",
        default="./face_db",
        metavar="PATH",
        help='On-disk storage path for Qdrant segments (default: "./face_db").',
    )
    p.add_argument(
        "--detector",
        default=DEFAULT_DETECTOR,
        choices=["retinaface", "opencv", "mtcnn", "ssd", "mediapipe"],
        help=(
            f'Face detector backend (default: "{DEFAULT_DETECTOR}"). '
            '"retinaface" is most accurate; "opencv" is fastest.'
        ),
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )

    return p


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> int:  # noqa: C901
    parser = _build_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        log.setLevel(logging.DEBUG)

    # ------------------------------------------------------------------
    # Initialise indexer
    # ------------------------------------------------------------------
    try:
        indexer = LocalFaceIndexer(
            collection_name=args.collection,
            db_path=args.db_path,
            detector_backend=args.detector,
        )
    except Exception as exc:
        log.error("Failed to initialise LocalFaceIndexer: %s", exc)
        return 1

    # ------------------------------------------------------------------
    # Dispatch mode
    # ------------------------------------------------------------------

    # -- register --------------------------------------------------------
    if args.mode == "register":
        if not args.name:
            log.error("--name is required for register mode.")
            return 1
        if not args.input:
            log.error("--input is required for register mode.")
            return 1

        metadata: Optional[Dict[str, Any]] = None
        if args.metadata:
            try:
                metadata = json.loads(args.metadata)
                if not isinstance(metadata, dict):
                    raise ValueError("Metadata must be a JSON object (dict).")
            except (json.JSONDecodeError, ValueError) as exc:
                log.error("Invalid --metadata value: %s", exc)
                return 1

        point_id = indexer.register_face(args.name, args.input, metadata)
        if point_id:
            print(
                json.dumps(
                    {"status": "registered", "name": args.name, "id": point_id},
                    indent=2,
                )
            )
            return 0
        else:
            print(json.dumps({"status": "failed", "name": args.name}, indent=2))
            return 1

    # -- search ----------------------------------------------------------
    elif args.mode == "search":
        if not args.input:
            log.error("--input is required for search mode.")
            return 1

        matches = indexer.search_face(
            query_img_path=args.input,
            threshold=args.threshold,
            limit=args.limit,
        )
        print(
            json.dumps(
                {
                    "query": str(Path(args.input).resolve()),
                    "threshold": args.threshold,
                    "num_matches": len(matches),
                    "matches": matches,
                },
                indent=2,
            )
        )
        return 0

    # -- list ------------------------------------------------------------
    elif args.mode == "list":
        records = indexer.list_registered()
        print(
            json.dumps(
                {"total": len(records), "faces": records},
                indent=2,
            )
        )
        return 0

    # -- delete ----------------------------------------------------------
    elif args.mode == "delete":
        if not args.id:
            log.error("--id is required for delete mode.")
            return 1
        success = indexer.delete_face(args.id)
        print(
            json.dumps(
                {"status": "deleted" if success else "failed", "id": args.id},
                indent=2,
            )
        )
        return 0 if success else 1

    # -- stats ------------------------------------------------------------
    elif args.mode == "stats":
        stats = indexer.collection_stats()
        print(json.dumps(stats, indent=2))
        return 0

    # Should never reach here due to argparse choices validation
    log.error("Unknown mode: %s", args.mode)
    return 1


if __name__ == "__main__":
    sys.exit(main())
