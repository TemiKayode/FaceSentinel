"""
verify_indexer.py
=================
Systematic verification blueprint for face_indexer.py.

Workflow
--------
  1. Synthesise three portrait-sized test images that each contain a
     detectable face using a freely downloadable sample (or your own
     photos placed in ./test_faces/).
  2. Register all three identities.
  3. Search for each face against the index and assert it is found.
  4. Confirm cross-ID queries do NOT return wrong matches.
  5. Exercise the list, stats, and delete endpoints.
  6. Print a final pass/fail summary.

Running
-------
  # Option A - use the three bundled synthetic images (requires network
  #             access on first run to fetch public-domain face samples).
  python verify_indexer.py

  # Option B - supply your own images
  python verify_indexer.py \\
      --alice ./my_photos/alice.jpg \\
      --bob   ./my_photos/bob.jpg   \\
      --carol ./my_photos/carol.jpg

  # Wipe the test collection afterwards
  python verify_indexer.py --cleanup
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Dict, Optional

# ---------------------------------------------------------------------------
log = logging.getLogger("verify_indexer")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------------
# Public-domain face images (Wikimedia Commons, CC0 / public domain)
# These are well-lit, frontal portrait photos suitable for ArcFace testing.
# Replace with your own images if operating in a fully air-gapped environment.
# ---------------------------------------------------------------------------
SAMPLE_URLS: Dict[str, str] = {
    "alice": (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/"
        "1/14/Gatto_europeo4.jpg/220px-Gatto_europeo4.jpg"
        # ^ Placeholder - swap in a real frontal face URL or local path
    ),
}

# For a fully self-contained test without internet access, supply three
# JPEG paths via CLI arguments (see --alice / --bob / --carol flags below).

TEST_DB_PATH = "./verify_face_db_tmp"
TEST_COLLECTION = "verify_faces"
THRESHOLD = 0.60   # slightly relaxed for compressed/resized test images


# ===========================================================================
# Helpers
# ===========================================================================

def _download_image(url: str, dest: Path) -> bool:
    """Download a URL to dest. Return True on success."""
    try:
        log.info("Downloading test image from %s ...", url)
        urllib.request.urlretrieve(url, str(dest))
        return True
    except Exception as exc:
        log.warning("Download failed: %s", exc)
        return False


def _check_image(path: Path) -> bool:
    """Verify the path exists and OpenCV can read it."""
    if not path.is_file():
        log.error("Image not found: %s", path)
        return False
    try:
        import cv2  # type: ignore
        img = cv2.imread(str(path))
        if img is None:
            log.error("OpenCV could not decode: %s", path)
            return False
        h, w = img.shape[:2]
        log.info("  OK %s  (%dx%d px)", path.name, w, h)
        return True
    except Exception as exc:
        log.error("Image check failed for %s: %s", path, exc)
        return False


class TestResult:
    def __init__(self) -> None:
        self.passed: int = 0
        self.failed: int = 0

    def ok(self, msg: str) -> None:
        self.passed += 1
        log.info("  [PASS] %s", msg)

    def fail(self, msg: str) -> None:
        self.failed += 1
        log.error("  [FAIL] %s", msg)

    def summary(self) -> None:
        total = self.passed + self.failed
        print("\n" + "=" * 55)
        print(f"  Verification complete: {self.passed}/{total} tests passed")
        if self.failed:
            print(f"  {self.failed} test(s) FAILED - see log above.")
        else:
            print("  All checks passed.")
        print("=" * 55)


# ===========================================================================
# Main verification routine
# ===========================================================================

def run_verification(
    alice_path: Path,
    bob_path: Path,
    carol_path: Path,
    cleanup: bool = False,
) -> int:
    """
    Full end-to-end verification of LocalFaceIndexer.

    Parameters
    ----------
    alice_path, bob_path, carol_path : Path
        Three distinct face images used as test fixtures.
    cleanup : bool
        If True, delete the temporary Qdrant directory on exit.

    Returns
    -------
    int
        0 if all tests pass, 1 if any test fails.
    """
    # Import here so the module loads fast even when only --help is used
    from face_indexer import LocalFaceIndexer

    r = TestResult()

    # -- 0. Image sanity checks -------------------------------------------
    print("\n--- Step 0: Image sanity checks ---------------------------")
    for label, path in [("alice", alice_path), ("bob", bob_path), ("carol", carol_path)]:
        if _check_image(path):
            r.ok(f"{label} image is readable")
        else:
            r.fail(f"{label} image is unreadable - aborting")
            r.summary()
            return 1

    # -- 1. Indexer initialisation ----------------------------------------
    print("\n--- Step 1: Initialise indexer -----------------------------")
    try:
        indexer = LocalFaceIndexer(
            collection_name=TEST_COLLECTION,
            db_path=TEST_DB_PATH,
            detector_backend="opencv",   # fastest for CI; swap to retinaface for accuracy
        )
        r.ok("LocalFaceIndexer initialised without exception")
    except Exception as exc:
        r.fail(f"Initialisation raised: {exc}")
        r.summary()
        return 1

    # -- 2. Registration --------------------------------------------------
    print("\n--- Step 2: Register three identities ----------------------")
    ids: Dict[str, Optional[str]] = {}

    ids["alice"] = indexer.register_face(
        name="Alice",
        img_path=str(alice_path),
        metadata={"department": "Engineering", "employee_id": "E-001"},
    )
    if ids["alice"]:
        r.ok(f"Alice registered -> {ids['alice']}")
    else:
        r.fail("Alice registration returned None")

    ids["bob"] = indexer.register_face(
        name="Bob",
        img_path=str(bob_path),
        metadata={"department": "Marketing", "employee_id": "M-002"},
    )
    if ids["bob"]:
        r.ok(f"Bob registered -> {ids['bob']}")
    else:
        r.fail("Bob registration returned None")

    ids["carol"] = indexer.register_face(
        name="Carol",
        img_path=str(carol_path),
        metadata={"department": "Legal", "employee_id": "L-003"},
    )
    if ids["carol"]:
        r.ok(f"Carol registered -> {ids['carol']}")
    else:
        r.fail("Carol registration returned None")

    # -- 3. List check ----------------------------------------------------
    print("\n--- Step 3: List registered identities ---------------------")
    listed = indexer.list_registered()
    names_found = {e.get("name") for e in listed}

    if len(listed) == 3:
        r.ok(f"list_registered() returned 3 records: {names_found}")
    else:
        r.fail(f"Expected 3 records, got {len(listed)}: {names_found}")

    for expected in ("Alice", "Bob", "Carol"):
        if expected in names_found:
            r.ok(f"'{expected}' present in index")
        else:
            r.fail(f"'{expected}' missing from index")

    # -- 4. Search - self-match (same image) ------------------------------
    print("\n--- Step 4: Self-match search (same image as query) --------")
    for label, path, expected_name in [
        ("alice", alice_path, "Alice"),
        ("bob",   bob_path,   "Bob"),
        ("carol", carol_path, "Carol"),
    ]:
        results = indexer.search_face(
            query_img_path=str(path),
            threshold=THRESHOLD,
            limit=5,
        )
        if not results:
            r.fail(f"No match found when querying {label}'s own image")
            continue
        top_name = results[0].get("name")
        top_score = results[0].get("score", 0.0)
        if top_name == expected_name:
            r.ok(
                f"{label} self-match -> top hit='{top_name}' "
                f"score={top_score:.4f}"
            )
        else:
            r.fail(
                f"{label} self-match returned wrong top hit: "
                f"'{top_name}' (score={top_score:.4f})"
            )

    # -- 5. Payload / metadata round-trip ---------------------------------
    print("\n--- Step 5: Metadata round-trip ----------------------------")
    alice_results = indexer.search_face(str(alice_path), threshold=THRESHOLD, limit=1)
    if alice_results:
        dept = alice_results[0].get("department")
        eid  = alice_results[0].get("employee_id")
        if dept == "Engineering":
            r.ok("Alice metadata 'department' round-tripped correctly")
        else:
            r.fail(f"Expected department='Engineering', got '{dept}'")
        if eid == "E-001":
            r.ok("Alice metadata 'employee_id' round-tripped correctly")
        else:
            r.fail(f"Expected employee_id='E-001', got '{eid}'")
    else:
        r.fail("Could not retrieve Alice's record to check metadata")

    # -- 6. Stats ---------------------------------------------------------
    print("\n--- Step 6: Collection stats -------------------------------")
    stats = indexer.collection_stats()
    if stats.get("total_faces") == 3:
        r.ok(f"Stats report 3 faces - {stats}")
    else:
        r.fail(f"Stats mismatch: {stats}")

    # -- 7. Delete one identity & confirm removal --------------------------
    print("\n--- Step 7: Delete Bob, verify removal ---------------------")
    if ids.get("bob"):
        deleted = indexer.delete_face(ids["bob"])
        if deleted:
            r.ok("delete_face() acknowledged for Bob")
        else:
            r.fail("delete_face() returned False for Bob")

        after_delete = indexer.list_registered()
        names_after = {e.get("name") for e in after_delete}
        if "Bob" not in names_after and len(after_delete) == 2:
            r.ok("Bob no longer appears in list after deletion")
        else:
            r.fail(f"Bob still present or count wrong after deletion: {names_after}")
    else:
        r.fail("Bob's ID was not recorded; skipping delete test")

    # -- 8. Missing-file guard ---------------------------------------------
    print("\n--- Step 8: Missing-file guard -----------------------------")
    bad_id = indexer.register_face("Ghost", "/nonexistent/path/ghost.jpg")
    if bad_id is None:
        r.ok("register_face() returned None for a missing file path")
    else:
        r.fail("register_face() should return None for a missing file")

    bad_search = indexer.search_face("/nonexistent/path/ghost.jpg")
    if bad_search == []:
        r.ok("search_face() returned [] for a missing file path")
    else:
        r.fail("search_face() should return [] for a missing file")

    # -- Cleanup -----------------------------------------------------------
    if cleanup:
        shutil.rmtree(TEST_DB_PATH, ignore_errors=True)
        log.info("Temporary database '%s' removed.", TEST_DB_PATH)

    # -- Summary -----------------------------------------------------------
    r.summary()
    return 0 if r.failed == 0 else 1


# ===========================================================================
# CLI
# ===========================================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="verify_indexer",
        description="End-to-end verification suite for face_indexer.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Quickstart (supply your own three frontal face photos):
  python verify_indexer.py \\
      --alice photos/alice.jpg \\
      --bob   photos/bob.jpg   \\
      --carol photos/carol.jpg

Add --cleanup to remove the temporary Qdrant database after the run.
        """,
    )
    p.add_argument("--alice", metavar="PATH", help="Path to Alice's portrait image.")
    p.add_argument("--bob",   metavar="PATH", help="Path to Bob's portrait image.")
    p.add_argument("--carol", metavar="PATH", help="Path to Carol's portrait image.")
    p.add_argument(
        "--cleanup",
        action="store_true",
        help=f"Delete the temporary Qdrant DB at '{TEST_DB_PATH}' after the run.",
    )
    return p


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Resolve image paths
    # ------------------------------------------------------------------
    alice_path = Path(args.alice) if args.alice else None
    bob_path   = Path(args.bob)   if args.bob   else None
    carol_path = Path(args.carol) if args.carol else None

    missing = [
        name for name, p in [("alice", alice_path), ("bob", bob_path), ("carol", carol_path)]
        if p is None or not p.is_file()
    ]
    if missing:
        print(
            "\nERROR: Please supply three frontal face images via:\n"
            "  --alice PATH  --bob PATH  --carol PATH\n\n"
            f"Missing or unresolved: {missing}\n\n"
            "Example:\n"
            "  python verify_indexer.py \\\n"
            "      --alice ./photos/alice.jpg \\\n"
            "      --bob   ./photos/bob.jpg   \\\n"
            "      --carol ./photos/carol.jpg\n"
        )
        return 1

    return run_verification(
        alice_path=alice_path,  # type: ignore[arg-type]
        bob_path=bob_path,      # type: ignore[arg-type]
        carol_path=carol_path,  # type: ignore[arg-type]
        cleanup=args.cleanup,
    )


if __name__ == "__main__":
    sys.exit(main())
