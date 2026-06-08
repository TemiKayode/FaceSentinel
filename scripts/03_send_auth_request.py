"""
03_send_auth_request.py
=======================
Send a single authentication request to the auth.requests Kafka topic.

Simulates what a physical terminal (badge reader + camera) would publish
when a user presents their badge and face.

Usage
-----
    # Basic: image read from file
    python scripts/03_send_auth_request.py \\
        --user-id  "emp-001" \\
        --terminal "door-lobby-nyc" \\
        --image    "./test_faces/alice_test.jpg"

    # Repeat N times (load test)
    python scripts/03_send_auth_request.py \\
        --user-id emp-001 --terminal door-lobby-nyc \\
        --image ./test_faces/alice_test.jpg --repeat 10
"""

from __future__ import annotations

import argparse
import base64
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

KAFKA_BOOTSTRAP   = os.getenv("KAFKA_BOOTSTRAP_SERVERS",   "localhost:9092")
SCHEMA_REG_URL    = os.getenv("KAFKA_SCHEMA_REGISTRY_URL", "http://localhost:8081")
REQUEST_TOPIC     = os.getenv("KAFKA_REQUEST_TOPIC",       "auth.requests")
MAX_IMAGE_BYTES   = 524_288   # must match worker cap


def _load_image_b64(image_path: str) -> str:
    import io
    import cv2  # type: ignore
    path = Path(image_path).resolve()
    if not path.is_file():
        print(f"ERROR: Image not found: {path}")
        sys.exit(1)
    raw = path.read_bytes()
    if len(raw) <= MAX_IMAGE_BYTES:
        return base64.b64encode(raw).decode("ascii")

    # Auto-resize: scale down until the JPEG fits under the cap.
    img = cv2.imread(str(path))
    if img is None:
        print(f"ERROR: Could not decode image: {path}")
        sys.exit(1)
    scale = 1.0
    for quality in (95, 85, 75, 65):
        h, w = img.shape[:2]
        resized = cv2.resize(img, (int(w * scale), int(h * scale)))
        ok, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok and len(buf) <= MAX_IMAGE_BYTES:
            print(f"  Auto-resized: {len(raw):,} -> {len(buf):,} bytes (scale={scale:.2f} q={quality})")
            return base64.b64encode(buf.tobytes()).decode("ascii")
        scale *= 0.75
    print(f"ERROR: Could not compress image below {MAX_IMAGE_BYTES} bytes.")
    sys.exit(1)


def send_request(
    user_id: str,
    terminal_id: str,
    image_b64: str,
    transaction_id: str | None = None,
) -> str:
    from confluent_kafka import Producer  # type: ignore
    from confluent_kafka.schema_registry import SchemaRegistryClient  # type: ignore
    from confluent_kafka.schema_registry.avro import AvroSerializer  # type: ignore
    from confluent_kafka.serialization import MessageField, SerializationContext  # type: ignore

    tx_id = transaction_id or str(uuid.uuid4())

    schema_client = SchemaRegistryClient({"url": SCHEMA_REG_URL})

    # Fetch schema from registry (registered by 01_register_schemas.py)
    schema_str = schema_client.get_latest_version(f"{REQUEST_TOPIC}-value").schema.schema_str
    serializer = AvroSerializer(schema_client, schema_str)

    producer = Producer({
        "bootstrap.servers": KAFKA_BOOTSTRAP,
        "security.protocol": "PLAINTEXT",
        "acks": "all",
    })

    payload = {
        "transaction_id":  tx_id,
        "terminal_id":     terminal_id,
        "user_claimed_id": user_id,
        "image_b64":       image_b64,
        "request_ts":      int(datetime.now(timezone.utc).timestamp() * 1000),
    }

    serialized = serializer(
        payload,
        SerializationContext(REQUEST_TOPIC, MessageField.VALUE),
    )

    producer.produce(
        topic=REQUEST_TOPIC,
        key=tx_id.encode(),
        value=serialized,
    )
    producer.flush(timeout=5)
    return tx_id


def main() -> None:
    p = argparse.ArgumentParser(
        prog="send_auth_request",
        description="Publish a test authentication request to auth.requests.",
    )
    p.add_argument("--user-id",   required=True, metavar="ID",
                   help="The user ID being claimed (must be enrolled).")
    p.add_argument("--terminal",  required=True, metavar="ID",
                   help="Terminal identifier, e.g. 'door-lobby-nyc'.")
    p.add_argument("--image",     required=True, metavar="PATH",
                   help="Path to the face image to use as the auth capture.")
    p.add_argument("--repeat",    type=int, default=1, metavar="N",
                   help="Send N requests (for load testing).")
    p.add_argument("--delay-ms",  type=int, default=0, metavar="MS",
                   help="Delay between repeated requests in milliseconds.")
    args = p.parse_args()

    image_b64 = _load_image_b64(args.image)
    print(f"Image loaded: {len(base64.b64decode(image_b64))} bytes\n")

    for i in range(args.repeat):
        tx_id = send_request(args.user_id, args.terminal, image_b64)
        print(
            f"  [{i+1}/{args.repeat}]  Sent -> transaction_id={tx_id}"
            f"  user={args.user_id}  terminal={args.terminal}"
        )
        if args.delay_ms > 0 and i < args.repeat - 1:
            time.sleep(args.delay_ms / 1000)

    print(f"\nDone. Check auth.responses for decisions.")


if __name__ == "__main__":
    main()
