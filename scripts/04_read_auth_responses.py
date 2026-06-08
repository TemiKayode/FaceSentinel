"""
04_read_auth_responses.py
=========================
Consume and pretty-print decisions from the auth.responses topic.

Run this in a separate terminal while the worker is running to watch
decisions appear in real time.

Usage
-----
    python scripts/04_read_auth_responses.py

    # Wait for exactly N responses then exit
    python scripts/04_read_auth_responses.py --count 5

    # Read from the beginning of the topic
    python scripts/04_read_auth_responses.py --from-beginning
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS",   "localhost:9092")
SCHEMA_REG_URL  = os.getenv("KAFKA_SCHEMA_REGISTRY_URL", "http://localhost:8081")
RESPONSE_TOPIC  = os.getenv("KAFKA_RESPONSE_TOPIC",      "auth.responses")


DECISION_COLOR = {
    "GRANTED": "\033[92m",   # green
    "DENIED":  "\033[91m",   # red
}
RESET = "\033[0m"


def main() -> None:
    p = argparse.ArgumentParser(
        prog="read_auth_responses",
        description="Tail the auth.responses topic and display decisions.",
    )
    p.add_argument("--count", type=int, default=0, metavar="N",
                   help="Stop after N responses (0 = run forever).")
    p.add_argument("--from-beginning", action="store_true",
                   help="Read all messages from the start of the topic.")
    args = p.parse_args()

    from confluent_kafka import Consumer, KafkaError  # type: ignore
    from confluent_kafka.schema_registry import SchemaRegistryClient  # type: ignore
    from confluent_kafka.schema_registry.avro import AvroDeserializer  # type: ignore
    from confluent_kafka.serialization import MessageField, SerializationContext  # type: ignore

    schema_client = SchemaRegistryClient({"url": SCHEMA_REG_URL})
    schema_str    = schema_client.get_latest_version(f"{RESPONSE_TOPIC}-value").schema.schema_str
    deserializer  = AvroDeserializer(schema_client, schema_str)

    consumer = Consumer({
        "bootstrap.servers": KAFKA_BOOTSTRAP,
        "security.protocol": "PLAINTEXT",
        "group.id":          f"response-reader-{uuid.uuid4().hex[:8]}",
        "auto.offset.reset": "earliest" if args.from_beginning else "latest",
        "enable.auto.commit": True,
    })
    consumer.subscribe([RESPONSE_TOPIC])

    print(f"Listening on '{RESPONSE_TOPIC}' ... (Ctrl-C to stop)\n")
    received = 0

    try:
        while True:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                print(f"Kafka error: {msg.error()}")
                break

            event = deserializer(
                msg.value(),
                SerializationContext(RESPONSE_TOPIC, MessageField.VALUE),
            )

            decision  = event.get("decision", "?")
            color     = DECISION_COLOR.get(decision, "")
            conf      = event.get("confidence", 0.0)
            reason    = event.get("denial_reason") or ""
            tx_id     = event.get("transaction_id", "?")
            user      = event.get("user_claimed_id", "?")
            terminal  = event.get("terminal_id", "?")
            latency   = event.get("processing_ms", 0)
            region    = event.get("worker_region", "?")

            print(
                f"{color}{'-'*60}{RESET}\n"
                f"  Decision   : {color}{decision}{RESET}"
                + (f"  (reason: {reason})" if reason else f"  (confidence: {conf:.4f})") + "\n"
                f"  User       : {user}\n"
                f"  Terminal   : {terminal}\n"
                f"  TX         : {tx_id}\n"
                f"  Latency    : {latency} ms   Region: {region}\n"
            )

            received += 1
            if args.count > 0 and received >= args.count:
                print(f"Received {received} response(s). Exiting.")
                break

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        consumer.close()


if __name__ == "__main__":
    main()
