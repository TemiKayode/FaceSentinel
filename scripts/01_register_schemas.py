"""
01_register_schemas.py
======================
Register the AuthRequest and AuthResponse Avro schemas with the local
Schema Registry before starting the worker or sending test messages.

Run once after `docker compose up -d`:
    python scripts/01_register_schemas.py
"""

import sys
import requests

SCHEMA_REGISTRY_URL = "http://localhost:8081"

AUTH_REQUEST_SCHEMA = """{
  "type": "record",
  "name": "AuthRequest",
  "namespace": "com.company.biometric",
  "fields": [
    {"name": "transaction_id",        "type": "string"},
    {"name": "terminal_id",           "type": "string"},
    {"name": "user_claimed_id",       "type": "string"},
    {"name": "image_b64",             "type": "string"},
    {"name": "request_ts",            "type": "long"},
    {"name": "secondary_factor_token","type": ["null", "string"], "default": null}
  ]
}"""

AUTH_RESPONSE_SCHEMA = """{
  "type": "record",
  "name": "AuthResponse",
  "namespace": "com.company.biometric",
  "fields": [
    {"name": "transaction_id",  "type": "string"},
    {"name": "terminal_id",     "type": "string"},
    {"name": "user_claimed_id", "type": "string"},
    {"name": "decision",        "type": {"type": "enum", "name": "Decision",
                                 "symbols": ["GRANTED", "DENIED"]}},
    {"name": "confidence",      "type": "float",              "default": 0.0},
    {"name": "denial_reason",   "type": ["null", "string"],   "default": null},
    {"name": "processing_ms",   "type": "long"},
    {"name": "worker_region",   "type": "string"},
    {"name": "response_ts",     "type": "long"}
  ]
}"""

SUBJECTS = {
    "auth.requests-value":  AUTH_REQUEST_SCHEMA,
    "auth.responses-value": AUTH_RESPONSE_SCHEMA,
}


def register(subject: str, schema: str) -> int:
    url = f"{SCHEMA_REGISTRY_URL}/subjects/{subject}/versions"
    resp = requests.post(
        url,
        json={"schema": schema},
        headers={"Content-Type": "application/vnd.schemaregistry.v1+json"},
        timeout=10,
    )
    if resp.status_code in (200, 201):
        schema_id: int = resp.json()["id"]
        print(f"  OK  {subject}  ->  schema id {schema_id}")
        return schema_id
    else:
        print(f"  FAIL  {subject}  ->  HTTP {resp.status_code}: {resp.text}")
        sys.exit(1)


def main() -> None:
    print(f"Registering schemas at {SCHEMA_REGISTRY_URL} ...\n")
    for subject, schema in SUBJECTS.items():
        register(subject, schema)
    print("\nDone. You can now start the worker and run test scripts.")


if __name__ == "__main__":
    main()
