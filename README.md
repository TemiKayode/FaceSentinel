# FaceSentinel — Kafka-Native Consent-Based Biometric Authentication

> **Consent-first, intelligence-grade face verification pipeline.**  
> User presents face at a terminal → Kafka → ArcFace 1:1 verification → Kafka → GRANTED / DENIED  
> Built for high-security environments: per-tenant isolation, HMAC-signed audit trail, anomaly detection, GDPR right-to-be-forgotten, 2027-ready model upgrade path.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Stack](https://img.shields.io/badge/stack-Kafka%20%7C%20ArcFace%20%7C%20Qdrant-orange)

**GitHub topics:** `biometric-authentication` `face-recognition` `arcface` `insightface` `kafka` `qdrant` `python` `access-control` `gdpr` `consent` `zero-trust` `vector-database` `anomaly-detection` `otel`

---

## Project Layout

```
FaceSentinel/
|-- docker-compose.yml          Local infrastructure (Kafka, Schema Registry, Qdrant, Kafka-UI)
|-- .env.local                  Environment template for local dev
|-- workers/
|   |-- biometric_worker.py     Core auth worker — SecurityTiers, twin-risk, anomaly detection, HMAC audit
|   `-- requirements.txt        Worker Python dependencies
|-- scripts/
|   |-- 01_register_schemas.py  Register Avro schemas (run once after docker compose up)
|   |-- 02_enroll_face.py       Enroll faces into Qdrant
|   |-- 03_send_auth_request.py Send test auth requests to Kafka
|   `-- 04_read_auth_responses.py  Watch decisions in real time
|-- k8n/
|   `-- biometric-worker.yaml   Kubernetes + KEDA deployment manifest
|-- face_detector.py            Standalone ensemble face detection CLI (separate tool)
|-- face_indexer.py             Standalone local face embedding + vector DB CLI (separate tool)
|-- face_indexer_requirements.txt  Dependencies for face_indexer.py
|-- requirements.txt            Dependencies for face_detector.py
`-- verify_indexer.py           Test suite for face_indexer.py
```

---

## Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Python | 3.10 or 3.11 | https://www.python.org/downloads/ |
| Docker Desktop | 4.x+ | https://www.docker.com/products/docker-desktop/ |
| docker compose | v2 plugin (bundled with Docker Desktop) | included |

> **Windows note**: Always use `.\venv\Scripts\python.exe`, never the bare `python` command,
> which may resolve to a different Python version and break ML package imports.

---

## Quick Start (Windows PowerShell)

### Step 1 — Create virtual environment

```powershell
cd C:\Users\hp\Downloads\Tool
```

If `venv\` does not exist yet:
```powershell
py -3.11 -m venv venv
```

> If you see `Permission denied: venv\Scripts\python.exe`, the venv already exists
> and a Python process (e.g. the worker) is holding a file lock. Stop it first,
> then delete the folder and retry: `Remove-Item -Recurse -Force venv`

Activate the venv:
```powershell
.\venv\Scripts\Activate.ps1
```

### Step 2 — Install dependencies

**CPU (default, works without a GPU):**
```powershell
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\venv\Scripts\python.exe -m pip install -r workers\requirements.txt
```

**GPU (CUDA 12.1):**
```powershell
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
.\venv\Scripts\python.exe -m pip install onnxruntime-gpu
.\venv\Scripts\python.exe -m pip install -r workers\requirements.txt
```

### Step 3 — Start infrastructure

```powershell
docker compose up -d
```

Wait ~20 seconds, then verify all containers are healthy:

```powershell
docker compose ps
```

Expected:
```
NAME                   STATUS
auth-kafka             running (healthy)
auth-schema-registry   running (healthy)
auth-qdrant            running (healthy)
auth-kafka-ui          running
```

Kafka UI (optional): http://localhost:8085

### Step 4 — Load environment variables

```powershell
Get-Content .env.local | ForEach-Object {
    if ($_ -match '^\s*([^#][^=]+)=(.*)$') {
        [System.Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim())
    }
}
```

Verify:
```powershell
$env:KAFKA_BOOTSTRAP_SERVERS   # should print: localhost:9092
```

### Step 5 — Register Avro schemas (once)

```powershell
.\venv\Scripts\python.exe scripts\01_register_schemas.py
```

Expected:
```
Registering schemas at http://localhost:8081 ...

  OK  auth.requests-value  ->  schema id 1
  OK  auth.responses-value ->  schema id 2

Done.
```

> This step is idempotent. Safe to run again if you restart containers.

### Step 6 — Create Kafka topics (once)

```powershell
docker exec auth-kafka kafka-topics --bootstrap-server kafka:29092 --create --if-not-exists --topic auth.requests      --partitions 4 --replication-factor 1
docker exec auth-kafka kafka-topics --bootstrap-server kafka:29092 --create --if-not-exists --topic auth.responses     --partitions 4 --replication-factor 1
docker exec auth-kafka kafka-topics --bootstrap-server kafka:29092 --create --if-not-exists --topic auth.requests.dlq  --partitions 2 --replication-factor 1
docker exec auth-kafka kafka-topics --bootstrap-server kafka:29092 --create --if-not-exists --topic auth.audit         --partitions 4 --replication-factor 1
```

`auth.audit` receives a structured JSON event for every decision (GRANTED and DENIED), including face_hash, model version, liveness result, and tenant ID. Consumers such as a SIEM or compliance dashboard can subscribe to this topic.

### Step 7 — Pre-download ArcFace model (~300 MB, once)

```powershell
.\venv\Scripts\python.exe -c "
from insightface.app import FaceAnalysis
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1, det_size=(640,640))
print('Model ready.')
"
```

### Step 8 — Enroll a test face

Place a clear, frontal JPEG of a person in `test_faces\` then:

```powershell
.\venv\Scripts\python.exe scripts\02_enroll_face.py `
    --user-id         "emp-001" `
    --name            "Alice Smith" `
    --image           "test_faces\alice.jpg" `
    --consent-version "1.0" `
    --consent-method  "written"
```

Enrollment runs quality gates before storing: face count (exactly 1), sharpness
(Laplacian variance >= 100), and pose (yaw/pitch <= 30 degrees). The embedding
payload includes model name + version so stale records can be detected after upgrades.

Enroll a second identity for negative-match tests:
```powershell
.\venv\Scripts\python.exe scripts\02_enroll_face.py `
    --user-id "emp-002" `
    --name    "Bob Jones" `
    --image   "test_faces\carol.jpg"
```

List enrolled faces (shows model version and consent status):
```powershell
.\venv\Scripts\python.exe scripts\02_enroll_face.py --list
```

Revoke consent (soft delete -- worker denies with CONSENT_REVOKED, record kept for audit):
```powershell
.\venv\Scripts\python.exe scripts\02_enroll_face.py --revoke "emp-001"
```

Hard delete (GDPR right-to-be-forgotten -- removes from Qdrant):
```powershell
.\venv\Scripts\python.exe scripts\02_enroll_face.py --forget "emp-001"
```

Per-tenant enrollment (collection becomes `enrolled_faces_acme`):
```powershell
.\venv\Scripts\python.exe scripts\02_enroll_face.py `
    --tenant "acme" --user-id "emp-001" --name "Alice" --image "test_faces\alice.jpg"
```

### Step 9 — Start the worker

In a dedicated PowerShell window (re-run Step 4 to load env first):

```powershell
.\venv\Scripts\python.exe -u workers\biometric_worker.py
```

Expected startup log:
```
[INFO] auth.worker  Health server listening on :8080
[INFO] auth.worker  ArcFace model loaded on CPU.
[INFO] auth.worker  Connected to Qdrant at localhost.
[INFO] auth.worker  Subscribed to 'auth.requests'.
[INFO] auth.worker  Auth worker online - region=us-east-1-local
```

Health endpoints (in a new window):
```powershell
Invoke-RestMethod http://localhost:8080/healthz   # ok
Invoke-RestMethod http://localhost:8080/readyz    # ready
```

---

## Running Tests

Open two additional PowerShell windows and load env in each (Step 4).

### Window B — Watch decisions in real time

```powershell
.\venv\Scripts\python.exe -u scripts\04_read_auth_responses.py
```

### Window C — Send auth requests

**Test 1: correct user (expect GRANTED)**
```powershell
.\venv\Scripts\python.exe scripts\03_send_auth_request.py `
    --user-id  "emp-001" `
    --terminal "door-lobby" `
    --image    "test_faces\alice.jpg"
```

**Test 2: wrong face for claimed ID (expect DENIED - BELOW_THRESHOLD)**
```powershell
.\venv\Scripts\python.exe scripts\03_send_auth_request.py `
    --user-id  "emp-001" `
    --terminal "door-lobby" `
    --image    "test_faces\carol.jpg"
```

**Test 3: unknown user (expect DENIED - USER_NOT_ENROLLED)**
```powershell
.\venv\Scripts\python.exe scripts\03_send_auth_request.py `
    --user-id  "emp-999" `
    --terminal "door-lobby" `
    --image    "test_faces\alice.jpg"
```

**Test 4: load test (10 back-to-back requests)**
```powershell
.\venv\Scripts\python.exe scripts\03_send_auth_request.py `
    --user-id emp-001 --terminal door-lobby `
    --image test_faces\alice.jpg `
    --repeat 10 --delay-ms 100
```

**Read all historical responses (from the beginning of the topic):**
```powershell
.\venv\Scripts\python.exe scripts\04_read_auth_responses.py --from-beginning --count 10
```

### Expected output in Window B

```
------------------------------------------------------------
  Decision   : GRANTED  (confidence: 0.9838)
  User       : emp-001
  Terminal   : door-lobby
  TX         : 3f7a1b2c-...
  Latency    : 312 ms   Region: us-east-1-local

------------------------------------------------------------
  Decision   : DENIED  (reason: BELOW_THRESHOLD)
  User       : emp-001
  Terminal   : door-lobby
  TX         : 8e2d4c9a-...
  Latency    : 298 ms   Region: us-east-1-local
```

---

## Kafka UI Inspection

Open http://localhost:8085

- **Topics -> auth.requests** — every inbound auth event
- **Topics -> auth.responses** — every decision (GRANTED / DENIED)
- **Topics -> auth.requests.dlq** — malformed / unprocessable messages
- **Consumer Groups -> auth-worker-local** — partition lag (should stay at 0)

---

## Adjusting Thresholds

Edit `.env.local`, then reload env (Step 4) and restart the worker (Step 9):

| Variable | Default | Effect |
|----------|---------|--------|
| `MATCH_THRESHOLD` | `0.75` | Cosine similarity required for GRANTED. Raise to `0.80`+ for stricter matching. |
| `DET_SCORE_MIN` | `0.75` | Minimum face detection confidence. Raise to `0.85`+ for live cameras. |
| `RATE_LIMIT_MAX` | `10` | Max auth attempts per terminal per 30-second window. |
| `INFERENCE_TIMEOUT_S` | `10.0` | Worker kills hung ONNX inference after this many seconds. |
| `MIN_BLUR_SCORE` | `100.0` | Laplacian sharpness minimum for enrollment. Lower for low-quality cameras. |
| `MAX_POSE_ANGLE` | `30.0` | Max yaw/pitch degrees allowed at enrollment. Lower for stricter frontal requirement. |
| `LIVENESS_CHECK_ENABLED` | `false` | Enable MiniFAS anti-spoofing (requires `pip install deepface`). |
| `MODEL_NAME` | `buffalo_l` | ArcFace model pack. Stored in Qdrant payload for version tracking. |
| `MODEL_VERSION` | `insightface-0.7.3` | Pinned version string. Update when upgrading InsightFace. |
| `TENANT_ID` | _(empty)_ | If set, uses `enrolled_faces_{TENANT_ID}` collection for isolation. |
| `QDRANT_ENROLLED_COLLECTION_V2` | _(empty)_ | Dual-index migration: worker checks both collections and takes the higher score. Unset once re-enrollment is complete. |

---

## Stopping Everything

```powershell
# Stop worker: Ctrl-C in worker window

# Stop containers (keeps Qdrant data on disk)
docker compose down

# Full wipe including enrolled faces
docker compose down -v
Remove-Item -Recurse -Force .\data\qdrant
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `KAFKA_BOOTSTRAP_SERVERS not set` | Env not loaded | Re-run Step 4 |
| `Schema not found` | Schemas not registered | Re-run Step 5 |
| `DENIED: NO_FACE_DETECTED` | Image quality / DET_SCORE_MIN too high | Use clearer photo or lower `DET_SCORE_MIN` to `0.70` |
| `DENIED: USER_NOT_ENROLLED` | User ID not in Qdrant | Run Step 8 for that user_id |
| `DENIED: BELOW_THRESHOLD` | Different face than enrolled | Expected for mismatched identity tests |
| `Connection refused :9092` | Kafka not running | `docker compose up -d kafka` |
| `Connection refused :6333` | Qdrant not running | `docker compose up -d qdrant` |
| `ModuleNotFoundError: authlib` | Missing pip dep | `.\venv\Scripts\python.exe -m pip install -r workers\requirements.txt` |
| Worker exits immediately | Topics missing | Run Step 6 (now includes auth.audit) |
| `[SSL: WRONG_VERSION_NUMBER]` | QDRANT_HTTPS=true for HTTP server | Set `QDRANT_HTTPS=false` in .env.local (already set) |
| OTel `StatusCode.UNAVAILABLE` errors | No OTel collector locally | Set `OTEL_SDK_DISABLED=true` in .env.local (already set) |
| Enrollment fails: "Image too blurry" | Laplacian score below MIN_BLUR_SCORE | Use a sharper photo or lower `MIN_BLUR_SCORE` to `50.0` |
| Enrollment fails: "Face not frontal" | Pose angle > MAX_POSE_ANGLE | Use a forward-facing portrait or raise `MAX_POSE_ANGLE` to `45.0` |
| `DENIED: CONSENT_REVOKED` | User consent was soft-revoked | Re-enroll to restore access |
| `DENIED: ENROLLMENT_EXPIRED` | Enrollment older than tier limit | Re-enroll user (enhanced=180d, critical=90d) |
| `DENIED: SECONDARY_FACTOR_REQUIRED` | User requires MFA but no token sent | Include `secondary_factor_token` in request |
| `DENIED: SECONDARY_FACTOR_INVALID` | Wrong PIN/TOTP code | Verify PIN or regenerate TOTP secret |
| Anomaly score >= 0.5 in auth.audit | Unusual access pattern detected | Review audit trail; alert security team |
| `deepface not installed -- liveness skipped` | LIVENESS_CHECK_ENABLED=true but no deepface | `pip install deepface` or set `LIVENESS_CHECK_ENABLED=false` |
| Model mismatch warning in worker log | Enrolled with old model version | Re-enroll user or configure dual-index migration (`QDRANT_ENROLLED_COLLECTION_V2`) |

---

## Standalone Tools

### face_detector.py — Ensemble face detection CLI

Install dependencies:
```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

Usage:
```powershell
# Single image
.\venv\Scripts\python.exe face_detector.py --input photo.jpg --output out.jpg --verbose

# Folder
.\venv\Scripts\python.exe face_detector.py --input ./images/ --output ./out/ --json results.json

# Webcam
.\venv\Scripts\python.exe face_detector.py --input 0

# Video file
.\venv\Scripts\python.exe face_detector.py --input clip.mp4 --output annotated.mp4 --save-crops ./crops/
```

### face_indexer.py — Local face embedding + vector DB

Install dependencies:
```powershell
.\venv\Scripts\python.exe -m pip install -r face_indexer_requirements.txt
```

Usage:
```powershell
# Register a face
.\venv\Scripts\python.exe face_indexer.py register --name "Alice" --image photo.jpg

# Search for a face
.\venv\Scripts\python.exe face_indexer.py search --image query.jpg

# List all registered
.\venv\Scripts\python.exe face_indexer.py list

# Run test suite
.\venv\Scripts\python.exe verify_indexer.py
```

---

## Intelligence & High-Security Deployment

See [INTELLIGENCE_GUIDE.md](INTELLIGENCE_GUIDE.md) for:
- Security tier configuration (STANDARD / ENHANCED / CRITICAL)
- Identical twins and doppelganger disambiguation strategy
- Threat model coverage (deepfakes, relay attacks, insider fraud)
- 2027 technology roadmap (AdaFace, ViT, iris fusion, homomorphic encryption)
- Complete scenario walkthroughs (embassy, SCIF, data center, exec protection)
- Anomaly detection tuning and HMAC audit verification

**Quick intelligence-grade enrollment:**

```powershell
# 3-angle composite template + elevated twin-risk threshold + MFA
.\venv\Scripts\python.exe scripts\02_enroll_face.py `
    --user-id "agent-001" --name "Jane Doe" `
    --image "test_faces\alice.jpg" `
    --extra-image "test_faces\alice_left.jpg" `
    --security-clearance critical `
    --twin-risk `
    --require-secondary-factor `
    --secondary-factor-hash <sha256-of-pin>
```

---

## Production Deployment (Kubernetes)

See [k8n/biometric-worker.yaml](k8n/biometric-worker.yaml) for the full manifest including:

- GPU node affinity (g5.xlarge / A10G)
- KEDA autoscaling (2-50 replicas based on Kafka lag)
- Pod Disruption Budget (minAvailable: 2)
- Network Policy (deny-by-default, allow only Kafka/Qdrant/OTel egress)
- ReadOnly root filesystem + non-root user
- Vault secret injection for Kafka SASL and Qdrant API key

Before deploying, update the container image reference in the yaml:
```yaml
image: 123456789.dkr.ecr.us-east-1.amazonaws.com/auth-worker:3.0.0
```

And set `QDRANT_HTTPS=true` (already set in the k8n manifest) since production Qdrant uses TLS.
