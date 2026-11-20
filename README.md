# LLM Chatbot with Key Failover and Response Caching

A full-stack resilient AI chat application featuring automatic API key failover, deterministic disk-persisted response caching, per-session multi-turn history, and failure simulation capabilities.

---

## 1. Live URLs

- **Frontend:** `https://your-frontend-deployment.example.com` *(placeholder)*
- **Backend API:** `https://your-backend-deployment.example.com` *(placeholder)*
- **Health Endpoint:** `https://your-backend-deployment.example.com/health` *(placeholder)*
- **GitHub Repository:** `https://github.com/your-username/llm-chatbot-failover-caching` *(placeholder)*

---

## 2. Run locally

### Prerequisites
- Python 3.11+
- Node.js 20+ & npm
- Docker & Docker Compose *(optional for containerized execution)*

### Option A: With Docker Compose

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
2. Build and start both containers:
   ```bash
   make up
   ```
3. Open the application:
   - **Frontend UI:** [http://localhost:3000](http://localhost:3000)
   - **Backend API:** [http://localhost:8000](http://localhost:8000)
   - **Health Check:** [http://localhost:8000/health](http://localhost:8000/health)
4. Stop containers while preserving database volume:
   ```bash
   make down
   ```
5. View container logs:
   ```bash
   make logs
   ```
6. Run full test suite:
   ```bash
   make test
   ```

---

### Option B: Without Docker (Local Development)

#### 1. Backend Setup
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Start backend server
DATABASE_URL="sqlite:////tmp/app.db" \
LLM_API_KEYS="key_1_val,key_2_val" \
ALLOWED_ORIGINS="http://localhost:5173,http://localhost:3000" \
MOCK_LLM=true \
ENABLE_FAILURE_SIMULATION=true \
CACHE_DIR="/tmp/cache" \
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

#### 2. Frontend Setup
```bash
cd frontend
npm install
npm run dev -- --port 5173
```
Visit [http://localhost:5173](http://localhost:5173) in your browser.

---

## 3. Design decisions

### Explicit Key Loop vs. LiteLLM Router
Instead of relying on the LiteLLM Router's internal black-box scheduling and retry heuristics, we implement an explicit, deterministic failover loop across keys in configuration order (`key_1`, `key_2`, ...):
- **Fast Failover:** Every LiteLLM call sets `num_retries=0` and `timeout=30s`, preventing silent long retry hangs on exhausted or rate-limited keys.
- **Accurate Metric Tracking:** Explicit loops accurately track `fallback_attempts` (how many keys failed before success) and the exact `key_id` that resolved the prompt.
- **Security:** Key values are never logged, stored in message history, or returned in response payloads; only safe symbolic IDs (`key_1`, `key_2`) are exposed.
- **Exhaustion Guard:** If all keys in the pool fail (or return 429/5xx), a custom `AllProvidersUnavailable` exception triggers a clean `503 {"detail": "all providers unavailable"}`.

### Deterministic Response Caching
- **Canonical Cache Key:** Computed as `sha256(model + full_messages_list)`, including system prompt, chronological history, and the new user message.
- **Exclusion of Non-Content Artifacts:** API keys, session IDs, and timestamps are strictly excluded from the hash. An identical prompt in a brand-new session reliably hits `LLM_CACHE`.
- **Survives Process Restarts:** Uses persistent `diskcache` backed by SQLite/files on `/data/cache`.
- **Simulation Bypass:** When `X-Simulate-Failure` is present and active, cache **read** is deliberately bypassed so the failover path is exercised live, while successful live responses still populate the cache.

### SQLite Storage, WAL Mode & Transactions
- **WAL Mode (`PRAGMA journal_mode=WAL`):** Enables concurrent readers and a single writer without database locking delays.
- **Atomic Exchange Persistence:** Both user message and assistant reply (with complete metadata) are saved in a single atomic transaction via `save_exchange()`.
- **Zero Incomplete State:** If request validation fails (422) or all providers fail (503), nothing is saved to SQLite, keeping history pure and free of failed or corrupted exchanges.
- **Volume Persistence:** In Docker, SQLite data resides in `/data/app.db` on the `app_data` named volume, ensuring message history survives `docker compose down` and container rebuilds.

---

## 4. Verification

Use the following curl commands to verify each capability:

### 1. Health Check
```bash
curl -s http://localhost:8000/health | python3 -m json.tool
```
*Expected:* Status 200, `"status": "ok"`, `"keys_configured": 2`, `"cache_enabled": true`.

### 2. Live Inference (First Message)
```bash
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "sess_demo", "message": "What is the boiling point of nitrogen?"}' | python3 -m json.tool
```
*Expected:* `"resolution_source": "LIVE_INFERENCE"`, `"key_id": "key_1"`, `"fallback_attempts": 0`.

### 3. Cache Hit (New Session, Identical Message)
```bash
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "sess_demo_new", "message": "What is the boiling point of nitrogen?"}' | python3 -m json.tool
```
*Expected:* `"resolution_source": "LLM_CACHE"`, `"key_id": null`, `"fallback_attempts": 0`, identical reply text, significantly reduced `latency_ms`.

### 4. Key Failover Simulation (`key_1=429`)
```bash
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-Simulate-Failure: key_1=429" \
  -d '{"session_id": "sess_demo_fo", "message": "Explain quantum tunneling."}' | python3 -m json.tool
```
*Expected:* Status 200, `"key_id": "key_2"`, `"fallback_attempts": 1`.

### 5. All Providers Unavailable (`*=429`)
```bash
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-Simulate-Failure: *=429" \
  -d '{"session_id": "sess_demo_fail", "message": "Should not be stored."}' | python3 -m json.tool
```
*Expected:* Status 503 `{"detail": "all providers unavailable"}`.

### 6. Verify Failed Request Not Stored
```bash
curl -s http://localhost:8000/sessions/sess_demo_fail/messages | python3 -m json.tool
```
*Expected:* Status 404 `{"detail": "session not found"}`.

### UI Verification
1. Open the UI at [http://localhost:3000](http://localhost:3000).
2. Send a message: Observe the **LIVE_INFERENCE** green badge, latency, model, and token metrics.
3. Refresh the page (starts a new session) and send the exact same message: Observe the **LLM_CACHE** blue badge and near-zero latency.
4. Click **⚙ Dev Tools**, input `*=429`, and send a message: Observe a clean red error notification stating `"All providers are currently unavailable, please try again"`.

---

## 5. Setup Used

- **AI Assistant:** Google Antigravity (Powered by Google DeepMind Gemini models & Claude)
- **Primary IDE:** Antigravity IDE / VS Code
- **Runtime Environment:** Python 3.11, Node.js v22, macOS Darwin

---

## 6. Bonus features

- **Interactive Dev Tools Panel:** Collapsible UI control allowing on-the-fly injection of `X-Simulate-Failure` headers for interactive demoing.
- **Custom System Prompt Override:** Input field allowing users to define a custom session-level system prompt which deterministically alters LLM context and cache keys.
- **Persistent Disk Caching:** Disk-backed cache surviving application restarts, avoiding in-memory loss.
- **Frontend Error Boundary:** React class error boundary preventing blank screen crashes on unexpected UI errors.
- **Comprehensive Pytest Suite:** 49 automated unit and integration tests covering happy paths, failover cascades, token metrics, cache isolation, and edge-case validations.
- **Docker Healthchecks:** Automated healthcheck polling `/health` ensuring backend availability before frontend routing begins.
