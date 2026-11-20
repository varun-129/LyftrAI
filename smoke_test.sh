#!/usr/bin/env bash
# ==============================================================================
# smoke_test.sh – Automated End-to-End Smoke Test for LLM Chatbot Backend
#
# Usage:
#   ./smoke_test.sh <BACKEND_URL>
# Example:
#   ./smoke_test.sh http://localhost:8000
#   ./smoke_test.sh https://llm-chatbot-backend.onrender.com
# ==============================================================================

set -u

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

if [ $# -lt 1 ]; then
  echo -e "${RED}Usage: $0 <BACKEND_URL>${NC}"
  echo "Example: $0 http://localhost:8000"
  exit 1
fi

RAW_URL="$1"
# Strip trailing slash
BASE_URL="${RAW_URL%/}"

echo -e "\n${CYAN}======================================================${NC}"
echo -e "${CYAN} Running Smoke Tests against: ${YELLOW}${BASE_URL}${NC}"
echo -e "${CYAN}======================================================${NC}\n"

PASS_COUNT=0
FAIL_COUNT=0

record_pass() {
  local name="$1"
  echo -e "  [${GREEN}PASS${NC}] ${name}"
  PASS_COUNT=$((PASS_COUNT + 1))
}

record_fail() {
  local name="$1"
  local reason="$2"
  echo -e "  [${RED}FAIL${NC}] ${name}"
  echo -e "         ${YELLOW}Reason: ${reason}${NC}"
  FAIL_COUNT=$((FAIL_COUNT + 1))
}

RANDOM_SUFFIX="$(date +%s)_$((RANDOM % 9000 + 1000))"
SESS_1="smoke_sess1_${RANDOM_SUFFIX}"
SESS_2="smoke_sess2_${RANDOM_SUFFIX}"
SESS_FAILOVER="smoke_fo_${RANDOM_SUFFIX}"
SESS_503="smoke_503_${RANDOM_SUFFIX}"
COMMON_MSG="What is the chemical symbol for gold?"

# ------------------------------------------------------------------------------
# 1. Health check
# ------------------------------------------------------------------------------
echo -e "${CYAN}1. Testing GET /health...${NC}"
HEALTH_RES=$(curl -s -w "\n%{http_code}" "${BASE_URL}/health")
HEALTH_CODE=$(echo "$HEALTH_RES" | tail -n 1)
HEALTH_BODY=$(echo "$HEALTH_RES" | sed '$d')

if [ "$HEALTH_CODE" = "200" ] && echo "$HEALTH_BODY" | grep -q '"status":"ok"'; then
  record_pass "GET /health returned HTTP 200 with status: ok"
else
  record_fail "GET /health" "Expected 200 with status: ok, got code ${HEALTH_CODE}, body: ${HEALTH_BODY}"
fi

# ------------------------------------------------------------------------------
# 2. Normal Chat (LIVE_INFERENCE)
# ------------------------------------------------------------------------------
echo -e "\n${CYAN}2. Testing POST /chat (Live Inference)...${NC}"
CHAT_1_RES=$(curl -s -w "\n%{http_code}" -X POST "${BASE_URL}/chat" \
  -H "Content-Type: application/json" \
  -d "{\"session_id\":\"${SESS_1}\",\"message\":\"${COMMON_MSG}\",\"system_prompt\":\"You are a chemist.\"}")
CHAT_1_CODE=$(echo "$CHAT_1_RES" | tail -n 1)
CHAT_1_BODY=$(echo "$CHAT_1_RES" | sed '$d')

if [ "$CHAT_1_CODE" = "200" ] && echo "$CHAT_1_BODY" | grep -q '"resolution_source":"LIVE_INFERENCE"'; then
  record_pass "POST /chat returned 200 with LIVE_INFERENCE"
else
  record_fail "POST /chat (Live Inference)" "Expected 200 with LIVE_INFERENCE, got code ${CHAT_1_CODE}, body: ${CHAT_1_BODY}"
fi

# ------------------------------------------------------------------------------
# 3. Cache Hit (New session, identical message)
# ------------------------------------------------------------------------------
echo -e "\n${CYAN}3. Testing POST /chat (Cache Hit in new session)...${NC}"
CHAT_2_RES=$(curl -s -w "\n%{http_code}" -X POST "${BASE_URL}/chat" \
  -H "Content-Type: application/json" \
  -d "{\"session_id\":\"${SESS_2}\",\"message\":\"${COMMON_MSG}\",\"system_prompt\":\"You are a chemist.\"}")
CHAT_2_CODE=$(echo "$CHAT_2_RES" | tail -n 1)
CHAT_2_BODY=$(echo "$CHAT_2_RES" | sed '$d')

if [ "$CHAT_2_CODE" = "200" ] && echo "$CHAT_2_BODY" | grep -q '"resolution_source":"LLM_CACHE"' && echo "$CHAT_2_BODY" | grep -q '"key_id":null'; then
  record_pass "POST /chat returned 200 with LLM_CACHE and key_id: null"
else
  record_fail "POST /chat (Cache Hit)" "Expected 200 with LLM_CACHE & key_id: null, got code ${CHAT_2_CODE}, body: ${CHAT_2_BODY}"
fi

# ------------------------------------------------------------------------------
# 4. Key Failover (key_1=429 -> key_2)
# ------------------------------------------------------------------------------
echo -e "\n${CYAN}4. Testing POST /chat with X-Simulate-Failure: key_1=429...${NC}"
FO_RES=$(curl -s -w "\n%{http_code}" -X POST "${BASE_URL}/chat" \
  -H "Content-Type: application/json" \
  -H "X-Simulate-Failure: key_1=429" \
  -d "{\"session_id\":\"${SESS_FAILOVER}\",\"message\":\"Tell me about failover mechanisms.\"}")
FO_CODE=$(echo "$FO_RES" | tail -n 1)
FO_BODY=$(echo "$FO_RES" | sed '$d')

if [ "$FO_CODE" = "200" ] && echo "$FO_BODY" | grep -q '"key_id":"key_2"' && echo "$FO_BODY" | grep -q '"fallback_attempts":1'; then
  record_pass "POST /chat key_1=429 successfully failed over to key_2 (fallback_attempts: 1)"
else
  record_fail "POST /chat failover" "Expected 200 with key_id: key_2 and fallback_attempts: 1, got code ${FO_CODE}, body: ${FO_BODY}"
fi

# ------------------------------------------------------------------------------
# 5. Wildcard Provider Failure (*=429 -> 503)
# ------------------------------------------------------------------------------
echo -e "\n${CYAN}5. Testing POST /chat with X-Simulate-Failure: *=429 (Expected 503)...${NC}"
FAIL_RES=$(curl -s -w "\n%{http_code}" -X POST "${BASE_URL}/chat" \
  -H "Content-Type: application/json" \
  -H "X-Simulate-Failure: *=429" \
  -d "{\"session_id\":\"${SESS_503}\",\"message\":\"This request must fail with 503.\"}")
FAIL_CODE=$(echo "$FAIL_RES" | tail -n 1)
FAIL_BODY=$(echo "$FAIL_RES" | sed '$d')

if [ "$FAIL_CODE" = "503" ] && echo "$FAIL_BODY" | grep -q '"detail":"all providers unavailable"'; then
  record_pass "POST /chat *=429 returned HTTP 503 with {\"detail\":\"all providers unavailable\"}"
else
  record_fail "POST /chat *=429" "Expected 503 with detail: all providers unavailable, got code ${FAIL_CODE}, body: ${FAIL_BODY}"
fi

# ------------------------------------------------------------------------------
# 6. History check: Session 1 has messages; Failed 503 session was never created (404)
# ------------------------------------------------------------------------------
echo -e "\n${CYAN}6. Testing GET /sessions/{session_id}/messages...${NC}"
HIST_RES=$(curl -s -w "\n%{http_code}" "${BASE_URL}/sessions/${SESS_1}/messages")
HIST_CODE=$(echo "$HIST_RES" | tail -n 1)
HIST_BODY=$(echo "$HIST_RES" | sed '$d')

if [ "$HIST_CODE" = "200" ] && echo "$HIST_BODY" | grep -q '"role":"user"' && echo "$HIST_BODY" | grep -q '"role":"assistant"'; then
  record_pass "GET /sessions/${SESS_1}/messages returned history with user and assistant turns"
else
  record_fail "GET /sessions/${SESS_1}/messages" "Expected 200 with user/assistant turns, got code ${HIST_CODE}, body: ${HIST_BODY}"
fi

HIST_503_RES=$(curl -s -w "\n%{http_code}" "${BASE_URL}/sessions/${SESS_503}/messages")
HIST_503_CODE=$(echo "$HIST_503_RES" | tail -n 1)
HIST_503_BODY=$(echo "$HIST_503_RES" | sed '$d')

if [ "$HIST_503_CODE" = "404" ] && echo "$HIST_503_BODY" | grep -q '"detail":"session not found"'; then
  record_pass "GET /sessions/${SESS_503}/messages returned 404 (failed 503 request was never saved)"
else
  record_fail "GET /sessions/${SESS_503}/messages" "Expected 404 session not found, got code ${HIST_503_CODE}, body: ${HIST_503_BODY}"
fi

# ------------------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------------------
echo -e "\n${CYAN}======================================================${NC}"
echo -e " Smoke Test Summary:"
echo -e "   Total Passed: ${GREEN}${PASS_COUNT}${NC}"
echo -e "   Total Failed: ${RED}${FAIL_COUNT}${NC}"
echo -e "${CYAN}======================================================${NC}"

if [ "$FAIL_COUNT" -eq 0 ]; then
  echo -e "${GREEN}All smoke tests passed successfully!${NC}\n"
  exit 0
else
  echo -e "${RED}Some smoke tests failed. Please review errors above.${NC}\n"
  exit 1
fi
