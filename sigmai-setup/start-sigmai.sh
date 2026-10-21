#!/usr/bin/env bash
# start-sigmai.sh — Batch pool StartTask for pool-sigmai.
#
# Runs ONCE each time a node boots (and again after a reboot — so it must be idempotent).
# Batch schedules no bridge task on the node until this exits 0 (waitForSuccess=true).
#
# Brings up the Sigma-I stack (api + worker + redis) with docker compose and leaves it
# running; every run is then just a small bridge.sh task talking to http://127.0.0.1:8000.
#
# Inputs (StartTask environmentSettings — see documents/SigmaI/SigmaI-03-Azure-Resources.md):
#   STORAGE_ACCOUNT      storage account name (D-Wave key lives in a private container there)
#   SECRETS_CONTAINER    private container holding the D-Wave token blob      (default: sigmai-secrets)
#   DWAVE_SECRET_BLOB    blob name of the token                               (default: dwave-token.txt)
#   DWAVE_ENABLED        "true" → fetch the token; "false" → run SCIP-only    (default: false)
#   POOL_MI_RESOURCE_ID  resource id of the pool's user-assigned MI (IMDS token + ACR login)
#   ACR_LOGIN_SERVER     e.g. myacr.azurecr.io
#   SIGMAI_IMAGE         full image ref, e.g. myacr.azurecr.io/sigmai/screenas-workerassign-api:0.1.0
#   REDIS_IMAGE          full image ref, e.g. myacr.azurecr.io/mirror/redis:7-alpine
# Setup files (docker-compose.azure.yml, bridge.sh, wheels/*.whl) arrive via StartTask
# resourceFiles; SETUP_DIR can override the auto-detected location (used for local tests).
#
# Secrets: API_KEY is generated here and only ever written to chmod-600 files on this node.
# The D-Wave token is read from Blob via the node's Managed Identity — never put in
# environmentSettings (visible to anyone with Batch read access).
set -euo pipefail

log() { echo "[start-sigmai $(date -u +%H:%M:%S)] $*"; }
die() { log "ERROR: $*"; dump_logs || true; exit 1; }

SIGMAI_HOME="${SIGMAI_HOME:-${AZ_BATCH_NODE_SHARED_DIR:-/mnt/sigmai-shared}/sigmai}"
SIGMAI_PORT="${SIGMAI_PORT:-8000}"
SECRETS_CONTAINER="${SECRETS_CONTAINER:-sigmai-secrets}"
DWAVE_SECRET_BLOB="${DWAVE_SECRET_BLOB:-dwave-token.txt}"
DWAVE_ENABLED="${DWAVE_ENABLED:-false}"
READY_TIMEOUT_SEC="${READY_TIMEOUT_SEC:-1200}"      # 20 min: first start pip-installs the wheel
SKIP_ACR_LOGIN="${SKIP_ACR_LOGIN:-false}"           # local tests use locally loaded images
PY=""
for c in python3 python; do   # first one that actually runs (Windows has a non-working python3 stub)
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys' >/dev/null 2>&1; then PY="$c"; break; fi
done
[ -n "$PY" ] || { log "ERROR: python3 not found"; exit 1; }

COMPOSE=()
dump_logs() {
  [ ${#COMPOSE[@]} -gt 0 ] || return 0
  log "---- docker compose ps ----";  "${COMPOSE[@]}" ps 2>&1 | tail -20 || true
  log "---- api logs (tail) ----";    "${COMPOSE[@]}" logs --tail 60 api 2>&1 || true
  log "---- worker logs (tail) ----"; "${COMPOSE[@]}" logs --tail 80 worker 2>&1 || true
}

# ── IMDS token for the pool's Managed Identity ───────────────────────────
imds_token() {  # $1 = resource URI
  local res="$1" mi_q url
  mi_q="$("$PY" -c 'import sys,urllib.parse;print(urllib.parse.quote(sys.argv[1],safe=""))' "${POOL_MI_RESOURCE_ID:-}")"
  url="http://169.254.169.254/metadata/identity/oauth2/token?api-version=2018-02-01&resource=${res}"
  [ -n "${POOL_MI_RESOURCE_ID:-}" ] && url="${url}&msi_res_id=${mi_q}"
  curl -sf -H "Metadata: true" "$url" \
    | "$PY" -c 'import sys,json;print(json.load(sys.stdin)["access_token"])'
}

# ── 1. Locate setup files and lay out SIGMAI_HOME ────────────────────────
log "SIGMAI_HOME=$SIGMAI_HOME"
mkdir -p "$SIGMAI_HOME/data" "$SIGMAI_HOME/wheels"

if [ -z "${SETUP_DIR:-}" ]; then
  found="$(find "${AZ_BATCH_TASK_WORKING_DIR:-$PWD}" -name docker-compose.azure.yml -print -quit 2>/dev/null || true)"
  [ -n "$found" ] || die "docker-compose.azure.yml not found under ${AZ_BATCH_TASK_WORKING_DIR:-$PWD} (StartTask resourceFiles missing?)"
  SETUP_DIR="$(dirname "$found")"
fi
log "SETUP_DIR=$SETUP_DIR"

cp -f "$SETUP_DIR/docker-compose.azure.yml" "$SIGMAI_HOME/docker-compose.yml"
cp -f "$SETUP_DIR/bridge.sh" "$SIGMAI_HOME/bridge.sh"
chmod 755 "$SIGMAI_HOME/bridge.sh"

# Sigma-I refuses to start unless wheels/ holds EXACTLY ONE .whl.
shopt -s nullglob
src_wheels=("$SETUP_DIR"/wheels/*.whl)
[ ${#src_wheels[@]} -eq 1 ] || die "expected exactly 1 wheel in $SETUP_DIR/wheels, found ${#src_wheels[@]}"
rm -f "$SIGMAI_HOME"/wheels/*.whl
cp -f "${src_wheels[0]}" "$SIGMAI_HOME/wheels/"
log "wheel: $(basename "${src_wheels[0]}")"

# ── 2. docker compose ────────────────────────────────────────────────────
if docker compose version >/dev/null 2>&1; then
  COMPOSE=(docker compose -p sigmai --env-file "$SIGMAI_HOME/.env.server" -f "$SIGMAI_HOME/docker-compose.yml")
elif command -v docker-compose >/dev/null 2>&1; then
  COMPOSE=(docker-compose -p sigmai --env-file "$SIGMAI_HOME/.env.server" -f "$SIGMAI_HOME/docker-compose.yml")
else
  log "docker compose not found — trying to install the plugin (needs outbound internet)"
  (apt-get update -y && apt-get install -y docker-compose-plugin) >/dev/null 2>&1 || true
  docker compose version >/dev/null 2>&1 || die "docker compose is not available on this node image"
  COMPOSE=(docker compose -p sigmai --env-file "$SIGMAI_HOME/.env.server" -f "$SIGMAI_HOME/docker-compose.yml")
fi

# ── 3. API key (generated once per node, stable across reboots) ──────────
umask 077
if [ ! -f "$SIGMAI_HOME/.env.server" ] || ! grep -q '^API_KEY=' "$SIGMAI_HOME/.env.server"; then
  API_KEY="$(head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n')"
else
  API_KEY="$(grep '^API_KEY=' "$SIGMAI_HOME/.env.server" | head -1 | cut -d= -f2-)"
fi

# ── 4. D-Wave token from Blob (only when option ② is enabled) ───────────
DWAVE_API_TOKEN=""
if [ "$DWAVE_ENABLED" = "true" ]; then
  [ -n "${STORAGE_ACCOUNT:-}" ] || die "DWAVE_ENABLED=true but STORAGE_ACCOUNT is not set"
  log "fetching D-Wave token from $STORAGE_ACCOUNT/$SECRETS_CONTAINER/$DWAVE_SECRET_BLOB"
  st_token="$(imds_token https://storage.azure.com/)" || die "could not get a storage token from IMDS (is the MI attached to the pool?)"
  DWAVE_API_TOKEN="$(curl -sf -H "Authorization: Bearer $st_token" -H "x-ms-version: 2022-11-02" \
      "https://${STORAGE_ACCOUNT}.blob.core.windows.net/${SECRETS_CONTAINER}/${DWAVE_SECRET_BLOB}" \
      | tr -d '\r\n ')" || die "could not read the D-Wave token blob (role 'Storage Blob Data Reader' on the secrets container?)"
  [ -n "$DWAVE_API_TOKEN" ] || die "D-Wave token blob is empty"
else
  log "D-Wave disabled — Stage 1 can only use SCIP on this node"
fi

# ── 5. env files (chmod 600) ─────────────────────────────────────────────
cat > "$SIGMAI_HOME/.env.server" <<EOF
APP_IMAGE=${SIGMAI_IMAGE:?SIGMAI_IMAGE is required}
REDIS_IMAGE=${REDIS_IMAGE:-redis:7-alpine}
API_KEY=${API_KEY}
DWAVE_API_TOKEN=${DWAVE_API_TOKEN}
SIGMAI_HOME=${SIGMAI_HOME}
SIGMAI_PORT=${SIGMAI_PORT}
EOF
cat > "$SIGMAI_HOME/.env.client" <<EOF
API_BASE_URL=http://127.0.0.1:${SIGMAI_PORT}
API_KEY=${API_KEY}
EOF
chmod 600 "$SIGMAI_HOME/.env.server" "$SIGMAI_HOME/.env.client"

# ── 6. Pull images (ACR via Managed Identity) ────────────────────────────
if [ "$SKIP_ACR_LOGIN" != "true" ] && [ -n "${ACR_LOGIN_SERVER:-}" ]; then
  log "docker login $ACR_LOGIN_SERVER (Managed Identity)"
  arm_token="$(imds_token https://management.azure.com/)" || die "could not get an ARM token from IMDS"
  acr_refresh="$(curl -sf -X POST "https://${ACR_LOGIN_SERVER}/oauth2/exchange" \
      -d "grant_type=access_token" -d "service=${ACR_LOGIN_SERVER}" --data-urlencode "access_token=${arm_token}" \
      | "$PY" -c 'import sys,json;print(json.load(sys.stdin)["refresh_token"])')" \
      || die "ACR token exchange failed (AcrPull role for the pool MI? ACR reachable from this node?)"
  echo "$acr_refresh" | docker login "$ACR_LOGIN_SERVER" -u 00000000-0000-0000-0000-000000000000 --password-stdin >/dev/null \
      || die "docker login to ACR failed"
  "${COMPOSE[@]}" pull || die "docker compose pull failed"
fi

# ── 7. Start (idempotent) ────────────────────────────────────────────────
log "docker compose up -d"
"${COMPOSE[@]}" up -d || die "docker compose up failed"

# ── 8. Wait until the API answers AND the worker has finished its wheel install ──
log "waiting for Sigma-I to become ready (max ${READY_TIMEOUT_SEC}s; first start installs the wheel)"
deadline=$(( $(date +%s) + READY_TIMEOUT_SEC ))
api_ok=false; worker_ok=false
while [ "$(date +%s)" -lt "$deadline" ]; do
  curl -sf "http://127.0.0.1:${SIGMAI_PORT}/health" >/dev/null 2>&1 && api_ok=true || api_ok=false
  "${COMPOSE[@]}" logs worker 2>&1 | grep -qE 'celery@.* ready' && worker_ok=true || worker_ok=false
  if $api_ok && $worker_ok; then
    log "READY (api + worker)"
    exit 0
  fi
  sleep 10
done
die "Sigma-I did not become ready within ${READY_TIMEOUT_SEC}s (api_ok=$api_ok worker_ok=$worker_ok)"
