#!/usr/bin/env bash
# Provision + deploy the online-collaboration backend to Azure Container Apps.
#
#   ACA1  session API   — always on   (minReplicas 1)
#   ACA2  live relay     — scale-to-0  (minReplicas 0), Socket.IO, external ingress
#   Blob  session state  — one container, one blob per sessions/<id>/<file>.json
#   ACR   one image, ROLE env picks aca1 vs aca2
#
# Idempotent-ish: `az ... create` calls are safe to re-run; image build + app
# update always run. Review every line before running in a real subscription.
#
# Prereqs: az CLI >= 2.53, `az login` done, `az account set -s <sub>` done,
#          the containerapp extension (auto-installed on first use),
#          run from the repo so ./server (with its Dockerfile) is the build context.
set -euo pipefail

# ---- settings (override via env) --------------------------------------------
LOC="${LOC:-japaneast}"
RG="${RG:-rg-gantt-collab}"
ENVIRONMENT="${ENVIRONMENT:-cae-gantt}"
# ACR + storage account names are GLOBALLY unique and charset-limited
# (3-24 lowercase alphanumerics). Set SUFFIX to something short + unique.
SUFFIX="${SUFFIX:?set SUFFIX to a short unique string, e.g. your initials + 3 digits}"
ACR="${ACR:-acrganttcollab${SUFFIX}}"
STG="${STG:-stganttcollab${SUFFIX}}"
CONTAINER="${CONTAINER:-sessions}"
KV="${KV:-kv-gantt-${SUFFIX}}"
IMAGE="${IMAGE:-gantt-collab}"
TAG="${TAG:-$(git rev-parse --short HEAD 2>/dev/null || date +%Y%m%d%H%M)}"
# The web origin that will call the API from a browser (Static Web Apps URL or
# your custom domain). CORS is pinned to this on ACA1/ACA2. Update after SWA.
WEB_ORIGIN="${WEB_ORIGIN:-https://REPLACE-WITH-WEB-ORIGIN}"
CPU="${CPU:-0.5}"
MEMORY="${MEMORY:-1.0Gi}"
# ---------------------------------------------------------------------------

echo ">> resource group"
az group create -n "$RG" -l "$LOC" -o none

echo ">> container registry + image build ($IMAGE:$TAG from ./server)"
az acr create -n "$ACR" -g "$RG" --sku Basic --admin-enabled true -o none
az acr build -r "$ACR" -t "${IMAGE}:${TAG}" ./server

echo ">> storage account + container"
az storage account create -n "$STG" -g "$RG" -l "$LOC" \
  --sku Standard_LRS --kind StorageV2 --min-tls-version TLS1_2 --allow-blob-public-access false -o none
STG_CONN="$(az storage account show-connection-string -n "$STG" -g "$RG" --query connectionString -o tsv)"
az storage container create -n "$CONTAINER" --connection-string "$STG_CONN" -o none

echo ">> key vault + secrets (source of truth; also passed inline below)"
INTERNAL_KEY="${INTERNAL_KEY:-$(openssl rand -hex 32)}"
az keyvault create -n "$KV" -g "$RG" -l "$LOC" -o none || true
az keyvault secret set --vault-name "$KV" -n internal-key --value "$INTERNAL_KEY" -o none
az keyvault secret set --vault-name "$KV" -n blob-conn   --value "$STG_CONN"      -o none

echo ">> container apps environment"
az containerapp env create -n "$ENVIRONMENT" -g "$RG" -l "$LOC" -o none

COMMON_ENV=(
  "STORAGE=blob"
  "BLOB_CONTAINER=$CONTAINER"
  "WEB_ORIGIN=$WEB_ORIGIN"
  "INTERNAL_KEY=secretref:internal-key"
  "BLOB_CONNECTION_STRING=secretref:blob-conn"
)
SECRETS=( "internal-key=$INTERNAL_KEY" "blob-conn=$STG_CONN" )
IMG="${ACR}.azurecr.io/${IMAGE}:${TAG}"

echo ">> ACA2 (live relay)"
az containerapp create -n ca-gantt-aca2 -g "$RG" \
  --environment "$ENVIRONMENT" --image "$IMG" --registry-server "${ACR}.azurecr.io" \
  --target-port 4010 --ingress external --transport auto \
  --min-replicas 0 --max-replicas 5 --cpu "$CPU" --memory "$MEMORY" \
  --secrets "${SECRETS[@]}" \
  --env-vars ROLE=aca2 PORT=4010 "${COMMON_ENV[@]}" -o none
# Socket.IO handshake + upgrade must land on the same replica once maxReplicas > 1.
az containerapp ingress sticky-sessions set -n ca-gantt-aca2 -g "$RG" --affinity sticky -o none
ACA2_FQDN="$(az containerapp show -n ca-gantt-aca2 -g "$RG" --query properties.configuration.ingress.fqdn -o tsv)"
az containerapp update -n ca-gantt-aca2 -g "$RG" --set-env-vars "PUBLIC_RELAY_URL=https://${ACA2_FQDN}" -o none

echo ">> ACA1 (session API)"
az containerapp create -n ca-gantt-aca1 -g "$RG" \
  --environment "$ENVIRONMENT" --image "$IMG" --registry-server "${ACR}.azurecr.io" \
  --target-port 4000 --ingress external \
  --min-replicas 1 --max-replicas 2 --cpu "$CPU" --memory "$MEMORY" \
  --secrets "${SECRETS[@]}" \
  --env-vars ROLE=aca1 PORT=4000 "ACA2_URL=https://${ACA2_FQDN}" "${COMMON_ENV[@]}" -o none
ACA1_FQDN="$(az containerapp show -n ca-gantt-aca1 -g "$RG" --query properties.configuration.ingress.fqdn -o tsv)"

cat <<EOF

done.
  ACA1 (session API) : https://${ACA1_FQDN}
  ACA2 (live relay)   : https://${ACA2_FQDN}
  image              : ${IMG}

smoke test:
  curl -s https://${ACA1_FQDN}/api/health
  curl -s https://${ACA2_FQDN}/api/health
  curl -s -X POST https://${ACA1_FQDN}/api/sessions -H 'content-type: application/json' \\
    -d '{"name":"smoke","schedule":{"a":1},"envConfig":{"b":2},"currentView":"worker"}'

next:
  - build + deploy the web client to Static Web Apps with VITE_ACA1_URL=https://${ACA1_FQDN}
  - set WEB_ORIGIN to the SWA URL and re-run the two 'az containerapp update' lines
    (see infra/README.md) so CORS is pinned to it.
EOF
