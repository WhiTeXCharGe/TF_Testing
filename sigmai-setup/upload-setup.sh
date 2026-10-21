#!/usr/bin/env bash
# Upload the Sigma-I node setup files to Blob Storage so the pool StartTask can fetch them.
#   usage:  bash upload-setup.sh <version>        e.g.  bash upload-setup.sh 0.1.0-t1
#   needs:  ST (storage account), CONTAINER (target container)  — from ~/azure-timefold-company-env.sh
#           az login, and "Storage Blob Data Contributor" on the storage account
# Result: <CONTAINER>/sigmai-setup/<version>/{docker-compose.azure.yml,start-sigmai.sh,bridge.sh,wheels/<one>.whl}
set -euo pipefail
VERSION="${1:?usage: upload-setup.sh <version>}"
: "${ST:?ST is not set (source ~/azure-timefold-company-env.sh)}"
: "${CONTAINER:?CONTAINER is not set — set it to the container that will hold sigmai-setup/}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

shopt -s nullglob
wheels=("$HERE"/wheels/*.whl)
[ ${#wheels[@]} -eq 1 ] || { echo "ERROR: put exactly ONE screenas_workerassign_opt-*.whl in $HERE/wheels (found ${#wheels[@]})"; exit 1; }

up() {  # local-file  blob-name
  echo "  → $2"
  az storage blob upload --account-name "$ST" --container-name "$CONTAINER" --auth-mode login \
    --file "$1" --name "$2" --overwrite --only-show-errors >/dev/null
}
echo "Uploading to $ST/$CONTAINER/sigmai-setup/$VERSION/"
up "$HERE/docker-compose.azure.yml" "sigmai-setup/$VERSION/docker-compose.azure.yml"
up "$HERE/start-sigmai.sh"          "sigmai-setup/$VERSION/start-sigmai.sh"
up "$HERE/bridge.sh"                "sigmai-setup/$VERSION/bridge.sh"
up "${wheels[0]}"                   "sigmai-setup/$VERSION/wheels/$(basename "${wheels[0]}")"
echo "Done. Blobs under sigmai-setup/$VERSION/:"
az storage blob list --account-name "$ST" --container-name "$CONTAINER" --auth-mode login \
  --prefix "sigmai-setup/$VERSION/" --query "[].name" -o tsv
