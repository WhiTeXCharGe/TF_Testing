# GanttChartEditor — Deploying a Separate Production (Customer) ACA1 + ACA2 + Blob

> **Date:** 2026-10-07
> You already have one working set (ACA1 + ACA2 + Blob) that developers use. This doc is how to stand up a
> **second, independent set for real customers**, and how the desktop app is pointed at one or the other.
> Companion docs: `GanttChartEditor_ACA_ContainerBuildAndPush.md` (the build/push/update mechanics, same here),
> `GanttChartEditor_OnlineCollab_AzurePrep20260908.md` (RBAC, secrets, cost).

---

## 1. The short answer

**Yes — it is the same steps, just a second copy of the resources. No code change and no new image is needed.**

The backend is one Docker image; `ROLE=aca1` / `ROLE=aca2` and a handful of env vars decide everything. Nothing in
the image knows which "environment" it belongs to, so a production set is simply:

> new Blob storage account → new `INTERNAL_KEY` → new ACA2 → new ACA1 (pointing at that ACA2 and that Blob)
> → point the customer's `config.txt` at the new ACA1.

Things that are **easy to get wrong** (details in §5 and §7):

1. ACA1 and ACA2 must share **one** Blob container, and ACA1's `ACA2_URL` must be the **production** ACA2.
2. `WEB_ORIGIN` must include **`http://localhost:3010`** — that is the origin the desktop app's window has
   (it loads from its own embedded server), so without it every request from the app is blocked by CORS.
3. Use a **new `INTERNAL_KEY`**, and do not scale ACA2 past **1 replica** yet (needs a Redis adapter first).
4. `deploy.sh` hard-codes the app names `ca-gantt-aca1` / `ca-gantt-aca2` and also tries to create the
   RG/ACR/Key Vault — so don't run it unchanged for a second set; use the explicit commands in §5.

---

## 2. What is separate vs shared

| Resource | Production set | Why |
|---|---|---|
| **Blob storage account + container** | ➕ **new, separate** | Customer sessions must never mix with dev/test sessions. Separate account = separate keys, easy to wipe dev without touching prod. |
| **ACA1** (session API) | ➕ **new app** | Its own URL → this is the `azure_url` in the customer's `config.txt`. |
| **ACA2** (live relay) | ➕ **new app** | ACA1 calls it by URL; prod ACA1 must call prod ACA2. |
| **`INTERNAL_KEY`** | ➕ **new secret** | The ACA1↔ACA2 shared key. Never reuse the dev one. |
| **Container Apps environment** | ➕ new (recommended) | With a separate environment/RG the fixed app names in `deploy.sh` don't collide, and prod can have its own scaling/network settings. Sharing the existing environment also works if you rename the apps (§5, `ACA1_APP`/`ACA2_APP`). |
| **Resource group** | ➕ new (recommended) | Clean cost reporting, clean permissions, easy to delete or lock. |
| **Container Registry (ACR) + image** | ♻️ **reuse** | Same image for both. Dev and prod just run different **tags** (§8). |
| **Desktop installer** | ♻️ **reuse** | Same installer for both; only `config.txt` differs (§6). |

### Decide first (these change the steps)

| Question | If yes / no |
|---|---|
| **Is "customer" inside Timefold's own Azure subscription, or the customer's own subscription?** | Same subscription → this doc as written. Customer's own subscription → they need their own RG + registry (or access to yours), and you need push/deploy rights there; everything else is identical. |
| **Can ACA1/ACA2 reach the new Storage account?** | Recommended first cut is a **new standard Storage account with public network access** (session data is transient meeting data). If the company policy forces Private Endpoints, the Container Apps environment must be VNet-integrated with a route to it — see `AzurePrep20260908.md` §"who can reach it". |
| **Do the customers' PCs reach the internet freely?** | The app connects out over HTTPS (ACA1) and WebSocket/HTTPS (ACA2). If their firewall is strict, give them both production hostnames to allow. |

---

## 3. Names (suggested)

Use a clear suffix so nothing is ever confused with the dev set:

| Item | Dev (existing) | Production (new) |
|---|---|---|
| Resource group | `rg-gantt-collab` | `rg-gantt-collab-prod` |
| Container Apps environment | `cae-gantt` | `cae-gantt-prod` |
| Storage account (3–24 lowercase alnum, globally unique) | `stganttcollab…` | `stganttprod<suffix>` |
| Blob container | `sessions` | `sessions` |
| ACA1 | `ca-gantt-aca1` | `ca-gantt-aca1` *(different RG/env, so no clash)* |
| ACA2 | `ca-gantt-aca2` | `ca-gantt-aca2` |

If you put production in the **same** environment as dev, give the apps different names
(`ca-gantt-aca1-prod`, `ca-gantt-aca2-prod`).

---

## 4. Prerequisites

Same as `GanttChartEditor_ACA_ContainerBuildAndPush.md` §4: `az login`, the right subscription, and the
company network for anything that touches ACR. You need **Contributor** (or Container Apps Contributor +
Storage permissions) on the new resource group, and pull access to the existing registry.

Pick the image tag to run in production — **a tag that has already been running on the dev set**, not "whatever was
just pushed":

```bash
az acr repository show-tags -n <ACR> --repository gantt-collab -o table
```

---

## 5. Create the production set

Run from any shell with `az`. Nothing here needs the source code checked out — the image already exists in ACR.

```bash
# ---- settings ---------------------------------------------------------------
LOC=japaneast
RG=rg-gantt-collab-prod
ENVIRONMENT=cae-gantt-prod
ACR=<existing registry name, without .azurecr.io>      # reused
TAG=<tag already proven on dev>                        # e.g. 3aaee1d
STG=stganttprod<suffix>                                # globally unique
CONTAINER=sessions
ACA1_APP=ca-gantt-aca1                                 # rename if sharing an environment with dev
ACA2_APP=ca-gantt-aca2
WEB_ORIGIN="http://localhost:3010"                     # see §7 — required for the desktop app
CPU=0.5
MEMORY=1.0Gi
IMG="${ACR}.azurecr.io/gantt-collab:${TAG}"

# ---- 1. resource group ------------------------------------------------------
az group create -n "$RG" -l "$LOC" -o none

# ---- 2. Blob storage (production data lives ONLY here) ----------------------
az storage account create -n "$STG" -g "$RG" -l "$LOC" \
  --sku Standard_LRS --kind StorageV2 --min-tls-version TLS1_2 \
  --allow-blob-public-access false -o none
STG_CONN="$(az storage account show-connection-string -n "$STG" -g "$RG" --query connectionString -o tsv)"
az storage container create -n "$CONTAINER" --connection-string "$STG_CONN" -o none

# ---- 3. a NEW shared secret for ACA1 <-> ACA2 (>= 16 chars; never reuse dev's) -
INTERNAL_KEY="$(openssl rand -hex 32)"
# Keep it somewhere safe (Key Vault / password manager) — you need it to recreate or rotate.

# ---- 4. Container Apps environment ------------------------------------------
az containerapp env create -n "$ENVIRONMENT" -g "$RG" -l "$LOC" -o none

COMMON_ENV=(
  "STORAGE=blob"
  "BLOB_CONTAINER=$CONTAINER"
  "WEB_ORIGIN=$WEB_ORIGIN"
  "INTERNAL_KEY=secretref:internal-key"
  "BLOB_CONNECTION_STRING=secretref:blob-conn"
)
SECRETS=( "internal-key=$INTERNAL_KEY" "blob-conn=$STG_CONN" )

# ---- 5. ACA2 — live relay (create first: ACA1 needs its URL) ----------------
az containerapp create -n "$ACA2_APP" -g "$RG" \
  --environment "$ENVIRONMENT" --image "$IMG" --registry-server "${ACR}.azurecr.io" \
  --target-port 4010 --ingress external --transport auto \
  --min-replicas 0 --max-replicas 1 --cpu "$CPU" --memory "$MEMORY" \
  --secrets "${SECRETS[@]}" \
  --env-vars ROLE=aca2 PORT=4010 "${COMMON_ENV[@]}" -o none
ACA2_FQDN="$(az containerapp show -n "$ACA2_APP" -g "$RG" --query properties.configuration.ingress.fqdn -o tsv)"
# ACA2 hands this URL to clients so they know where to open their socket:
az containerapp update -n "$ACA2_APP" -g "$RG" --set-env-vars "PUBLIC_RELAY_URL=https://${ACA2_FQDN}" -o none

# ---- 6. ACA1 — session API (the URL customers put in config.txt) ------------
az containerapp create -n "$ACA1_APP" -g "$RG" \
  --environment "$ENVIRONMENT" --image "$IMG" --registry-server "${ACR}.azurecr.io" \
  --target-port 4000 --ingress external \
  --min-replicas 1 --max-replicas 2 --cpu "$CPU" --memory "$MEMORY" \
  --secrets "${SECRETS[@]}" \
  --env-vars ROLE=aca1 PORT=4000 "ACA2_URL=https://${ACA2_FQDN}" "${COMMON_ENV[@]}" -o none
ACA1_FQDN="$(az containerapp show -n "$ACA1_APP" -g "$RG" --query properties.configuration.ingress.fqdn -o tsv)"

echo "ACA1 (give this to config.txt azure_url): https://${ACA1_FQDN}"
echo "ACA2 (relay):                              https://${ACA2_FQDN}"
```

Notes on the choices above (they differ slightly from `deploy.sh` on purpose):

- **ACA2 `--max-replicas 1`** (dev's script uses 5). More than one relay replica needs the Socket.IO **Redis
  adapter** plus a shared session store (see Design doc §6.1) — neither exists yet, so two replicas could put
  two people in the *same* session on *different* relays and they would stop seeing each other. One replica is
  the safe production setting until that work is done. (`--affinity sticky` only matters with >1 replica.)
- **ACA2 `--min-replicas 0`** keeps cost near zero; the first person to open a session after idle waits for a
  cold start (the app already allows up to 60 s for this). If customers will find that annoying, use
  `--min-replicas 1` (a small always-on cost).
- **ACA1 `--min-replicas 1`** means listing/creating sessions never cold-starts.
- **Registry pull:** `--registry-server` works when ACR has the admin user enabled (as in `deploy.sh`). If the
  registry uses Managed Identity instead, grant the new apps `AcrPull` and use `--registry-identity system`.

### Optional env vars worth setting for production

Add to `COMMON_ENV` or `--set-env-vars` later (defaults in `server/src/config.ts`):

| Variable | Default | Meaning |
|---|---|---|
| `CREATE_RATE_PER_HOUR` | 10 | Session creations per IP per hour |
| `MAX_SESSIONS` | 100 | Cap on live sessions |
| `MAX_PARTICIPANTS` | 25 | Per-session connection cap |
| `MAX_UPLOAD_MB` | 5 | YAML upload cap |
| `IDLE_SESSION_TIMEOUT_MS` | 1 800 000 (30 min) | How long an empty session stays open before it is closed |

---

## 6. Point the desktop app at it (no rebuild, no new installer)

Each installed copy reads `config.txt` next to `GanttChartEditor.exe`:

```
# customer / production
mode=online
azure_url=https://<production ACA1 FQDN>
```

```
# developer
mode=online
azure_url=https://<dev ACA1 FQDN>
```

or `mode=local` for the bundled local server. Restart the app (or open a new window) after editing.
In `mode=online` the app never falls back to local: if the server can't be reached it shows
「サーバーに接続できません…」 and session creation is disabled.

**Which `config.txt` ships in the installer** is whatever is in the repo root `config.txt` at build time
(electron-builder copies it next to the exe). Before building the **customer** installer, set that file's
`azure_url` to the production ACA1 — or build one installer and have the customer's `config.txt` replaced
after install. Note that re-running the installer may overwrite an edited `config.txt`.

---

## 7. `WEB_ORIGIN` — the one that bites

`WEB_ORIGIN` is the CORS allow-list on ACA1 **and** ACA2, and with `STORAGE=blob` the server refuses to start
without it. It must be the **exact origin the browser/app sends**:

| Client | Origin it sends | Put in `WEB_ORIGIN` |
|---|---|---|
| Packaged desktop app | `http://localhost:3010` (the window loads from the app's own embedded server) | `http://localhost:3010` |
| `npm run dev` web client | `http://localhost:5173` | `http://localhost:5173` — **dev only**, leave it out of production |
| A hosted web client (Static Web Apps, custom domain) | its own `https://…` URL | add it |

More than one is comma-separated, no spaces: `WEB_ORIGIN=http://localhost:3010,https://gantt.example.com`.
Change it later without rebuilding:

```bash
az containerapp update -n "$ACA1_APP" -g "$RG" --set-env-vars "WEB_ORIGIN=http://localhost:3010,https://…"
az containerapp update -n "$ACA2_APP" -g "$RG" --set-env-vars "WEB_ORIGIN=http://localhost:3010,https://…"
```

If a customer's app says it can't connect but `curl https://<aca1>/api/health` works, a wrong `WEB_ORIGIN` is
the first thing to check.

---

## 8. Releasing: dev first, then production

One image, two environments — **never** promote by rebuilding; promote by tag:

```bash
# 1. build + push once (see ContainerBuildAndPush.md §5), update DEV, test it
# 2. when happy, point production at the exact same tag:
IMG="${ACR}.azurecr.io/gantt-collab:<tested-tag>"
az containerapp update -n "$ACA2_APP" -g "$RG" --image "$IMG"
az containerapp update -n "$ACA1_APP" -g "$RG" --image "$IMG"
```

Rollback is the same command with the previous tag. Updating ACA2 drops its open sockets (clients reconnect
automatically), so do production updates outside meeting hours.

An env-var-only change (rate limits, `WEB_ORIGIN`) needs no image and no rebuild — `az containerapp update
--set-env-vars …` on both apps.

---

## 9. Verify

```bash
curl -s "https://${ACA1_FQDN}/api/health"    # {"ok":true,"role":"aca1",...}
curl -s "https://${ACA2_FQDN}/api/health"    # {"ok":true,"role":"aca2",...}
curl -s "https://${ACA1_FQDN}/api/sessions"  # {"ok":true,"sessions":[]}  <- empty on a fresh set

# CORS as the desktop app would send it:
curl -s -i "https://${ACA1_FQDN}/api/sessions" -H "Origin: http://localhost:3010" | grep -i access-control-allow-origin
```

Then the real test: install/launch the app with `config.txt` pointing at production, create a session, join it
from a second PC, edit together, lock/unlock. Confirm the blob appeared in the **production** storage account
(and nothing in the dev one).

---

## 10. Things to decide before real customers use it

- **There is no authentication.** Anyone who has the ACA1 URL can list and delete sessions, and anyone with a
  session id can join it (Design doc, "No authentication at all"). For internal meetings that was acceptable; for
  customers decide whether it still is. Cheap mitigations: restrict ingress by IP
  (`az containerapp ingress access-restriction set`) — but ACA1 calls ACA2 over its public URL, so ACA2's
  rule must also allow the environment's outbound address; or add real auth later.
- **Data retention / backups.** Session data is transient (a "close" session is swept after its timeout), but
  the blobs persist until the sweep deletes them. Decide if the customer needs soft-delete / lifecycle rules on
  the storage account.
- **Secrets.** `INTERNAL_KEY` and the storage connection string live as Container App secrets. Keep a copy in a
  vault; to rotate, update the secret on **both** apps. Switching to Managed Identity for Blob (Option B in
  `AzurePrep20260908.md`) removes the connection string entirely but needs a small code change.
- **Monitoring.** Turn on the environment's Log Analytics workspace and set an alert on ACA1/ACA2 restarts or
  5xx — otherwise "the app can't connect" is the first signal.
- **Cost.** A second set roughly doubles the incremental cost in `AzurePrep20260908.md` (~$5–20/month: one
  always-on ACA1, ACA2 idle at ~$0, blob in pennies).

---

## 11. Older docs now out of date

`AzurePrep20260908.md`, `ContainerBuildAndPush.md` and `deploy.sh`'s closing notes still tell you to build the
client with `VITE_ACA1_URL=https://<aca1>`. That mechanism **no longer exists** — the URL now comes from
`config.txt` at runtime (see §6). Everything about creating the Azure resources in those docs is still right.

---

## Checklist

- [ ] Decided: same Timefold subscription or the customer's own
- [ ] Decided: Storage reachability (public new account, or Private Endpoint + VNet integration)
- [ ] Image tag to run in production chosen (one already proven on dev)
- [ ] New resource group, Storage account + `sessions` container created
- [ ] New `INTERNAL_KEY` generated and stored safely
- [ ] New Container Apps environment created
- [ ] ACA2 created (`--max-replicas 1`), `PUBLIC_RELAY_URL` set to its own FQDN
- [ ] ACA1 created, `ACA2_URL` = production ACA2, **same** Blob container and `INTERNAL_KEY`
- [ ] `WEB_ORIGIN` includes `http://localhost:3010` on **both** apps
- [ ] `/api/health` OK on both; CORS header returned for `Origin: http://localhost:3010`
- [ ] Customer installer's `config.txt` has `mode=online` + the **production** ACA1 URL
- [ ] End-to-end test from two PCs; blob appears in the production storage account only
- [ ] Authentication / IP-restriction / retention decisions made (§10)
