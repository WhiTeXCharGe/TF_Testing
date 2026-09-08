# GanttChartEditor Online Collaboration — Azure Preparation

> **Date:** 2026-09-08 · **Branch:** `online-collab-aca`
> What to reuse from the existing Timefold Azure setup, what to create new, and the decisions to make before running `infra/deploy.sh`.
> Assumes the company Azure account + Timefold resource group from `documents/SchedulerWeb/azure/` already exist.
> Companion: `GanttChartEditor/infra/deploy.sh` + `infra/README.md` (the raw script), `..._ACA_ImplementationPlan20260908.md` Appendix C.

---

## TL;DR

The GanttChartEditor online-collab backend needs **Blob + ACR + Container Apps** — the same three products SchedulerWeb already uses. You do **not** need a second resource group, a second registry, or a second Container Apps environment.

| Product | Reuse the SchedulerWeb one? | What GanttChartEditor adds |
|---|---|---|
| **Resource group** | ✅ reuse | — |
| **Container Registry (ACR)** | ✅ reuse | one new repo: `gantt-collab` (one image; `ROLE` env picks aca1/aca2) |
| **Container Apps environment** | ✅ reuse (`cae-…`) | — |
| **Storage account** | ✅ reuse | one new **container**: `gantt-sessions` (separate from SchedulerWeb's `input/output/status`) |
| **Container Apps app** | ➕ new | **two**: `ca-gantt-aca1` (always on) + `ca-gantt-aca2` (scale-to-zero) |
| **Web hosting** | ➕ new (or skip at first) | Azure **Static Web Apps** (Free) for the browser client — or, for first tests, run the client locally pointed at the Azure ACA1 |
| Batch, Managed Identity for pool, Key Vault | ❌ not needed | — |

Nothing new to register at the subscription level — `Microsoft.App`, `Microsoft.OperationalInsights`, `Microsoft.ContainerRegistry`, `Microsoft.Storage` are all already `Registered` from the SchedulerWeb work.

---

## The one decision to make first: who can reach it

The SchedulerWeb docs say **Storage and ACR sit behind Private Endpoints — reachable only from the company network**, not home, not VPN. That was fine there because the *user* only talked to the ACA API, and Storage/ACR were service-to-service.

For GanttChartEditor collab, the requirement is **"participants join from anywhere"**. That means:

- **ACA1 and ACA2 must have `--ingress external`** (public HTTPS). ACA ingress is public regardless of whether Storage is private — that part is fine.
- **But ACA1/ACA2 still need to reach the Storage account.** If that account is locked to a Private Endpoint, the Container Apps environment must be **VNet-integrated** with a route + private DNS to that endpoint. Ask the Azure admin:
  1. Is the existing Container Apps environment already VNet-integrated with the Timefold VNet? (If SchedulerWeb's ACA app reaches private Storage, it probably is.)
  2. Can the `gantt-sessions` container live in that same private-endpoint Storage account, and will ACA1/ACA2 in the existing environment reach it?
  3. If not — is it acceptable to use a Storage account (or a firewall rule) that the Container Apps environment can reach without extra networking? (For "internal meeting use, session data only", a Storage account with **"Allow access from selected networks" including the ACA environment's outbound IPs** is a common middle ground.)

**Recommended for a first cut:** a **new, standard (public-network-enabled) Storage account** dedicated to GanttChartEditor session state — `stganttcollab…`. Session blobs are transient meeting data (schedule + edit log), not the sensitive solver I/O. This side-steps the VNet question entirely and can be tightened later. Confirm this is acceptable with whoever owns the subscription's data policy.

---

## Reuse: values to collect from the SchedulerWeb env script

If you did `documents/SchedulerWeb/azure/Azure-Company-01-Access-And-Resources.md`, you already have `~/azure-timefold-company-env.sh`. GanttChartEditor reuses:

```bash
source ~/azure-timefold-company-env.sh
echo "SUBSCRIPTION_ID=$SUBSCRIPTION_ID"
echo "RG=$RG"                 # reuse
echo "LOC=$LOC"               # reuse (japaneast) — must be the slug, not "Japan East"
echo "ACR=$ACR  ACR_LOGIN=$ACR_LOGIN"   # reuse the registry
echo "ACA_ENV=$ACA_ENV"       # reuse the Container Apps environment
```

Then add GanttChartEditor-specific names to a **new** script so the two projects don't tangle:

```bash
cat > ~/azure-ganttcollab-env.sh <<'EOF'
source ~/azure-timefold-company-env.sh   # RG, LOC, ACR, ACR_LOGIN, ACA_ENV, SUBSCRIPTION_ID

export IMAGE=gantt-collab
export TAG=$(git -C /c/Users/PC_USER/OneDrive/Desktop/work/Timefold/web/GanttChartEditor rev-parse --short HEAD)

# Storage: decide per "who can reach it" above.
#  (a) reuse the SchedulerWeb account, new container:
# export ST="$ST"                       # from the timefold env
#  (b) new dedicated public account (recommended first cut):
export ST=stganttcollab$USER            # or a name the admin gives you; 3-24 lowercase alnum, globally unique
export SESS_CONTAINER=gantt-sessions

export ACA1=ca-gantt-aca1
export ACA2=ca-gantt-aca2

export INTERNAL_KEY=$(openssl rand -hex 32)   # shared ACA1<->ACA2 secret
export WEB_ORIGIN="https://REPLACE-WITH-WEB-ORIGIN"   # Static Web Apps URL or custom domain
EOF
source ~/azure-ganttcollab-env.sh
```

---

## New resources — what gets created

| Resource | Command source | Notes |
|---|---|---|
| ACR repo `gantt-collab` | `az acr build -r $ACR -t gantt-collab:$TAG ./server` | built from `GanttChartEditor/server/Dockerfile`; **must run on the company network** (ACR is private) |
| Storage container `gantt-sessions` | `az storage container create` | in whichever account §"who can reach it" settled on |
| `ca-gantt-aca2` (relay) | `infra/deploy.sh` §ACA2 | `--ingress external`, `--min-replicas 0 --max-replicas 5`, `--affinity sticky` |
| `ca-gantt-aca1` (session API) | `infra/deploy.sh` §ACA1 | `--ingress external`, `--min-replicas 1 --max-replicas 2` |
| Static Web Apps (web client) | `az staticwebapp create` | Free tier; or defer — see §"Web client" |

`infra/deploy.sh` as written *also* creates the RG, ACR, Storage account, Key Vault and ACA environment. For the company setup, **skip those create steps** (they exist) and set the script's variables to the existing names. The two `az containerapp create` blocks and the `az acr build` are the parts you actually run.

---

## Secrets: two options

### Option A — connection-string secret (what `infra/deploy.sh` does today)

`ca-gantt-aca1` / `ca-gantt-aca2` get `--secrets internal-key=<...> blob-conn=<storage connection string>` and reference them as `INTERNAL_KEY=secretref:internal-key`, `BLOB_CONNECTION_STRING=secretref:blob-conn`. Works immediately, no code change. The connection string is stored encrypted in the Container App. Rotate by updating the secret + `BLOB_CONNECTION_STRING`.

### Option B — Managed Identity, no secrets (matches the SchedulerWeb pattern) — needs a small code change

SchedulerWeb uses **system-assigned Managed Identity + `Storage Blob Data Contributor`** and stores no connection strings. To do the same here, `server/src/collab/storage/blobStorage.ts` must switch from `fromConnectionString` to `DefaultAzureCredential`:

```ts
// add dependency: npm i @azure/identity  (inside GanttChartEditor/server)
import { BlobServiceClient } from '@azure/storage-blob';
import { DefaultAzureCredential } from '@azure/identity';

// createBlobStorage(accountUrl: string, containerName: string) — accountUrl = https://<acct>.blob.core.windows.net
const service = new BlobServiceClient(accountUrl, new DefaultAzureCredential());
```

and `config.ts` / `makeStorage` pass `BLOB_ACCOUNT_URL` instead of `BLOB_CONNECTION_STRING`. Then in Azure:
- turn **System assigned identity = On** on both `ca-gantt-aca1` and `ca-gantt-aca2`,
- assign each one **`Storage Blob Data Contributor`** on the Storage account (portal — see SchedulerWeb `Azure-Company-02-RBAC.md`).

**Recommendation:** ship with **Option A** to get online fast; move to **Option B** as a follow-up (it's ~15 lines + 2 role assignments, and it's the company's established no-secrets pattern).

`INTERNAL_KEY` stays a secret either way (it's the ACA1↔ACA2 shared key, not an Azure credential). Once stable, move it to Key Vault with a `secretref` if the team wants — noted in `infra/README.md`.

---

## RBAC needed (portal — CLI role assignment was unreliable on this subscription)

| Identity | Role | Scope | Already have it? |
|---|---|---|---|
| **Your account** | `AcrPush` | the ACR | ✅ if you did SchedulerWeb Phase 2.1 |
| **Your account** | `Container Apps Contributor` | the two new apps (or the RG) | likely, from SchedulerWeb |
| **Your account** | `Storage Blob Data Contributor` | the session Storage account | needed if it's a **new** account; already have it for the shared one |
| `ca-gantt-aca1` system MI | `AcrPull` | the ACR | new — only if using MI to pull the image (or use `--registry-server` with admin creds / the env's own pull identity) |
| `ca-gantt-aca2` system MI | `AcrPull` | the ACR | new — same |
| `ca-gantt-aca1` / `aca2` system MI | `Storage Blob Data Contributor` | session Storage account | **only for Secrets Option B** |

If the existing Container Apps environment already has a registry pull identity configured, `az containerapp create --registry-server $ACR_LOGIN` may just work without per-app `AcrPull` — ask the admin how SchedulerWeb's `ca-tf-api` authenticates to ACR and mirror it.

---

## Web client

Two ways to give participants a page:

**Now (fastest) — run the client locally, pointed at Azure ACA1.** Exactly like SchedulerWeb Phase 7:
```bash
cd GanttChartEditor
echo "VITE_ACA1_URL=https://<ca-gantt-aca1 FQDN>" > .env.local
npm run dev    # serves on :5173; use `host: true` (already set) so LAN machines can reach it
```
Good for validating the deployed backend before committing to hosting.

**Proper — Azure Static Web Apps (Free).**
```bash
cd GanttChartEditor
VITE_ACA1_URL=https://<ca-gantt-aca1 FQDN> npx vite build     # bakes the URL in
az staticwebapp create -n swa-gantt-collab -g $RG -l $LOC --sku Free
# then deploy ./dist  (swa CLI or the portal's "upload" / GitHub Action)
```
Take its URL, set `WEB_ORIGIN` to it, and re-run the two `az containerapp update --set-env-vars WEB_ORIGIN=<url>` lines so CORS on ACA1/ACA2 is pinned to it. Add a custom domain + managed cert later (`az containerapp hostname add/bind`, and the SWA's own custom-domain blade).

> When `STORAGE=blob`, `config.ts` **requires** `WEB_ORIGIN` and an `INTERNAL_KEY` of ≥ 16 chars — the deploy will refuse to start without them. That's intentional (pinned CORS in production).

---

## Step-by-step (once the decisions above are settled)

All of this runs **on the company network** (ACR + possibly Storage are private).

```bash
source ~/azure-ganttcollab-env.sh
az account set --subscription "$SUBSCRIPTION_ID"

# 1. session-state container (in the account §"who can reach it" chose)
ST_CONN=$(az storage account show-connection-string -n "$ST" -g "$RG" --query connectionString -o tsv)
az storage container create -n "$SESS_CONTAINER" --connection-string "$ST_CONN"

# 2. build + push the one image (from GanttChartEditor/)
cd /c/Users/PC_USER/OneDrive/Desktop/work/Timefold/web/GanttChartEditor
az acr build -r "$ACR" -t "${IMAGE}:${TAG}" ./server
IMG="${ACR_LOGIN}/${IMAGE}:${TAG}"

COMMON=( STORAGE=blob "BLOB_CONTAINER=$SESS_CONTAINER" "WEB_ORIGIN=$WEB_ORIGIN"
         INTERNAL_KEY=secretref:internal-key BLOB_CONNECTION_STRING=secretref:blob-conn )
SECRETS=( "internal-key=$INTERNAL_KEY" "blob-conn=$ST_CONN" )

# 3. ACA2 (relay) — public, scale-to-zero, sticky
az containerapp create -n "$ACA2" -g "$RG" --environment "$ACA_ENV" \
  --image "$IMG" --registry-server "$ACR_LOGIN" \
  --target-port 4010 --ingress external --transport auto \
  --min-replicas 0 --max-replicas 5 --cpu 0.5 --memory 1.0Gi \
  --secrets "${SECRETS[@]}" --env-vars ROLE=aca2 PORT=4010 "${COMMON[@]}"
az containerapp ingress sticky-sessions set -n "$ACA2" -g "$RG" --affinity sticky
ACA2_FQDN=$(az containerapp show -n "$ACA2" -g "$RG" --query properties.configuration.ingress.fqdn -o tsv)
az containerapp update -n "$ACA2" -g "$RG" --set-env-vars "PUBLIC_RELAY_URL=https://${ACA2_FQDN}"

# 4. ACA1 (session API) — public, always on
az containerapp create -n "$ACA1" -g "$RG" --environment "$ACA_ENV" \
  --image "$IMG" --registry-server "$ACR_LOGIN" \
  --target-port 4000 --ingress external \
  --min-replicas 1 --max-replicas 2 --cpu 0.5 --memory 1.0Gi \
  --secrets "${SECRETS[@]}" --env-vars ROLE=aca1 PORT=4000 "ACA2_URL=https://${ACA2_FQDN}" "${COMMON[@]}"
ACA1_FQDN=$(az containerapp show -n "$ACA1" -g "$RG" --query properties.configuration.ingress.fqdn -o tsv)

# 5. smoke test
curl -s "https://${ACA1_FQDN}/api/health"    # {"ok":true,"role":"aca1",...}
curl -s "https://${ACA2_FQDN}/api/health"    # {"ok":true,"role":"aca2",...}
curl -s -X POST "https://${ACA1_FQDN}/api/sessions" -H 'content-type: application/json' \
  -d '{"name":"smoke","schedule":{"planRange":{"startDate":"2026-01-01","endDate":"2026-01-31"},"workflowTaskList":[],"assignmentList":[]},"envConfig":{"workerList":[]},"currentView":"worker"}'
```

Then §"Web client" for the front end, and re-run the `WEB_ORIGIN` update once its URL exists.

Redeploys later: `az acr build … -t gantt-collab:<newtag>` then `az containerapp update -n ca-gantt-aca2/aca1 --image …`. `.github/workflows/deploy-collab.yml` automates this once the resources exist and `AZURE_CREDENTIALS` (a service principal) is set as a repo secret.

---

## What to send whoever owns the subscription

> **GanttChartEditor online collaboration — Azure resources needed**
>
> Reuses the existing Timefold resource group, Container Registry, and Container Apps environment. New resources, all in the same resource group / region (japaneast):
>
> 1. **Storage container** `gantt-sessions` — either in the existing Timefold storage account, or a new standard Storage account `stganttcollab…` (session data is transient meeting state — schedule + edit log — not solver I/O). **Please advise which is acceptable and whether the Container Apps environment can reach it.**
> 2. **Two Container Apps** in the existing environment: `ca-gantt-aca1` (public ingress, always on, ~1 small replica) and `ca-gantt-aca2` (public ingress, scale-to-zero). Both run the same new image `gantt-collab` in the existing registry.
> 3. **Azure Static Web Apps** (Free tier) `swa-gantt-collab` for the browser client — or we can defer this and run the client locally at first.
>
> RBAC: I have `AcrPush` from the SchedulerWeb work. I need `Storage Blob Data Contributor` on the session storage account, and `Container Apps Contributor` on the two new apps (or the resource group). If we go the no-secrets route, each Container App's system-assigned identity also needs `AcrPull` on the registry and `Storage Blob Data Contributor` on the storage account.
>
> Nothing new to register at subscription level. No Batch, no Key Vault (an `INTERNAL_KEY` secret lives in the Container App; can move to Key Vault later). Estimated cost: **+$5–20/month** (one always-on ACA1 replica; ACA2 ~$0 idle; blob pennies; Static Web Apps free) on top of the existing ~$5/mo ACR.

---

## Cost

| Item | Monthly |
|---|---|
| ACR (already paid for SchedulerWeb) | $0 incremental |
| `ca-gantt-aca1` — 1 replica @ 0.5 vCPU / 1 GiB, mostly within the ACA free grant | ~$0–15 |
| `ca-gantt-aca2` — scale-to-zero between meetings | ~$0 idle, ~$0.01/min active |
| Storage container — session blobs (schedule + log, a few MB each) | pennies |
| Static Web Apps (Free tier) | $0 |
| **Total incremental** | **~$5–20/month** |

Scale-out past a handful of concurrent meetings later needs a Socket.IO Redis adapter for `ca-gantt-aca2` (Azure Managed Redis, entry tier ~$40–60/mo) — see `..._OnlineCollabAzure_Design20260904.md` §6.1. Not needed for internal meeting use.

---

## Checklist

- [ ] `source ~/azure-timefold-company-env.sh` still resolves `RG`, `LOC`, `ACR`, `ACA_ENV`
- [ ] Decision made: session Storage = existing account (new container) **or** new `stganttcollab…` account
- [ ] Admin confirmed the Container Apps environment can reach that Storage account
- [ ] Decision made: Secrets Option A (connection string) now, or Option B (Managed Identity + `@azure/identity` code change)
- [ ] `WEB_ORIGIN` decided (Static Web Apps URL, or deferred)
- [ ] `~/azure-ganttcollab-env.sh` created and sourced
- [ ] On the company network (for `az acr build` and, if private, Storage)
- [ ] `az acr build` pushed `gantt-collab:<tag>`
- [ ] `gantt-sessions` container created
- [ ] `ca-gantt-aca2` created (external, scale-to-zero, sticky) + `PUBLIC_RELAY_URL` set
- [ ] `ca-gantt-aca1` created (external, always on) with `ACA2_URL`
- [ ] `curl https://<aca1>/api/health` and `<aca2>/api/health` both return `{"ok":true}`
- [ ] `POST /api/sessions` smoke test returns a `sessionId` + `ownerToken`
- [ ] Web client built with `VITE_ACA1_URL` (or run locally with `.env.local`)
- [ ] `WEB_ORIGIN` updated on both apps once the web URL exists
- [ ] Full click-through: create → join from a second device off the office network → co-edit → lock/unlock
