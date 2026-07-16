# Company Phase 1 — Sign in and find every resource

**Goal of this phase:** get logged into the company's Azure account, find the
resource group the company set up for this project, and identify every
resource inside it. By the end you'll have a personal script with the real
names of everything, so every later phase can just say "use `$ST`" instead
of you re-typing (or re-guessing) a name.

**Time:** ~20 minutes.
**You do NOT create anything in this phase** — the company already created
the resources. You're just finding them and writing their names down.

**Written for someone following this without deep Azure knowledge** — every
step says exactly where to click or what to type. If a step doesn't match
what you see on screen, stop and check the Troubleshooting section at the
bottom before improvising.

---

## Why this phase exists

On your personal account (Phase 01-07, if you did those) you picked every
name yourself, so the docs could just say `sttimefolddevseiya`. On the
company account, an admin created the resources and picked their own names —
you don't know them yet. This phase is entirely about **discovery**: find
each resource, write its real name into one script, and every phase after
this just uses that script.

**Never guess a name and act on it.** If you're not 100% sure a resource is
the right one, ask whoever set it up rather than proceeding on a guess —
wrong names fail loudly (a typo won't destroy anything), but you'll waste
time chasing the wrong resource.

---

## Step 1 — Sign in to the Azure Portal

1. Go to https://portal.azure.com
2. Sign in with the account the company gave you.
3. Top-right corner → click your account avatar → confirm you see the
   **company's** organization name, not your personal Microsoft account.
   If you're signed into both, you may need to pick the right one from a
   list, or open a new **InPrivate/Incognito** browser window to avoid mixing
   sessions.

## Step 2 — Confirm the subscription and tenant

1. Portal search bar (top) → type `subscriptions` → click the result.
2. Click into the subscription that appears (there should be exactly one
   you have access to; if you see several, ask which one is for this
   project).
3. Note the **Subscription ID** (a GUID like `12345abc-...`) — click the
   copy icon next to it.

## Step 3 — Install and sign in to Azure CLI (if not already)

Some steps in later phases are faster with the CLI. The portal always works
as a fallback if a CLI command misbehaves (see
[Azure-03-2-RoleAssignment-Workaround.md](./Azure-03-2-RoleAssignment-Workaround.md)
— role assignment specifically should be done in the **portal**, not CLI,
for the whole rest of this project).

```bash
az --version
```

If that fails, install from https://aka.ms/installazurecliwindows, then open
a **new** Git Bash window.

Sign in with the company account:

```bash
az login
```

A browser window opens — pick the company account. Then confirm:

```bash
az account show --output table
```

If you have more than one subscription available and the wrong one is
active:

```bash
az account list --output table
az account set --subscription "<subscription-id-from-step-2>"
```

## Step 4 — Find the resource group

1. Portal search → `resource groups` → click the result.
2. You should see a list. If there's exactly one, that's almost certainly
   the project's group — click into it. If there are several, look for one
   whose name suggests this project (ask the admin if it's not obvious —
   **do not guess**).
3. Once you're confident you've found it, **do not write its name into any
   shared/company document** — see the note at the top of
   [Azure-Company-Setup-Checklist.md](./Azure-Company-Setup-Checklist.md).
   You can write it into the private env script below; that file stays on
   your machine only.

Confirm from CLI once you know the name:

```bash
az group show --name "<resource-group-name>" --output table
```

Should print one row.

## Step 5 — List everything inside the resource group

Inside the resource group's **Overview** page, you'll see a table of every
resource with its **Name** and **Type**. This is your master list. Expect
to see roughly these types (names will be whatever the company chose):

| Type (what the portal will call it) | What it is in this project        |
| ------------------------------------- | ---------------------------------- |
| Storage account                       | Blob storage for input/output/status |
| Container registry                    | ACR — holds the Docker images       |
| Container Apps Environment            | Hosts the API Controller           |
| Container App                         | The API Controller itself           |
| Batch account                         | Runs the Timefold solver as tasks   |
| Managed Identity                      | Identity the Batch pool uses        |

If any type is **missing** entirely, don't panic — it might not be created
yet, or might live in a different resource group you don't have access to.
Ask the admin; don't create a duplicate yourself without checking first
(see the fallback procedure in
[Azure-Company-Setup-Checklist.md §7](./Azure-Company-Setup-Checklist.md#7-fallback--if-something-is-genuinely-missing)).

## Step 6 — Build your personal env script

This script is the single source of truth for the rest of every phase.
Create it now with what you know so far, and you'll add more lines to it
as you go through each resource below.

**Get `$LOC` from the resource group itself, not the portal.** The portal's
Overview page shows a human-readable **display name** like `Japan East`,
but every CLI command that takes `--location` wants the **slug** form
(`japaneast` — no space, lowercase). Passing the display name doesn't
always error clearly — some `--location`-based commands just silently
return nothing instead of failing loudly, which is a confusing way to lose
10 minutes. Skip the portal for this one value:

```bash
az group show --name "<resource-group-name>" --query location -o tsv
```

That always prints the correct slug — use exactly that.

```bash
cat > ~/azure-timefold-company-env.sh <<'EOF'
# Source this at the start of every terminal session for this project:
#   source ~/azure-timefold-company-env.sh
# Fill in every <placeholder> below with the REAL name you found in the portal.

export SUBSCRIPTION_ID="<subscription-id>"
export RG="<resource-group-name>"
export LOC="<region-slug-from-az-group-show>"   # e.g. japaneast — NOT "Japan East"

echo "Loaded company env: RG=$RG  LOC=$LOC"
EOF

source ~/azure-timefold-company-env.sh
```

Verify:
```bash
echo "SUBSCRIPTION_ID=$SUBSCRIPTION_ID  RG=$RG  LOC=$LOC"
```
All three should print real values, not the literal word `<placeholder>`.

## Step 7 — Record the Storage account

**If you see more than one resource of type Storage account, stop and read
this first.** Don't just pick one — Azure Batch's task definitions in this
project use `autoStorageContainerName` to find input files, which resolves
to whichever storage account is linked to the **Batch account's own
auto-storage setting** — not necessarily the one that looks most obviously
"for this project." Using the wrong one means Batch tasks will fail to find
their input files even though everything else looks correctly configured.

Find out which one is actually linked, using the Batch account you'll
record in Step 10 below (do this step, or come back to it once you've
found the Batch account if you're going top-to-bottom):

```bash
az batch account show --name "<batch-account-name>" --resource-group "$RG" --query "autoStorage.storageAccountId" -o tsv
```

The resource ID printed ends in the *real* storage account's name — **that
one**, and only that one, is what you record below. If there's only one
storage account in the resource group, you can skip this check (there's no
ambiguity), but it doesn't hurt to run it anyway to confirm.

1. In the resource group list, click the storage account confirmed above.
2. Note its **Name** (top of the Overview page).
3. Left sidebar → **Containers** → note the container name(s) listed
   (there should be one holding `input/`, `output/`, `status/` blobs — open
   it and check the folder-looking names to confirm).

Append to your env script:
```bash
cat >> ~/azure-timefold-company-env.sh <<'EOF'

# Storage
export ST="<storage-account-name>"
export CONTAINER="<blob-container-name>"
EOF
source ~/azure-timefold-company-env.sh
```

Confirm from CLI:
```bash
az storage account show --name "$ST" --resource-group "$RG" --query "{name:name, kind:kind, sku:sku.name}" -o table
```

## Step 8 — Record the Container Registry (ACR)

1. Click the resource of type **Container registry**.
2. Note its **Name**. The **Login server** field (same Overview page) is
   `<name>.azurecr.io` — you'll need the full login server later.

```bash
cat >> ~/azure-timefold-company-env.sh <<'EOF'

# Container Registry
export ACR="<acr-name>"
export ACR_LOGIN="<acr-name>.azurecr.io"
EOF
source ~/azure-timefold-company-env.sh
```

Confirm:
```bash
az acr show --name "$ACR" --resource-group "$RG" --query "{name:name, loginServer:loginServer, sku:sku.name}" -o table
```

## Step 9 — Record the Container Apps environment + app

1. Click the resource of type **Container Apps Environment** — note its name.
2. Click the resource of type **Container App** — note its name, and note
   the **Application Url** on its Overview page (the public HTTPS address).

```bash
cat >> ~/azure-timefold-company-env.sh <<'EOF'

# Container Apps
export ACA_ENV="<aca-environment-name>"
export ACA_APP="<aca-app-name>"
EOF
source ~/azure-timefold-company-env.sh

APP_URL=$(az containerapp show --name "$ACA_APP" --resource-group "$RG" --query "properties.configuration.ingress.fqdn" -o tsv)
echo "APP_URL=$APP_URL"

cat >> ~/azure-timefold-company-env.sh <<EOF

export APP_URL=$APP_URL
EOF
```

Open `https://$APP_URL` in a browser — whatever it currently shows (a
placeholder page, an error, or nothing deployed yet) is fine; you're just
confirming the URL resolves. You'll deploy the real API to it in
[Azure-Company-06-API-Controller-Deploy.md](./Azure-Company-06-API-Controller-Deploy.md).

## Step 10 — Record the Batch account + pool

1. Click the resource of type **Batch account** — note its name.
2. Left sidebar → **Pools** → note the Pool ID (this is a user-chosen ID,
   not a random name, e.g. `pool-timefold-prod` or similar).
3. Left sidebar → **Jobs** → if a job already exists, note its Job ID. If
   the Jobs list is empty, that's fine — a job may not exist yet; you (or
   this guide) will create one in
   [Azure-Company-05-Batch-Setup-And-Run.md](./Azure-Company-05-Batch-Setup-And-Run.md).

```bash
cat >> ~/azure-timefold-company-env.sh <<'EOF'

# Batch
export BATCH="<batch-account-name>"
export POOL_ID="<batch-pool-id>"
export JOB_ID="<batch-job-id-or-leave-blank-if-none-yet>"
EOF
source ~/azure-timefold-company-env.sh

BATCH_URL=$(az batch account show --name "$BATCH" --resource-group "$RG" --query accountEndpoint -o tsv)
echo "BATCH_URL=https://$BATCH_URL"

cat >> ~/azure-timefold-company-env.sh <<EOF

export BATCH_URL=https://$BATCH_URL
EOF
```

## Step 11 — Record the User-assigned Managed Identity

1. On the Batch pool's page → **Identity** blade → you should see a
   **User-assigned** identity listed. Click through to it, or find it
   directly as a resource of type **Managed Identity** in the resource
   group.
2. On its Overview page, note the **Name**, and the **Resource ID** (usually
   need to click "JSON View" or check the Properties blade — it looks like
   `/subscriptions/<id>/resourceGroups/<rg>/providers/Microsoft.ManagedIdentity/userAssignedIdentities/<name>`).
3. Also note the **Client ID** and **Object (principal) ID** — you'll need
   the principal ID for the RBAC phase next.

```bash
cat >> ~/azure-timefold-company-env.sh <<'EOF'

# Batch pool's Managed Identity
export MI_NAME="<user-assigned-identity-name>"
EOF
source ~/azure-timefold-company-env.sh

MI_ID=$(az identity show --name "$MI_NAME" --resource-group "$RG" --query id -o tsv)
MI_OID=$(az identity show --name "$MI_NAME" --resource-group "$RG" --query principalId -o tsv)
MI_CLIENT_ID=$(az identity show --name "$MI_NAME" --resource-group "$RG" --query clientId -o tsv)

cat >> ~/azure-timefold-company-env.sh <<EOF

export MI_ID=$MI_ID
export MI_OID=$MI_OID
export MI_CLIENT_ID=$MI_CLIENT_ID
EOF
source ~/azure-timefold-company-env.sh
echo "MI_ID=$MI_ID"
```

## Step 12 — Sanity check the whole script

Open a **brand new** Git Bash window (to make sure nothing is left over from
your session) and run:

```bash
source ~/azure-timefold-company-env.sh
echo "RG=$RG"
echo "ST=$ST  CONTAINER=$CONTAINER"
echo "ACR=$ACR  ACR_LOGIN=$ACR_LOGIN"
echo "ACA_ENV=$ACA_ENV  ACA_APP=$ACA_APP  APP_URL=$APP_URL"
echo "BATCH=$BATCH  BATCH_URL=$BATCH_URL  POOL_ID=$POOL_ID  JOB_ID=$JOB_ID"
echo "MI_NAME=$MI_NAME  MI_ID=$MI_ID"
```

Every line should show a real value, no empty variables, no leftover
`<placeholder>` text. If something's empty, go back to that resource's step
above.

---

## What you should have at the end of this phase

- [ ] Signed into the company Azure Portal and confirmed the right subscription
- [ ] `~/azure-timefold-company-env.sh` exists with every real name filled in
- [ ] Re-sourcing the script in a fresh terminal prints all real values
- [ ] You know which resource types exist and which (if any) are still missing

Next: [Azure-Company-02-RBAC.md](./Azure-Company-02-RBAC.md) — confirm (or
assign) the permissions each identity needs.

---

## Troubleshooting

| Symptom                                                     | Cause                                        | Fix                                                                     |
| ------------------------------------------------------------ | --------------------------------------------- | ------------------------------------------------------------------------ |
| Portal shows no resource groups at all                        | You only have access scoped to specific resources, not the group | Ask the admin for direct links to each resource, or to grant you Reader on the group |
| `az login` opens browser but signs into the wrong account     | Browser has multiple Microsoft sessions cached | Use an InPrivate/Incognito window, or sign out of other accounts first  |
| `az group show` says group not found                          | Wrong subscription active                     | Re-run `az account set --subscription "<subscription-id>"`             |
| Batch account has no Pools or Jobs at all                     | Batch side genuinely not provisioned yet      | Note this as a gap — you'll need it before Phase 5; ask the admin       |
| You can't find a Managed Identity resource anywhere           | It may be system-assigned (lives inside the ACA app, not a standalone resource) rather than user-assigned (which the Batch pool needs) | Check ACA app → Identity blade for system-assigned; the Batch pool specifically needs a user-assigned one — flag this to the admin if truly absent |
| More than one Storage account in the resource group, and you're not sure which is "the" one | The company may have created one for this project and one for something unrelated (or a Batch-default diagnostics account) | Run the `az batch account show --query autoStorage.storageAccountId` check in Step 7 — that's the authoritative answer, not a guess. Use only that one for `$ST` everywhere (Phases 2, 3, 6, 7 too) |
| A `--location`-based command returns nothing, no error | `$LOC` was set to the portal's display name (`Japan East`) instead of the slug (`japaneast`) | `az group show --name "$RG" --query location -o tsv` prints the correct slug — use exactly that, not what the portal Overview page displays |
| `az batch location quotas show --location "$LOC"` returns nothing even with the correct slug | That command checks an old subscription+region quota model most Batch accounts don't use anymore | Use `az batch account show --name "$BATCH" --resource-group "$RG" --query "{dedicatedCoreQuota:dedicatedCoreQuota, lowPriorityCoreQuota:lowPriorityCoreQuota}"` instead — see [Azure-Company-05-Batch-Setup-And-Run.md Step 2](./Azure-Company-05-Batch-Setup-And-Run.md#step-2--check-the-vcpu-quota) |
