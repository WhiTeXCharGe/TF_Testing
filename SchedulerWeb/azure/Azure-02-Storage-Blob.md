# Phase 2 — Storage account + first blob container

**Goal of this phase:** a working Azure Blob Storage area with the exact
folder layout the API Controller will use (`input/`, `output/`, `status/`),
and you confident you can upload/download/share files in it from CLI.

**Time:** ~25 minutes.
**Cost:** ~$0.01 / month for what we put in it (a few YAML files).

**Prereqs:** Phase 1 done — you have `rg-timefold-dev` and `az` works.

---

## Concepts (read once, ~1 min)

Three nested things. Pay attention to the difference because they each have
their own naming rules and their own permissions.

| Concept             | What it is                                                              | Naming rules                              |
| ------------------- | ----------------------------------------------------------------------- | ----------------------------------------- |
| **Storage account** | The billing/security boundary. One account can hold blobs, files, queues, tables. | 3–24 lowercase letters+digits, **globally unique across all Azure** |
| **Container**       | A "folder" inside a storage account. Holds blobs.                       | 3–63 lowercase letters/digits/hyphens, unique within the account |
| **Blob**            | A file. Can have `/` in the name to **look like** subfolders (no real folders exist). | Up to 1024 chars, any printable |

So `input/20260527_001/EnvConfig.yaml` is one blob, with a name that *contains
slashes*. The portal renders it as a folder tree, but under the hood there's
no folder object — just blobs whose names start with `input/...`.

---

## Step 1 — Set CLI variables (set once, use everywhere)

Open Git Bash and set these. Adjust `LOC` to whatever region you picked in
Phase 1, and pick a unique suffix for `ST`:

```bash
RG=rg-timefold-dev
LOC=eastus                              # or eastus / westeurope / whatever you used
ST=sttimefolddevseiya                      # MUST be globally unique, 3-24 lowercase letters+digits
CONTAINER=timefold
```

> **About the storage account name:** it has to be unique across the *entire
> Azure universe* (because it becomes a DNS subdomain like
> `sttimefolddevseiya.blob.core.windows.net`). If your first choice is taken,
> add more digits or another suffix. The `st` prefix is a common Azure naming
> convention for "storage."

Verify the variables stuck:
```bash
echo "RG=$RG  LOC=$LOC  ST=$ST  CONTAINER=$CONTAINER"
```

---

## Step 2 — Create the storage account (cheapest tier)

```bash
az storage account create \
  --name $ST \
  --resource-group $RG \
  --location $LOC \
  --sku Standard_LRS \
  --kind StorageV2 \
  --access-tier Hot \
  --allow-blob-public-access false \
  --min-tls-version TLS1_2
```

What each flag means:
- `--sku Standard_LRS` — cheapest redundancy: **L**ocally **R**edundant **S**torage. Three copies in one datacenter. Fine for dev; for prod we'd use ZRS or GRS.
- `--kind StorageV2` — modern general-purpose account (the only one you should pick today).
- `--access-tier Hot` — files are read frequently. (We're using almost nothing so the tier choice doesn't matter financially.)
- `--allow-blob-public-access false` — **belt-and-braces security**: even if someone accidentally configures a container as public, this account-level switch blocks it. We'll use SAS URLs instead.
- `--min-tls-version TLS1_2` — modern TLS only.

It takes ~30 seconds. When done you'll see a big JSON blob — important parts
already named so you can ignore most of it.

Confirm:
```bash
az storage account show --name $ST --resource-group $RG --query "{name:name, location:location, sku:sku.name, kind:kind}" -o table
```

---

## Step 3 — Grant YOUR user the data-plane role (critical step)

This trips up everyone the first time. The "Owner" or "Contributor" role you
have on the subscription/resource group **does NOT** include blob data access.
Azure separates *management plane* (create/delete the storage account) from
*data plane* (read/write the bytes inside).

For data plane, you need **`Storage Blob Data Contributor`** specifically.

```bash
# Find your AAD user object id
USER_OID=$(az ad signed-in-user show --query id -o tsv)
echo "Your user object id: $USER_OID"

# Find the storage account resource id
ST_ID=$(az storage account show --name $ST --resource-group $RG --query id -o tsv)
echo "Storage account id: $ST_ID"

# Grant the data role to yourself, scoped to this storage account
az role assignment create \
  --assignee "$USER_OID" \
  --role "Storage Blob Data Contributor" \
  --scope "$ST_ID"
```

The grant takes **30–60 seconds to propagate**. Run this loop until it stops
erroring (or just wait a minute):

```bash
until az storage container list --account-name $ST --auth-mode login --query "[].name" -o tsv 2>/dev/null; do
  echo "waiting for RBAC to propagate..."; sleep 5
done
echo "RBAC ready."
```

The empty output (no container names) is correct — we haven't created any yet.

> Why `--auth-mode login`: by default `az storage` commands try to use the
> storage account *key* (a long-lived secret). We turn that off and use AAD
> instead. **Never use account keys** — they're impossible to rotate cleanly.

---

## Step 4 — Create the container

```bash
az storage container create \
  --name $CONTAINER \
  --account-name $ST \
  --auth-mode login \
  --public-access off
```

Verify:
```bash
az storage container list --account-name $ST --auth-mode login -o table
```
You should see a row for `timefold`.

---

## Step 5 — Upload a test YAML

Use one of the sample YAMLs from your local project:

```bash
# Adjust the local path to where your YAML lives
LOCAL_YAML=/c/Users/Seiya/Desktop/work/Timefold/web/Timefold/src/main/resource/EnvConfig.yaml

az storage blob upload \
  --account-name $ST \
  --container-name $CONTAINER \
  --auth-mode login \
  --name "input/test-run-001/EnvConfig.yaml" \
  --file "$LOCAL_YAML" \
  --overwrite
```

Confirm it landed:
```bash
az storage blob list \
  --account-name $ST \
  --container-name $CONTAINER \
  --auth-mode login \
  --query "[].{name:name, size:properties.contentLength}" \
  -o table
```

You should see one row, name `input/test-run-001/EnvConfig.yaml`, size in bytes.

---

## Step 6 — Read it back from the cloud

Download the blob to a different local path so we know it really round-tripped:

```bash
az storage blob download \
  --account-name $ST \
  --container-name $CONTAINER \
  --auth-mode login \
  --name "input/test-run-001/EnvConfig.yaml" \
  --file "/tmp/downloaded.yaml"

head -20 /tmp/downloaded.yaml      # show the first 20 lines
```

If you see your EnvConfig contents — Blob is working end-to-end.

---

## Step 7 — Generate a SAS URL (what the API will hand to the browser)

A **SAS URL** is a regular HTTPS URL with a signed query string that grants
limited, time-bound access to one blob. The web app's "Download Result"
button will receive one of these and follow it directly.

```bash
# Expiry: 1 hour from now (UTC, ISO 8601)
EXPIRY=$(date -u -d '+1 hour' '+%Y-%m-%dT%H:%MZ')
echo "Expires at: $EXPIRY"

az storage blob generate-sas \
  --account-name $ST \
  --container-name $CONTAINER \
  --name "input/test-run-001/EnvConfig.yaml" \
  --permissions r \
  --expiry "$EXPIRY" \
  --auth-mode login \
  --as-user \
  --https-only \
  --full-uri \
  --output tsv
```

That prints a URL like:
```
https://sttimefolddevseiya.blob.core.windows.net/timefold/input/test-run-001/EnvConfig.yaml?sv=2024-11-04&sr=b&...&sig=...
```

**Paste that URL into your browser.** You should see the YAML content (or get
a "save as" prompt). After 1 hour, the same URL returns `403 Forbidden` —
that's the security model in action.

> **User-delegation SAS**: `--as-user` makes the SAS signed by your AAD
> identity (good — it inherits your RBAC and is revocable if you lose your
> credentials). Without `--as-user`, the CLI would use the account key,
> which is exactly what we're trying to avoid.

---

## Step 8 — Set up a lifecycle policy (cost guardrail, recommended)

Auto-delete anything older than 90 days so abandoned test runs don't pile up.
Adjust the number to taste.

```bash
cat > /tmp/lifecycle.json <<'EOF'
{
  "rules": [
    {
      "enabled": true,
      "name": "expire-old-blobs",
      "type": "Lifecycle",
      "definition": {
        "actions": {
          "baseBlob": {
            "delete": { "daysAfterModificationGreaterThan": 90 }
          }
        },
        "filters": {
          "blobTypes": ["blockBlob"],
          "prefixMatch": ["timefold/"]
        }
      }
    }
  ]
}
EOF

az storage account management-policy create \
  --account-name $ST \
  --resource-group $RG \
  --policy @/tmp/lifecycle.json
```

This silently deletes any blob in the `timefold/` container that hasn't been
modified in 90+ days. Hard to forget about old data with this on.

---

## What you should have at the end of Phase 2

- [ ] Storage account `$ST` exists in `rg-timefold-dev`
- [ ] You have the `Storage Blob Data Contributor` role on it
- [ ] Container `timefold` exists, private
- [ ] Test YAML `input/test-run-001/EnvConfig.yaml` uploaded
- [ ] You can `az storage blob download` it back to disk
- [ ] You generated a SAS URL and downloaded via browser
- [ ] Lifecycle policy is set (deletes blobs >90 days old)

When all green, tell me **"Phase 2 done"** and we'll do Phase 3 — Container
Apps environment + a hello-world API.

---

## Cost reality check

After completing Phase 2 your storage account holds maybe 5 KB. Monthly cost
to keep this around:

| Item                          | Cost                       |
| ----------------------------- | -------------------------- |
| 5 KB hot blob                 | < $0.01                    |
| 1000 read operations          | < $0.01                    |
| Lifecycle policy              | free                       |
| Account just sitting there    | $0                         |

You can leave this storage account up forever without watching the bill.

---

## Troubleshooting

| Symptom                                                          | Cause                                              | Fix                                                                          |
| ---------------------------------------------------------------- | -------------------------------------------------- | ---------------------------------------------------------------------------- |
| `The specified account name is already taken`                    | Storage account names are global                   | Add a few digits / different suffix to `$ST` and retry                       |
| `AuthorizationPermissionMismatch` on blob upload                 | RBAC hasn't propagated yet, or wrong role          | Wait 60 s and retry; double-check `Storage Blob Data Contributor` is assigned |
| `--auth-mode login` errors despite recent sign-in                | Default tenant mismatch                            | `az account show` → confirm tenant, then `az login --tenant <tenantId>`      |
| SAS URL returns `AuthenticationFailed` in browser                | Clock skew (rare) or expired                       | Regenerate with a longer expiry; check your PC clock is correct              |
| `date -d '+1 hour'` errors                                       | macOS BSD date (different syntax)                  | On macOS: `date -u -v+1H '+%Y-%m-%dT%H:%MZ'`; on Git Bash / Linux the GNU form works |
| Portal upload works, CLI doesn't                                 | Portal uses your AAD token automatically; CLI needs explicit `--auth-mode login` | Add the flag, or set `AZURE_STORAGE_AUTH_MODE=login` env var               |
| `az role assignment create` errors with "Insufficient privileges" | You don't have `Owner` on the storage account     | Personal subscriptions: re-run `az login --scope https://management.core.windows.net/.default --tenant <tenantId>` to refresh tokens |

---

## What's next (Phase 3 preview)

Phase 3 deploys a tiny **Hello World** container as an Azure Container Apps
HTTP app. We'll:

1. Create an ACA environment in `rg-timefold-dev`
2. Deploy a `mcr.microsoft.com/k8se/quickstart:latest` image (no code yet)
3. Hit its public URL and see "Hello"
4. Configure scale-to-zero so it costs $0 idle
5. Assign it a Managed Identity (the same one will get the Blob role in
   Phase 4 so it can talk to the storage account we just created)

By the end of Phase 3 you'll understand ACA without needing to write any
backend code yet. Phase 4 is when we replace the hello-world with a real
API Controller (Node/Express) that uses the storage account from this phase.
