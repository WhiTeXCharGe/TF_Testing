# Company-PC Workflow — what YOU do after the leader provisions resources

You don't have permission on the company subscription to create resources,
budgets, or RBAC roles. The leader / Azure admin must provision those
**before you can start**. Once they do, this doc is everything you can do
yourself, in order.

> Send [`Azure-Company-Permission-Request.md`](./Azure-Company-Permission-Request.md)
> to your leader first. Don't proceed until they confirm everything in
> "What the leader provides back to you" section of that doc is done.

---

## What the leader gives you (verify before starting)

You should have received from the leader:

| Item                                 | Looks like                                                                    |
| ------------------------------------ | ----------------------------------------------------------------------------- |
| Subscription ID                      | `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`                                        |
| Resource group name                  | `rg-timefold-prod` (or whatever they chose)                                   |
| Storage account name                 | `sttimefoldprod<suffix>`                                                      |
| Storage container name               | `timefold`                                                                    |
| ACR name + login server              | `acrtimefoldprod<suffix>` / `acrtimefoldprod<suffix>.azurecr.io`              |
| ACA environment name                 | `cae-timefold-prod`                                                           |
| ACA app name                         | `ca-tf-api`                                                                   |
| ACA app public URL                   | `https://ca-tf-api.<random>.<region>.azurecontainerapps.io`                   |
| Region                               | e.g. `japaneast`                                                              |
| Confirmation YOU have these roles:   | Reader on RG; `Storage Blob Data Contributor` on storage; `AcrPush` on ACR; `Container Apps Contributor` on the ACA app |
| Confirmation the ACA APP's MI has:   | `Storage Blob Data Contributor` on storage; `AcrPull` on ACR                  |

**If anything in this list is missing, stop and message the leader.** Don't
try to provision around them — you'll either fail (no permission) or
create rogue resources outside their governance.

---

## Step 0 — One-time setup on the company PC

### 0a. Install tools
- **Git** — https://git-scm.com/download/win
- **Docker Desktop** — https://docs.docker.com/desktop/install/windows-install/ (open it once, wait for whale icon steady)
- **Node 20+** — https://nodejs.org/ (LTS installer)
- **Azure CLI** — https://aka.ms/installazurecliwindows

### 0b. Sign in with your COMPANY Azure account
```bash
az login                                  # browser opens — pick the company account
az account list --output table            # confirm subscription appears
az account set --subscription "<the subscription id from the leader>"
az account show --query "{name:name, id:id}" -o table
```

### 0c. Save the company env to a script
```bash
cat > ~/azure-timefold-company-env.sh <<'EOF'
# Fill these with what the leader provided
export SUB_ID=<subscription-id>
export RG=rg-timefold-prod
export LOC=japaneast
export ST=sttimefoldprod<suffix>
export CONTAINER=timefold
export ACR=acrtimefoldprod<suffix>
export ACR_LOGIN=${ACR}.azurecr.io
export ACA_ENV=cae-timefold-prod
export ACA_APP=ca-tf-api
export APP_URL=ca-tf-api.<random>.<region>.azurecontainerapps.io
EOF

source ~/azure-timefold-company-env.sh
echo "Sourced. ST=$ST  ACR=$ACR  APP_URL=https://$APP_URL"
```

### 0d. Verify access (read-only checks; cheap, won't change anything)
```bash
# Should succeed — proves Reader access to RG
az group show --name $RG -o table

# Should succeed — proves Blob role on storage
az storage container list --account-name $ST --auth-mode login -o table

# Should succeed — proves AcrPush on ACR
az acr login --name $ACR
docker logout $ACR_LOGIN     # immediately log out; we just wanted to confirm login works

# Should succeed — proves ACA visibility
az containerapp show --name $ACA_APP --resource-group $RG --query "{name:name, fqdn:properties.configuration.ingress.fqdn}" -o table
```

If any of these say "AuthorizationFailed" or "permission denied", the
leader hasn't finished granting roles. Stop here, send them the relevant
command output, ask them to fix.

---

## Step 1 — Clone the project

You need two branches from the project repo:

```bash
cd /c/Users/YourName/Desktop
git clone https://github.com/WhiTeXCharGe/TF_Testing.git
cd TF_Testing
```

The repo has two branches with the code you need:
- `SchedulerWeb` — the React webapp
- `TimefoldSolver` — the Java solver + Dockerfile

You'll also need the API Controller code (currently in `web/api-controller/`
on the personal machine). Either push it to a new branch first
(`ApiController`) and pull it here, or copy it via a USB stick / file share
if internet sharing is restricted.

```bash
git checkout SchedulerWeb      # default branch
# you should see Dockerfile, src/, vite.config.ts, public/, web/webapp/azure/...
```

---

## Step 2 — Push the Timefold Docker image to the company ACR

This step uses the same Dockerfile from the personal flow — just with a
different ACR name.

```bash
git checkout TimefoldSolver
ls                                      # should show Dockerfile, pom.xml, src/, docker/, etc.

az acr login --name $ACR

TAG=v1
docker build -t $ACR_LOGIN/timefold:$TAG .
docker push $ACR_LOGIN/timefold:$TAG

az acr repository show-tags --name $ACR --repository timefold -o table
```

When you see `v1` in the tags list, the image is in the company registry.

---

## Step 3 — Push the API Controller image to the company ACR

```bash
cd ../api-controller     # wherever the api-controller folder lives on this machine

az acr login --name $ACR

docker build -t $ACR_LOGIN/api-controller:v1 .
docker push $ACR_LOGIN/api-controller:v1

az acr repository show-tags --name $ACR --repository api-controller -o table
```

---

## Step 4 — Deploy the API to the company ACA app

```bash
az containerapp update \
  --name $ACA_APP \
  --resource-group $RG \
  --image $ACR_LOGIN/api-controller:v1 \
  --set-env-vars STORAGE_ACCOUNT=$ST BLOB_CONTAINER=$CONTAINER

az containerapp ingress update \
  --name $ACA_APP \
  --resource-group $RG \
  --target-port 8080

# Wait ~30 seconds, then test
curl https://$APP_URL/health
# {"ok":true}
```

---

## Step 5 — Wire the webapp to the company API

```bash
cd /c/Users/YourName/Desktop/TF_Testing      # back to the webapp branch
git checkout SchedulerWeb
```

Create `.env.local`:
```bash
cat > .env.local <<EOF
VITE_API_BASE_URL=https://$APP_URL
EOF
```

Install + run:
```bash
npm install
npm run dev
```

Open http://localhost:5173 — the **Azure** badge should appear in the top
bar (assuming you applied the Phase 7 changes to the webapp).

---

## Step 6 — Run the demo

1. **New Run** in the webapp → drag the two YAML files → optionally paste original paths → Submit
2. Confirm in the portal: storage account → `timefold` container → `input/<runId>/` shows both YAMLs
3. In a side terminal, fake completion (no Batch yet on company side):
   ```bash
   curl -X POST https://$APP_URL/mock-complete/<runId>
   ```
4. **Show Result** in the webapp → opens the editor placeholder dialog; the result-cell box shows the saved output path
5. **Delete** → confirm → blobs are gone

That's your end-to-end demo: webapp on your laptop, API in Azure Container
Apps, files in Azure Blob Storage, all with the company's resources.

---

## Things you CAN'T do without leader help

If the leader granted exactly the roles in the request letter, here's what
you CAN'T do — flag any of these to them if you hit them:

- **Create or delete resources** (storage accounts, ACR, ACA env, etc.).
  You're a user of pre-provisioned resources only.
- **Change RBAC role assignments** of any kind. Even on resources you
  "own" via Contributor — RBAC management is a separate role.
- **Set or modify budgets / cost alerts.**
- **Provision Azure Batch** or change Batch quotas — separate request when
  the team is ready for the compute layer.
- **Add new Managed Identities or change which MIs are attached** to ACA /
  Batch pools.

For all of these, ask the leader. Document each request in writing so the
audit trail is clean.

---

## When something doesn't work

Most "permission denied" errors fall into one of three buckets:

| Bucket                  | Symptom                                            | Who fixes it                                       |
| ----------------------- | -------------------------------------------------- | -------------------------------------------------- |
| Missing user role       | CLI says "AuthorizationFailed" with role name      | Leader — they grant the listed role                |
| Missing MI role         | ACA app logs say "AuthorizationPermissionMismatch" reading Blob | Leader — they grant Blob Data Contributor to the ACA app's MI |
| Wrong subscription      | "SubscriptionNotFound" or resource shows up empty  | You — run `az account set --subscription $SUB_ID`  |

Copy the exact error message when you escalate to the leader — they need it
to find the right role.

---

## Clean shutdown at end of demo (optional)

You can't delete resources, but you can stop them costing money by scaling
the ACA app to 0 manually (it'll do this automatically after 5 min idle anyway):

```bash
# Look up the latest revision name
REVISION=$(az containerapp revision list --name $ACA_APP --resource-group $RG --query "[0].name" -o tsv)

# Force it down (it'll wake on next request)
az containerapp revision deactivate --name $ACA_APP --resource-group $RG --revision $REVISION
```

Or just close your laptop — ACA scales to zero on its own.

For ACR and Storage you can't stop billing yourself; those are leader
decisions.
