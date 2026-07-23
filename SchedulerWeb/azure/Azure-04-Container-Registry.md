# Phase 4 — Azure Container Registry + push Timefold image

**Goal of this phase:** your Timefold Docker image (the one in
`web/Timefold/`) lives in a private Azure Container Registry, ready for
Azure Batch (Phase 5) and the API Controller's ACA app (Phase 6) to pull.

By the end:
- ACR account exists in `rg-timefold-dev`
- Your local image is tagged for ACR and pushed
- `az acr repository list` shows `timefold` with at least one tag

**Time:** ~20 minutes (most of it is the push, which depends on your upload speed).
**Cost:** ~$5/month flat for ACR Basic — **this is the first non-zero recurring cost** in the project. See cost section at the bottom.
**Prereqs:** Phase 3 done. Docker Desktop installed and running (Phase 4 of Docker.md).

---

## Concepts (1 min read)

| Concept           | What it is                                                                                                              |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------- |
| **ACR**           | Azure's private Docker registry. Same role as Docker Hub or GitHub Container Registry, just integrated with Azure auth.  |
| **Repository**    | A named bucket inside the registry that holds versions (tags) of one image. E.g. `timefold`.                            |
| **Tag**           | A version label on an image. `timefold:v1`, `timefold:latest`, `timefold:2025-06-04` are all tags of the same repo.     |
| **Full image ref**| `<acrName>.azurecr.io/<repository>:<tag>` — the unique address Docker uses to pull.                                     |
| **SKU**           | Pricing tier: **Basic** ($5/mo), **Standard** ($20/mo), **Premium** ($50/mo). Basic is fine for dev — same image format, just less storage and no geo-replication. |

---

## Step 0 — Source env, confirm prereqs

```bash
source ~/azure-timefold-env.sh
echo "RG=$RG  LOC=$LOC"

# Docker must be running (whale icon steady in system tray)
docker --version

# ACR provider should already be Registered (Phase 2.2)
az provider show --namespace Microsoft.ContainerRegistry --query registrationState -o tsv
```

If the provider isn't `Registered` yet:
```bash
az provider register --namespace Microsoft.ContainerRegistry
until [ "$(az provider show --namespace Microsoft.ContainerRegistry --query registrationState -o tsv)" = "Registered" ]; do
  echo "waiting..."; sleep 5
done
```

---

## Step 1 — Create the ACR

```bash
ACR=acrtimefolddevseiya       # 5-50 lowercase alphanumeric, GLOBALLY unique (becomes <name>.azurecr.io)

az acr create \
  --name $ACR \
  --resource-group $RG \
  --location $LOC \
  --sku Basic \
  --admin-enabled false
```

Flag notes:
- `--sku Basic` — $5/mo, 10 GB storage. Plenty for our solver image (~300 MB).
- `--admin-enabled false` — disables the legacy admin username/password. We'll authenticate with AAD (your `az login` session), no secrets.

Provisioning takes ~30 seconds. Confirm:
```bash
az acr show --name $ACR --resource-group $RG --query "{name:name, loginServer:loginServer, sku:sku.name, adminEnabled:adminUserEnabled}" -o table
```

The `loginServer` field is your full registry hostname — should be
`acrtimefolddevseiya.azurecr.io`.

Save to env script:
```bash
ACR_LOGIN=$(az acr show --name $ACR --resource-group $RG --query loginServer -o tsv)
ACR_ID=$(az acr show --name $ACR --resource-group $RG --query id -o tsv)

cat >> ~/azure-timefold-env.sh <<EOF

# Phase 4 additions
export ACR=$ACR
export ACR_LOGIN=$ACR_LOGIN
export ACR_ID=$ACR_ID
EOF

source ~/azure-timefold-env.sh
echo "ACR_LOGIN=$ACR_LOGIN"
```

---

## Step 2 — Authenticate Docker to ACR

`az acr login` exchanges your AAD token for a short-lived Docker credential
and stuffs it into your Docker daemon. No password to copy or store.

```bash
az acr login --name $ACR
```

Expected output ends with `Login Succeeded`. If it errors, the most common
cause is Docker Desktop not running — open it, wait for the whale icon to
go steady, try again.

---

## Step 3 — Build the Timefold image, tagged for ACR

The Dockerfile in `web/Timefold/` already builds an image. Now we just
build it with a name that points at our registry:

```bash
cd /c/Users/Seiya/Desktop/work/Timefold/web/Timefold

TAG=v1
docker build -t $ACR_LOGIN/timefold:$TAG .
```

This is the same multi-stage build from Docker.md — Maven downloads (first
time only), compiles the jar, then copies into a JRE-only image. **First
build takes 5–10 minutes**; later rebuilds reuse cache and finish in seconds
unless `pom.xml` changes.

Verify the local image exists with its new name:
```bash
docker images "$ACR_LOGIN/timefold"
```
Should show one row, ~280 MB.

---

## Step 4 — Push to ACR

```bash
docker push $ACR_LOGIN/timefold:$TAG
```

This uploads each layer of the image to ACR. Layers shared with the base
image (eclipse-temurin) push first; your application layers (the jar +
entrypoint) are smaller. Upload time depends entirely on your internet
speed — typical: 3–10 minutes for ~280 MB total over residential broadband.

When it finishes you'll see "X: digest: sha256:..." lines for each layer.

---

## Step 5 — Confirm the image is in ACR

```bash
# Repositories in this registry
az acr repository list --name $ACR -o table

# Tags of the timefold repository
az acr repository show-tags --name $ACR --repository timefold -o table

# Full metadata for this specific tag
az acr repository show --name $ACR --image timefold:$TAG \
  --query "{name:name, tag:tag, size:size, lastUpdated:lastUpdateTime}" -o table
```

You should see:
- One repository: `timefold`
- One tag: `v1`
- Size around 280 MB

---

## Step 6 — (Optional) Smoke-test pulling from ACR

To prove ACR auth works end-to-end, delete your local image and pull it back:

```bash
docker rmi $ACR_LOGIN/timefold:$TAG
docker pull $ACR_LOGIN/timefold:$TAG
```

If the pull succeeds, your auth + storage round-trip works.

---

## What you should have at the end of Phase 4

- [ ] ACR `acrtimefolddevseiya` exists in `rg-timefold-dev`, SKU Basic, admin-user disabled
- [ ] `$ACR_LOGIN` set to `acrtimefolddevseiya.azurecr.io`
- [ ] Local Docker built `$ACR_LOGIN/timefold:v1` from `web/Timefold/`
- [ ] `docker push` succeeded
- [ ] `az acr repository list` shows `timefold`; `show-tags` shows `v1`
- [ ] Env script updated with `$ACR`, `$ACR_LOGIN`, `$ACR_ID`

Tell me **"Phase 4 done"** and we'll do Phase 5 — the big one. Create the
Batch account + pool, grant the pool's identity AcrPull on this registry,
and **run the Timefold image as a Batch task once via CLI**. That's the
end-to-end proof we've been building toward.

---

## Cost reality check

| Item                                         | Cost                              |
| -------------------------------------------- | --------------------------------- |
| ACR Basic (this phase forward)               | **$5/month flat**                 |
| Storage for the image (~280 MB)              | included in the $5                |
| Image pulls (Batch / ACA / local)            | free in-region                    |
| Phase 2 storage + Phase 3 ACA still          | ~$0                               |

**Total project cost after Phase 4 ≈ $5/month** while ACR exists.

### How to stop paying when you're done with the project

```bash
# Nukes ONLY the registry (keeps storage and ACA)
az acr delete --name $ACR --resource-group $RG --yes

# Or nuke everything in one go
az group delete --name $RG --yes --no-wait
```

For your dev account, $5/mo is fine. For the company subscription, the same
ACR Basic + the actual compute usage is the realistic cost line item for
the access doc.

---

## Troubleshooting

| Symptom                                                      | Cause                                              | Fix                                                                          |
| ------------------------------------------------------------ | -------------------------------------------------- | ---------------------------------------------------------------------------- |
| `az acr create` says name already taken                      | ACR names are global                               | Add digits to `$ACR` and retry                                               |
| `az acr login` errors with `Error response from daemon`      | Docker Desktop not running                         | Open Docker Desktop; wait for whale icon to be steady; retry                 |
| `docker build` fails with `Non-resolvable parent POM`        | Building outside the Docker context that includes `docker/pom-standalone.xml` | Run from `web/Timefold/` (where the Dockerfile lives), not from a parent dir |
| `docker push` hangs                                          | Slow upstream / large image                        | Be patient. Use a wired connection if on flaky wifi. Push resumes if it drops. |
| `denied: requested access to the resource is denied`         | Your `az acr login` token expired (24 h)           | `az acr login --name $ACR` again                                             |
| `repository name must be lowercase`                          | Capital letters in tag/repo name                   | Use lowercase only (`$ACR_LOGIN/timefold:v1`)                                |

---

## What's next (Phase 5 preview)

Phase 5 is the milestone you've been working toward:

1. Create an **Azure Batch account** in `rg-timefold-dev`
2. Create a **pool** of compute nodes (start with 0; autoscale on demand)
3. Grant the pool's Managed Identity:
   - **AcrPull** on the ACR we just made
   - **Storage Blob Data Contributor** on the storage account from Phase 2
4. Create a standing **job** to hold our tasks
5. **Submit ONE task** referencing the Timefold image:
   - reads `input/test-run-001/{EnvConfig,Schedule}.yaml` from Blob
   - writes `output/test-run-001/result_Schedule.yaml` to Blob
   - writes `status/test-run-001.json` throughout
6. Watch the node provision, image pull, task run, output appear, status flip to Completed
7. Download the result via SAS URL

After Phase 5 you'll have proven the architecture works end-to-end **without
writing any backend code yet**. Phase 6 will then automate this manual flow
into a real API Controller deployed to ACA.
