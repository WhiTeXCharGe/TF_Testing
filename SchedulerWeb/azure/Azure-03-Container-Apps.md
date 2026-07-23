# Phase 3 — Container Apps environment + hello-world API + MI → Blob wiring

**Goal of this phase:** an Azure Container Apps (ACA) environment up and
running, a tiny hello-world container deployed to it with a public URL you
can hit, scale-to-zero confirmed, and a **Managed Identity** on the app
granted the Blob role on the storage account from Phase 2.

By the end you'll have proved:
- The ACA platform works on your subscription
- Scale-to-zero is genuinely costing you $0 when nobody's calling
- The hello-world app's identity can talk to the storage account (so when
  we replace it with real API code in Phase 6, the wiring already exists)

**Time:** ~30 minutes (plus 5 min for the first ACA environment to provision).
**Cost:** $0 (ACA free tier covers our test load entirely).

**Prereqs:** Phase 1 + Phase 2 done; the env script from Phase 2.2 sourced.

---

## Concepts (read once, ~2 min)

| Concept                       | What it is                                                                                                            |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| **ACA Environment**           | A logical "cluster" — a managed Kubernetes-under-the-hood that hosts one or more apps. One environment per region per project is the norm. |
| **ACA App**                   | A container deployment with HTTP ingress, auto-scaling, and a public URL. Lives inside an environment.                |
| **Ingress**                   | The HTTPS endpoint your app exposes. External = public internet; Internal = only reachable from inside the env.       |
| **Revision**                  | A snapshot of an app's config (image tag, env vars, scale rules). Updating the app creates a new revision; you can roll back. |
| **Scale rule**                | A trigger that adds/removes replicas: HTTP requests, queue length, CPU, custom KEDA scalers. Min replicas = 0 → scale to zero. |
| **Managed Identity (MI)**     | An Azure-managed credential attached to the app. Use it to authenticate to other Azure services (Blob, Batch, etc.) without storing secrets in code. |

> **Why ACA over App Service or AKS:** ACA is the lowest-ops way to run an
> HTTP container that scales to zero. App Service is always-on (more $),
> AKS is a whole Kubernetes cluster (way more ops).

---

## Step 0 — Source your env vars and confirm prereqs

```bash
source ~/azure-timefold-env.sh
echo "RG=$RG  LOC=$LOC  ST=$ST"

# Make sure ACA providers are registered (you queued these in Phase 2.2 — finish waiting)
for ns in Microsoft.App Microsoft.OperationalInsights; do
  until [ "$(az provider show --namespace $ns --query registrationState -o tsv)" = "Registered" ]; do
    echo "waiting for $ns..."; sleep 5
  done
  echo "$ns is Registered."
done

# Install the containerapp CLI extension (silent if already installed)
az extension add --name containerapp --upgrade
```

---

## Step 1 — Create the ACA environment

```bash
ACA_ENV=cae-timefold-dev

az containerapp env create \
  --name $ACA_ENV \
  --resource-group $RG \
  --location $LOC \
  --logs-destination none
```

What's happening:
- `cae-` prefix = "Container App Environment" naming convention.
- `--logs-destination none` skips creating a Log Analytics workspace. Saves
  ~$2/mo and we don't need centralized logging yet for dev. You can attach
  one later with a single command if you do.

**This takes 3–7 minutes the first time** (Microsoft is provisioning the
underlying managed Kubernetes for you). Subsequent app deployments to the
same environment are fast.

Confirm:
```bash
az containerapp env show --name $ACA_ENV --resource-group $RG --query "{name:name, location:location, state:properties.provisioningState}" -o table
```
`state` should say `Succeeded`.

---

## Step 2 — Deploy a hello-world app

We'll use Microsoft's official quickstart image. No code from us yet.

```bash
ACA_APP=ca-tf-api

az containerapp create \
  --name $ACA_APP \
  --resource-group $RG \
  --environment $ACA_ENV \
  --image mcr.microsoft.com/k8se/quickstart:latest \
  --target-port 80 \
  --ingress external \
  --min-replicas 0 \
  --max-replicas 1 \
  --cpu 0.25 \
  --memory 0.5Gi \
  --system-assigned
```

- `ca-` prefix = "Container App" naming convention.
- `--ingress external` exposes a public HTTPS URL (Azure handles TLS).
- `--min-replicas 0` = scale-to-zero when idle (the whole point).
- `--max-replicas 1` = cap at 1 instance for our test (cost safety).
- `--cpu 0.25 --memory 0.5Gi` = smallest sensible sizing for a placeholder.
- `--system-assigned` = create a **system-assigned Managed Identity** for
  this app. We'll grant it Blob access next.

Provisioning takes ~30 seconds. When done, grab the public URL:

```bash
APP_URL=$(az containerapp show --name $ACA_APP --resource-group $RG --query "properties.configuration.ingress.fqdn" -o tsv)
echo "https://$APP_URL"
```

---

## Step 3 — Verify the app responds

Paste the `https://...` URL from above into your browser. You should see
the Microsoft "Welcome to Azure Container Apps" page.

Or from the terminal:
```bash
curl -s "https://$APP_URL" | head -10
```
You'll see HTML — that's the hello-world container's response.

---

## Step 4 — Watch scale-to-zero happen

The whole point of ACA is "no traffic = no charge." After 5–10 minutes of
no requests, your app drops to zero replicas.

```bash
# Right now (just after browsing it) — probably 1 replica
az containerapp revision list --name $ACA_APP --resource-group $RG --query "[].{name:name, active:properties.active, replicas:properties.replicas}" -o table

# Wait 10 minutes, then run the same command — replicas should be 0
```

Next request to the URL after scale-to-zero triggers a **cold start**
(takes 1–3 seconds) and the replica comes back. That's the entire scale
model — there's nothing else to configure.

---

## Step 5 — Grab the app's Managed Identity object id

```bash
APP_MI_OID=$(az containerapp show --name $ACA_APP --resource-group $RG --query "identity.principalId" -o tsv)
echo "App Managed Identity OID: $APP_MI_OID"
```

This is the AAD object id of the system-assigned identity Azure created
when we passed `--system-assigned` in Step 2. Treat it like a "service
account" — it's how the app authenticates to other Azure services.

---

## Step 6 — Grant the MI Blob access on the Phase 2 storage account

```bash
az role assignment create \
  --assignee-object-id "$APP_MI_OID" \
  --assignee-principal-type ServicePrincipal \
  --role "Storage Blob Data Contributor" \
  --scope "$ST_ID"
```

Note the difference from Phase 2: `--assignee-principal-type ServicePrincipal`
(in Phase 2 it was `User`, because that was YOU). Managed Identities are
classified as ServicePrincipals in AAD.

Verify:
```bash
az role assignment list \
  --assignee "$APP_MI_OID" \
  --scope "$ST_ID" \
  --query "[].{role:roleDefinitionName, scope:scope}" \
  -o table
```
Should show one row with `Storage Blob Data Contributor`.

---

## Step 7 — Save the new IDs to your env script

So they survive across sessions:

```bash
cat >> ~/azure-timefold-env.sh <<EOF

# Phase 3 additions
export ACA_ENV=$ACA_ENV
export ACA_APP=$ACA_APP
export APP_MI_OID=$APP_MI_OID
export APP_URL=$APP_URL
EOF
```

Re-source it to confirm it works:
```bash
source ~/azure-timefold-env.sh
echo "ACA_APP=$ACA_APP  APP_URL=https://$APP_URL"
```

---

## Step 8 — (Optional) Test the MI from inside the container

The hello-world image is just nginx, so we can't easily test from inside.
What we *can* do is confirm the MI is wired correctly by simulating what
real code would do, from your local CLI but acting as the MI:

```bash
# This isn't a real test of the MI itself, but it does verify:
# 1. The role assignment is correct
# 2. The storage account is reachable via AAD auth
az storage blob list \
  --account-name $ST \
  --container-name $CONTAINER \
  --auth-mode login \
  -o table
```

If you see your test blob from Phase 2, the data-plane wiring works. When
we deploy real API code in Phase 6, it'll use `DefaultAzureCredential`
which automatically picks up the MI on the ACA app — same path, just
authenticated as the MI instead of you.

---

## What you should have at the end of Phase 3

- [ ] `Microsoft.App` and `Microsoft.OperationalInsights` providers Registered
- [ ] ACA environment `cae-timefold-dev` provisioned (`state: Succeeded`)
- [ ] ACA app `ca-tf-api` running with a public HTTPS URL
- [ ] Browser shows the "Welcome to Azure Container Apps" page
- [ ] App has a system-assigned Managed Identity
- [ ] MI has `Storage Blob Data Contributor` on the storage account
- [ ] `$APP_MI_OID`, `$ACA_ENV`, `$ACA_APP`, `$APP_URL` saved in your env script

Tell me **"Phase 3 done"** and we'll do Phase 4 — Azure Container Registry +
pushing your `web/Timefold/` Docker image to it.

---

## Cost reality check

| Item                          | Cost                                                    |
| ----------------------------- | ------------------------------------------------------- |
| ACA environment (idle)        | $0                                                      |
| ACA app (scaled to 0)         | $0                                                      |
| ACA app (running, 0.25 CPU)   | ~$0.0034/hour while active                              |
| Free tier (per month)         | 180k vCPU-seconds, 360k GiB-seconds — covers all our testing |
| Log Analytics                 | $0 (we set `--logs-destination none`)                   |

For your test usage you'll never see a charge in this phase. Even leaving
the app deployed for a month at idle = $0.

---

## Cleanup if you want to stop everything

```bash
az containerapp delete --name $ACA_APP --resource-group $RG --yes
az containerapp env delete --name $ACA_ENV --resource-group $RG --yes
```

Or nuke the entire resource group (kills Phase 2 storage too — only if
you want a clean slate):
```bash
az group delete --name $RG --yes --no-wait
```

---

## Troubleshooting

| Symptom                                                          | Cause                                              | Fix                                                                          |
| ---------------------------------------------------------------- | -------------------------------------------------- | ---------------------------------------------------------------------------- |
| `az containerapp env create` fails with `Microsoft.App not registered` | Provider not registered yet                  | Re-run the Step 0 wait loop                                                  |
| `containerapp` command not found                                 | CLI extension not installed                        | `az extension add --name containerapp --upgrade`                             |
| Env create takes >10 min                                         | Normal for the first env in a region               | Be patient; `state:Succeeded` when done                                      |
| Browser shows "404 Site Not Found" instead of welcome page       | App is in cold start (just woke from scale-to-zero) | Wait 5 s, refresh                                                            |
| `APP_MI_OID` is empty                                            | Forgot `--system-assigned` on app create           | `az containerapp identity assign --name $ACA_APP --resource-group $RG --system-assigned` |
| Role assignment fails with `MissingSubscription`                 | `Microsoft.Authorization` not registered           | See Phase 2.2                                                                |

---

## What's next (Phase 4 preview)

Phase 4 = Azure Container Registry + push your Timefold image:

1. Create an ACR (`acrtimefolddev`) in `rg-timefold-dev`
2. From `web/Timefold/`, build the image and push it to ACR
3. Confirm ACA + Batch can pull from it

Then Phase 5 is the big one: create a Batch account, a pool, and **run the
Timefold container once via CLI**. End-to-end proof.
