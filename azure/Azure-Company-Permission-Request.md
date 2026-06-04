# Azure Resource & Access Requirements — Timefold Scheduler

This document specifies the Azure resources and RBAC role assignments
required to deploy the Timefold scheduler. It is intended to be readable
by any reviewer in the provisioning chain.

---

## Project context

The Timefold scheduler is a multi-part system:

1. A React webapp (runs on a user's laptop) that uploads two YAML files
   (`EnvConfig.yaml` + `Schedule.yaml`) and downloads a result YAML.
2. An HTTP API service hosted on Azure Container Apps that receives those
   uploads, stores them in Azure Blob Storage, and returns a signed
   download URL for the result.
3. The Timefold solver itself, packaged as a Docker container, executed
   on demand by Azure Batch on auto-scaling compute nodes. The solver
   reads input YAMLs from Blob, writes the result YAML and status to Blob.

The full target architecture (including sequence diagrams) is documented
in [`Azure.md`](../Azure.md).

This request covers **all components needed for the full system** —
including the Azure Batch compute layer.

---

## Resources to provision

All resources live in a single resource group for clean ownership and
simple lifecycle management.

### 1. Resource group
| Property         | Value                                                        |
| ---------------- | ------------------------------------------------------------ |
| Name             | `rg-timefold-prod` *(or per company naming convention)*      |
| Region           | `japaneast` *(closest low-latency region; can substitute)*   |
| Tags             | `project=timefold`, `env=prod`                               |

### 2. Storage account + blob container
| Property                 | Value                                                                  |
| ------------------------ | ---------------------------------------------------------------------- |
| Storage account name     | `sttimefoldprod<suffix>` *(must be globally unique)*                   |
| SKU                      | `Standard_LRS` (locally redundant; cheapest)                           |
| Kind                     | `StorageV2`                                                            |
| Access tier              | `Hot`                                                                  |
| Allow public blob access | **Disabled**                                                           |
| Min TLS version          | `TLS1_2`                                                               |
| Container name           | `timefold` (private)                                                   |
| Lifecycle policy         | Auto-delete blobs older than 90 days under container `timefold/`       |

### 3. Azure Container Registry
| Property         | Value                                                                       |
| ---------------- | --------------------------------------------------------------------------- |
| Name             | `acrtimefoldprod<suffix>` *(must be globally unique)*                       |
| SKU              | `Basic` (~$5/mo flat; sufficient for the ~280 MB images)                    |
| Admin user       | **Disabled** (AAD authentication only — no shared keys)                     |

### 4. Azure Container Apps environment + HTTP app
| Property                    | Value                                                                    |
| --------------------------- | ------------------------------------------------------------------------ |
| Environment name            | `cae-timefold-prod`                                                      |
| App name                    | `ca-tf-api`                                                              |
| Initial image (placeholder) | `mcr.microsoft.com/k8se/quickstart:latest` *(replaced via revision update after roles are granted)* |
| Target port                 | `8080`                                                                   |
| Ingress                     | External (public HTTPS, Azure-managed TLS)                               |
| Min replicas                | `0` (scale-to-zero)                                                      |
| Max replicas                | `2`                                                                      |
| CPU / Memory                | 0.5 vCPU / 1 GiB                                                         |
| Managed Identity            | **System-assigned** (auto-created with the app)                          |
| Environment variables       | `STORAGE_ACCOUNT=<storage-account-name>`, `BLOB_CONTAINER=timefold`, `BATCH_ACCOUNT=<batch-account-name>`, `BATCH_POOL_ID=pool-timefold-prod` |
| Logs destination            | None (no Log Analytics dependency for this phase)                        |

### 5. Azure Batch account
| Property                | Value                                                                     |
| ----------------------- | ------------------------------------------------------------------------- |
| Name                    | `batchtimefoldprod<suffix>` *(must be globally unique)*                   |
| Linked storage account  | `sttimefoldprod<suffix>` (the storage account above; enables auto-storage for resourceFiles/outputFiles) |
| Identity                | **System-assigned** (Batch account itself uses this for ARM operations)   |
| Pool allocation mode    | Batch service                                                             |
| Public network access   | Enabled (locked down to AAD auth)                                         |

### 6. Azure Batch pool
| Property                       | Value                                                                |
| ------------------------------ | -------------------------------------------------------------------- |
| Pool ID                        | `pool-timefold-prod`                                                 |
| VM SKU                         | `Standard_F2s_v2` (2 vCPU / 4 GB; compute-optimized for solver)      |
| Node OS image                  | `microsoft-azure-batch / ubuntu-server-container 20-04-lts`          |
| Target dedicated nodes         | 0                                                                    |
| Target low-priority nodes      | 0 (autoscale 0–3 based on pending tasks)                             |
| Maximum nodes                  | 3 (cost safety cap)                                                  |
| Tasks per node                 | 1                                                                    |
| Container configuration        | dockerCompatible; registry `acrtimefoldprod<suffix>.azurecr.io`      |
| Identity                       | **User-assigned** managed identity `mi-tf-pool` (see #7)             |
| Autoscale formula              | 1 node per pending task; drop idle nodes after 10 min                |

### 7. User-assigned Managed Identity (for the Batch pool)
| Property | Value             |
| -------- | ----------------- |
| Name     | `mi-tf-pool`      |
| Region   | Same as resource group |

This identity is attached to the Batch pool. Each compute node uses it to
pull the Timefold container image from ACR and to read/write blobs.

---

## Subscription-level resource provider registrations

Required providers (one-time per subscription, no cost). Each `Microsoft.X`
is the namespace owned by one Azure service team; a provider must be
**registered** on the subscription before any resource of that type can be
created.

| Provider                          | What it owns                                                            | Why we need it                                                                  |
| --------------------------------- | ----------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| `Microsoft.Storage`               | Storage accounts, blob containers, queues, tables, file shares          | The Blob storage account where all YAMLs and `status.json` live                 |
| `Microsoft.Authorization`         | RBAC: role assignments, role definitions, locks                         | Required for every `az role assignment` — without it no permissions can be granted |
| `Microsoft.App`                   | Azure Container Apps environments + HTTP apps                           | The serverless HTTP service that hosts the API Controller                       |
| `Microsoft.OperationalInsights`   | Log Analytics workspaces                                                | A dependency of ACA even when logs are disabled (provider must still be registered) |
| `Microsoft.ContainerRegistry`     | Azure Container Registries (ACR)                                        | The private Docker registry holding the `api-controller` and `timefold` images  |
| `Microsoft.ManagedIdentity`       | User-assigned managed identities                                        | Required to create the standalone `mi-tf-pool` identity attached to the Batch pool |
| `Microsoft.Batch`                 | Batch accounts, pools, jobs, tasks                                      | The compute layer that runs the Timefold solver as containerized tasks          |

Verification:
```bash
az provider list --query "[?contains(['Microsoft.Storage','Microsoft.Authorization','Microsoft.App','Microsoft.OperationalInsights','Microsoft.ContainerRegistry','Microsoft.ManagedIdentity','Microsoft.Batch'], namespace)].{name:namespace, state:registrationState}" -o table
```
For any provider showing `NotRegistered`:
```bash
az provider register --namespace Microsoft.<X>
```

---

## vCPU quota request (Azure Batch)

Personal and new company subscriptions often have **0 vCPU quota** for
Batch. The quota must be increased before any pool can scale up.

| Quota                                    | Requested value | Justification                                                       |
| ---------------------------------------- | --------------- | ------------------------------------------------------------------- |
| Batch — Total Low-priority vCPUs         | `6`             | 3 max nodes × 2 vCPU each (Standard_F2s_v2 family)                  |
| Batch — Dedicated cores per VM family (Fsv2) | `4` (optional) | If dedicated (non-low-priority) VMs are ever needed for prod        |

Submitted via: Portal → **Quotas** → **Compute** → filter by subscription
+ region → find the relevant row → request increase. Typical approval
time: minutes to 48 hours.

---

## RBAC role assignments required

### A. Project user account

The user who will deploy and operate the application requires:

| Role                                  | Scope                                  | Purpose                                                          |
| ------------------------------------- | -------------------------------------- | ---------------------------------------------------------------- |
| **Reader**                            | Resource group `rg-timefold-prod`      | View resources in portal / CLI                                   |
| **Storage Blob Data Contributor**     | Storage account `sttimefoldprod*`      | Upload / download / inspect blobs from CLI during debugging      |
| **AcrPush**                           | ACR `acrtimefoldprod*`                 | Push application images to the registry                          |
| **Container Apps Contributor**        | ACA app `ca-tf-api`                    | Deploy new revisions of the API (image updates)                  |
| **Azure Batch Account Contributor**   | Batch account `batchtimefoldprod*`     | Submit and terminate Batch tasks during testing                  |

`Owner` and `User Access Administrator` are explicitly **not** required.
All RBAC management remains with the provisioning team.

### B. ACA app's system-assigned Managed Identity

After the ACA app `ca-tf-api` is created, it has a system-assigned
Managed Identity with an auto-generated principal ID. That identity
requires:

| Role                                  | Scope                                  | Purpose                                                          |
| ------------------------------------- | -------------------------------------- | ---------------------------------------------------------------- |
| **Storage Blob Data Contributor**     | Storage account `sttimefoldprod*`      | API reads/writes input, output, and status blobs                 |
| **AcrPull**                           | ACR `acrtimefoldprod*`                 | ACA pulls the API image on deploy and cold-start                 |
| **Azure Batch Account Contributor**   | Batch account `batchtimefoldprod*`     | API creates Batch tasks on `POST /runSolver` and terminates them on `POST /cancel/{runId}` |

### C. Batch pool's user-assigned Managed Identity (`mi-tf-pool`)

| Role                                  | Scope                                  | Purpose                                                          |
| ------------------------------------- | -------------------------------------- | ---------------------------------------------------------------- |
| **Storage Blob Data Contributor**     | Storage account `sttimefoldprod*`      | Compute nodes read input YAMLs, write output YAML and status.json |
| **AcrPull**                           | ACR `acrtimefoldprod*`                 | Compute nodes pull the Timefold container image                  |

**Total RBAC assignments: 9** (5 user + 3 ACA MI + 2 Pool MI).

Assignment locations:
- ACA app's MI principal ID — portal → ACA app `ca-tf-api` → **Identity** (left sidebar) → System assigned → Object (principal) ID
- Pool MI principal ID — portal → Managed Identity `mi-tf-pool` → Overview → Object (principal) ID

---

## Cost expectation

| Component               | Monthly cost (idle) | Per solve (8 hr)                       | Notes                                                       |
| ----------------------- | ------------------- | -------------------------------------- | ----------------------------------------------------------- |
| Resource group          | $0                  | $0                                     | Metadata only                                               |
| Storage account         | < $0.10             | pennies                                | A few MB of YAML; ~$0.02/GB hot tier                        |
| ACR Basic               | $5 flat             | $0                                     | 10 GB storage; image pulls free in-region                   |
| ACA environment + app   | $0                  | ~$0.01                                  | Scale-to-zero confirmed in personal-account testing         |
| Batch account           | $0                  | $0                                     | The service itself is free                                  |
| Batch pool (scaled to 0)| $0                  | $0                                     | Autoscale formula keeps the pool at zero when idle           |
| Batch — compute time    | $0                  | ~$0.30 (low-priority) – $1.60 (standard) | Per 8 hr solve on `Standard_F2s_v2`                         |
| User-assigned MI        | $0                  | $0                                     | No cost                                                     |
| **Total idle**          | **~$5/mo**          | —                                      | Dominated by ACR Basic                                      |
| **Per run**             | —                   | **~$0.30 – $1.60**                     | Mostly VM compute time                                      |

Recommended subscription-level budget alert: $30/month with notifications
at 50% / 90% / 100%.

---

## Information required after provisioning

To deploy the application code, the following values are needed:

1. Subscription ID (GUID)
2. Resource group name
3. Storage account name (final globally-unique name chosen)
4. Container name (`timefold` unless changed)
5. ACR name + login server (e.g. `acrtimefoldprodxyz` / `acrtimefoldprodxyz.azurecr.io`)
6. ACA environment name + ACA app name
7. ACA app public URL (the full `https://...azurecontainerapps.io` ingress endpoint)
8. Batch account name + URL (e.g. `batchtimefoldprodxyz` / `https://batchtimefoldprodxyz.<region>.batch.azure.com`)
9. Batch pool ID (`pool-timefold-prod`)
10. User-assigned MI name + resource ID (`mi-tf-pool` and its `/subscriptions/.../mi-tf-pool` resource ID)
11. Region used
12. Confirmation that all five user roles in section A are assigned
13. Confirmation that all three ACA-MI roles in section B are assigned
14. Confirmation that both Pool-MI roles in section C are assigned
15. Confirmation that the Batch vCPU quota has been increased

These items unblock application deployment. No further admin involvement
is required after this point.

---

## Steps performed by the project user after provisioning

For visibility into what runs after provisioning is complete:

1. `docker build` and `docker push` the API Controller image to ACR
   (using `AcrPush` role).
2. `docker build` and `docker push` the Timefold solver image to ACR.
3. `az containerapp update` to deploy the API image to `ca-tf-api`
   (using `Container Apps Contributor` role).
4. Submit a Batch task referencing the Timefold image (using `Azure Batch
   Account Contributor` role) — first run is performed manually via CLI
   to validate the pool + image pull path.
5. Run the React webapp locally with `VITE_API_BASE_URL` pointed at the
   ACA app's public URL.
6. Test the upload → status → download flow end-to-end through the API,
   which now creates real Batch tasks.

No further provisioning, RBAC changes, or admin action are required.

---

## Out of scope for this request

- **Production hardening** — Azure Entra ID authentication on the API,
  custom domain, private endpoints, geo-redundant storage. These are
  post-PoC concerns.
- **CI/CD pipeline** — manual `docker push` and `az containerapp update`
  are sufficient for this phase.

---

## Reference

See [`Azure.md`](../Azure.md) for the full v1 architecture, sequence
diagrams, and design decisions. Section 8 ("Auth and security") of that
document lists every Managed Identity and the role each one requires —
this request covers all of them.

See [`Azure-Products-Required.md`](./Azure-Products-Required.md) for a
single-page reference of the four core Azure products used by this system
(Blob, ACR, ACA, Batch) and the services explicitly **not** used.
