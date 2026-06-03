# Azure Resource & Access Requirements — Timefold Scheduler

This document specifies the Azure resources and RBAC role assignments
required to deploy the Timefold scheduler proof-of-concept. It is intended
to be readable by any reviewer in the provisioning chain.

---

## Project context

The Timefold scheduler is a two-part system:

1. A React webapp (runs on a user's laptop) that uploads two YAML files
   (`EnvConfig.yaml` + `Schedule.yaml`) and downloads a result YAML.
2. An HTTP API service hosted on Azure Container Apps that receives those
   uploads, stores them in Azure Blob Storage, and returns a signed
   download URL for the result.

The full target architecture (see [`Azure.md`](../Azure.md)) also includes
Azure Batch for the Timefold solver compute. **This request covers only
the API + Storage components.** A separate request will follow for the
Batch compute layer once the API + Storage path is validated.

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
| Environment variables       | `STORAGE_ACCOUNT=<storage-account-name>`, `BLOB_CONTAINER=timefold`      |
| Logs destination            | None (no Log Analytics dependency for this phase)                        |

---

## Subscription-level resource provider registrations

Required providers (one-time per subscription, no cost):

- `Microsoft.Storage`
- `Microsoft.Authorization`
- `Microsoft.App`
- `Microsoft.OperationalInsights`
- `Microsoft.ContainerRegistry`
- `Microsoft.ManagedIdentity`

*(`Microsoft.Batch` is **not** required at this phase; it will be requested
with the Batch compute layer in a separate request.)*

Verification:
```bash
az provider list --query "[?contains(['Microsoft.Storage','Microsoft.Authorization','Microsoft.App','Microsoft.OperationalInsights','Microsoft.ContainerRegistry','Microsoft.ManagedIdentity'], namespace)].{name:namespace, state:registrationState}" -o table
```
For any provider showing `NotRegistered`:
```bash
az provider register --namespace Microsoft.<X>
```

---

## RBAC role assignments required

### A. Project user account

The user who will deploy and operate the application requires:

| Role                                  | Scope                                | Purpose                                                          |
| ------------------------------------- | ------------------------------------ | ---------------------------------------------------------------- |
| **Reader**                            | Resource group `rg-timefold-prod`    | View resources in portal / CLI                                   |
| **Storage Blob Data Contributor**     | Storage account `sttimefoldprod*`    | Upload / download / inspect blobs from CLI during debugging      |
| **AcrPush**                           | ACR `acrtimefoldprod*`               | Push application images to the registry                          |
| **Container Apps Contributor**        | ACA app `ca-tf-api`                  | Deploy new revisions of the API (image updates)                  |

`Owner` and `User Access Administrator` are explicitly **not** required.
All RBAC management remains with the provisioning team.

### B. ACA app's system-assigned Managed Identity

After the ACA app `ca-tf-api` is created, it will have a system-assigned
Managed Identity with an auto-generated principal ID. That identity
requires:

| Role                                  | Scope                                | Purpose                                                          |
| ------------------------------------- | ------------------------------------ | ---------------------------------------------------------------- |
| **Storage Blob Data Contributor**     | Storage account `sttimefoldprod*`    | API reads/writes input, output, and status blobs                 |
| **AcrPull**                           | ACR `acrtimefoldprod*`               | ACA can pull the API image on deploy and cold-start              |

Assignment location: portal → ACA app `ca-tf-api` → **Identity** (left
sidebar) → System assigned → copy Object (principal) ID → assign the two
roles above on the respective resources.

---

## Cost expectation

| Component       | Monthly cost  | Notes                                                       |
| --------------- | ------------- | ----------------------------------------------------------- |
| Resource group  | $0            | Metadata only                                               |
| Storage account | < $0.10       | A few MB of YAML; ~$0.02/GB hot tier                        |
| ACR Basic       | $5 flat       | 10 GB storage included                                      |
| ACA environment | $0            | Pays per active replica                                     |
| ACA app idle    | $0            | Scale-to-zero confirmed in personal-account testing         |
| **Total idle**  | **~$5/mo**    | Dominated by ACR Basic                                      |
| Per demo run    | ~$0.01        | Few seconds of ACA active time + minor blob operations      |

Recommended subscription-level budget alert: $20/month with notifications
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
8. Region used
9. Confirmation that all four user roles in section A are assigned
10. Confirmation that both Managed Identity roles in section B are assigned

These ten items unblock the application deployment. No further admin
involvement is required after this point for the API + Storage phase.

---

## Steps performed by the project user after provisioning

For visibility into what runs after provisioning is complete:

1. `docker build` and `docker push` the API Controller image and the
   Timefold solver image to ACR (using `AcrPush` role).
2. `az containerapp update` to deploy the API image to `ca-tf-api`
   (using `Container Apps Contributor` role).
3. Run the React webapp locally with `VITE_API_BASE_URL` pointed at the
   ACA app's public URL.
4. Test the upload → status → download flow end-to-end.

No further provisioning, RBAC changes, or admin action are required.

---

## Out of scope for this request

- **Azure Batch** (compute layer for Timefold solver) — will be requested
  separately. That request will add a Batch account, a compute pool,
  vCPU quota for the chosen VM SKU, and a user-assigned Managed Identity
  with `AcrPull` and `Storage Blob Data Contributor` roles.
- **Production hardening** — Azure Entra ID authentication on the API,
  custom domain, private endpoints, geo-redundant storage. These are
  post-PoC concerns.
- **CI/CD pipeline** — manual `docker push` is sufficient for this phase.

---

## Reference

See [`Azure.md`](../Azure.md) for the full v1 architecture, sequence
diagrams, and design decisions. Section 8 ("Auth and security") of that
document lists every Managed Identity and the role each one requires —
this request covers the API + Storage subset of that.
