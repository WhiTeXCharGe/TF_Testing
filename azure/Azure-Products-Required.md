# Azure Products Required — Complete System Reference

When the Timefold scheduler is fully built out (webapp → API → Blob → Batch
→ result download), this is the complete list of Azure products needed.
Use this as a single-page reference when discussing scope, cost, or
permissions.

---

## The 4 products that matter

These are the ones with names you'll say in meetings:

| #   | Product                          | What it does in our system                                                                              |
| --- | -------------------------------- | ------------------------------------------------------------------------------------------------------- |
| 1   | **Azure Blob Storage**           | Stores `input/{runId}/*.yaml`, `output/{runId}/result_Schedule.yaml`, and `status/{runId}.json`         |
| 2   | **Azure Container Apps (ACA)**   | Hosts the API Controller — the HTTP service the webapp calls                                            |
| 3   | **Azure Container Registry (ACR)** | Stores the Docker images for the API Controller AND the Timefold solver. Both ACA and Batch pull from it |
| 4   | **Azure Batch**                  | Runs the Timefold solver as containerized tasks on compute nodes that auto-scale from zero              |

That's the whole compute + storage + image story. Everything else is either
built in, free, or optional.

---

## Full required list (production system)

### Core resources (you pay for these)

| Resource                               | Purpose                                                                                | Idle cost / mo          | Per-run cost                    |
| -------------------------------------- | -------------------------------------------------------------------------------------- | ----------------------- | ------------------------------- |
| **Storage account** (Blob)             | Holds input/output/status files                                                        | < $0.10                 | pennies per few MB              |
| **Container Registry** (Basic SKU)     | Holds the two Docker images (API + Timefold)                                           | $5 flat                 | $0 (pulls are free in-region)   |
| **Container Apps environment**         | Logical "cluster" hosting the API app                                                  | $0                      | $0                              |
| **Container Apps app** (`ca-tf-api`)   | The actual API container — scales to zero between calls                                | $0                      | ~$0.01 / minute active          |
| **Batch account**                      | The Batch service itself (just a control-plane resource)                               | $0                      | $0                              |
| **Batch pool**                         | A fleet of VMs that auto-scale from 0; each task = one solve                           | $0 when scaled to 0     | ~$0.30 (low-pri) – $1.60 (std) per 8 hr solve |
| **User-assigned Managed Identity** (`mi-tf-pool`) | The identity Batch nodes use to pull from ACR and access Blob                | $0                      | $0                              |

### Free / built-in services (still needed, but no separate cost line)

| Service                          | What it does for us                                                                            |
| -------------------------------- | ---------------------------------------------------------------------------------------------- |
| **Azure Resource Manager (ARM)** | The control plane that creates / lists / deletes every resource. Used by every `az` command.   |
| **Azure RBAC**                   | Role assignments (`Storage Blob Data Contributor`, `AcrPull`, etc.). No separate resource.     |
| **Azure Entra ID (AAD)**         | User and service principal identity. Your `az login` uses it. No cost for the use we make.     |
| **System-assigned Managed Identity** (on the ACA app) | Auto-created with the ACA app. The API uses it to talk to Blob without secrets. |
| **Azure Cost Management + Budgets** | Free dashboards + alert emails when you cross a $ threshold.                                |

---

## Provider registrations required

These have to be `Registered` on the subscription. One-time setup, free:

- `Microsoft.Storage` — for Blob
- `Microsoft.ContainerRegistry` — for ACR
- `Microsoft.App` — for Container Apps
- `Microsoft.OperationalInsights` — Container Apps dependency (even if logs disabled)
- `Microsoft.Batch` — for Batch account + pools
- `Microsoft.ManagedIdentity` — for the user-assigned MI on the Batch pool
- `Microsoft.Authorization` — for `az role assignment` to work

---

## RBAC roles required (by identity)

### Your user account
| Role                                 | Scope                | Why                                          |
| ------------------------------------ | -------------------- | -------------------------------------------- |
| `Reader`                             | Resource group       | See resources in portal / CLI                |
| `Storage Blob Data Contributor`     | Storage account      | Debug / inspect blobs                        |
| `AcrPush`                            | ACR                  | Push application images                      |
| `Container Apps Contributor`         | ACA app              | Deploy new revisions                         |
| `Azure Batch Account Contributor`    | Batch account        | Create/cancel tasks (when on Batch phase)    |

### ACA app's system-assigned Managed Identity
| Role                                 | Scope                | Why                                          |
| ------------------------------------ | -------------------- | -------------------------------------------- |
| `Storage Blob Data Contributor`     | Storage account      | API reads/writes blobs                       |
| `AcrPull`                            | ACR                  | ACA pulls API image on deploy/cold-start     |
| `Azure Batch Account Contributor`    | Batch account        | API creates Batch tasks (when on Batch phase)|

### Batch pool's user-assigned Managed Identity (`mi-tf-pool`)
| Role                                 | Scope                | Why                                          |
| ------------------------------------ | -------------------- | -------------------------------------------- |
| `Storage Blob Data Contributor`     | Storage account      | Compute nodes read inputs, write outputs/status |
| `AcrPull`                            | ACR                  | Nodes pull the Timefold image                |

**Total RBAC assignments:** 4 user + 3 ACA-MI + 2 Pool-MI = **9 assignments.**

---

## Total cost expectation

| Phase you're in       | Monthly idle | Per demo run | Notes                                                  |
| --------------------- | ------------ | ------------ | ------------------------------------------------------ |
| API + Blob only       | ~$5/mo       | ~$0.01       | Just ACR + (mostly idle) ACA. No compute layer.        |
| Full system + Batch   | ~$5/mo       | ~$0.30–$1.60 | Same idle (ACR + ACA = $5); add the per-solve VM cost. |

The recurring monthly cost is **dominated by ACR Basic ($5)** — every other service is genuinely free at idle. Per-run cost is dominated by **VM time during the solve**, with low-priority/spot pricing knocking ~80% off.

---

## What we explicitly do NOT need (and why)

If anyone asks "what about X?", here's the short answer:

| Service                          | Why we don't use it                                                                                      |
| -------------------------------- | -------------------------------------------------------------------------------------------------------- |
| ❌ Azure Service Bus              | We don't have bursty traffic; the API can call Batch directly. (See Azure.md Appendix A.)                |
| ❌ Azure Functions                | 10-min timeout doesn't cover an 8 hr solve. We use ACA + Batch instead.                                  |
| ❌ Azure App Service              | Always-on (more $), no scale-to-zero. ACA does the same job for free at idle.                            |
| ❌ Azure Kubernetes Service (AKS) | We have one HTTP service and one batch-of-containers — ACA + Batch handles both without K8s ops.         |
| ❌ Virtual Machines               | No need to manage OS / patching. ACA and Batch are managed compute.                                      |
| ❌ Azure Key Vault                | No secrets to store — all auth is via Managed Identity. Could add later if we introduce 3rd-party keys.   |
| ❌ Azure API Management           | Post-PoC concern; useful when you need auth/throttling/versioning at scale.                              |
| ❌ Azure Front Door / Traffic Mgr | Single-region deployment; no global routing needs.                                                       |
| ❌ Application Gateway / Load Balancer | ACA ingress handles HTTPS termination + load balancing for our scale.                                |
| ❌ Cosmos DB / SQL Database       | No relational data. status.json + blob enumeration covers our state needs.                               |
| ❌ Logic Apps / Power Automate    | The API Controller does the orchestration directly; no low-code workflow needed.                         |
| ❌ Azure Files / NetApp           | Blob is cheaper and faster for our pattern. No SMB/NFS needs.                                            |
| ❌ Azure Durable Functions        | Single-step pipeline; no fan-out/in or human-in-loop. Overkill. (See Azure.md Appendix C.)               |

---

## Quotas to know about

These can block you on a new subscription — check before assuming "just create it works":

| Quota                                         | Default on personal | Default on company           | How to fix                                                    |
| --------------------------------------------- | ------------------- | ---------------------------- | ------------------------------------------------------------- |
| Total Regional vCPUs                          | 10–20               | varies by company policy     | Portal → Quotas → request increase                            |
| Batch — vCPU per VM family (e.g. Fsv2)        | often 0             | varies                       | Portal → Quotas → Compute → Batch tab → request increase      |
| Batch — Total Low-priority vCPUs              | often 0             | varies                       | Same place                                                    |
| Container Apps environments per region        | 5                   | usually fine                 | Almost never hit                                              |
| ACR registries per subscription               | 100                 | fine                         | Won't hit                                                     |
| Storage accounts per subscription             | 250                 | fine                         | Won't hit                                                     |

**Quota requests for Batch vCPUs typically take 1–48 hours.** Submit them
early in the project so they're approved by the time you need them.

---

## Visual map (who talks to whom)

```
                    ┌──────────┐
                    │  User    │
                    │ (browser)│
                    └────┬─────┘
                         │ HTTPS
                         ▼
                    ┌─────────────────────────┐
                    │  ACA HTTP app           │
                    │  (API Controller)       │── 1. upload input blob
                    │  System-assigned MI ────┼─→ Azure Blob Storage
                    │                         │── 2. create Batch task
                    │                         │─→ Azure Batch
                    │                         │── 4. read status.json
                    │                         │── 5. generate SAS URL
                    └────────┬────────────────┘
                             │
                ┌────────────▼────────────────┐
                │  ACR (Container Registry)   │
                │  • api-controller:v1        │ ← ACA pulls (AcrPull on MI)
                │  • timefold:v1              │ ← Batch pulls (AcrPull on pool MI)
                └─────────────────────────────┘

       ┌──────────────────────────────────┐
       │  Azure Batch                      │
       │  • account                        │
       │  • pool (autoscale 0→N nodes)     │── 3a. nodes pull timefold:v1 from ACR
       │  • user-assigned MI (mi-tf-pool)  │── 3b. read input from Blob
       │                                   │── 3c. write output + status to Blob
       └──────────────────────────────────┘

       ┌──────────────────────────────────┐
       │  Azure Blob Storage               │
       │  └─ container "timefold"          │
       │       ├─ input/{runId}/           │
       │       ├─ output/{runId}/          │
       │       └─ status/{runId}.json      │
       └──────────────────────────────────┘

       ┌──────────────────────────────────┐
       │  User                             │
       │  (browser, with SAS URL from API) │── 6. download result yaml from Blob
       └──────────────────────────────────┘
```

The arrows labelled 1-6 are the full request lifecycle. Note that **every
service-to-service call uses Managed Identity** — no passwords, no shared
keys, no certificates managed by humans.

---

## One-line summary

> **Blob + ACR + ACA + Batch.** 4 named services, ~$5/month at rest, ~$1
> per 8 hr solve. Everything else (RBAC, ARM, Entra, Managed Identities,
> Cost Management) is free and built in.
