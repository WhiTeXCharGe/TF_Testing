# Azure Company Setup — On-Site Checklist

> **Looking for the full guided walkthrough instead of a checklist?**
> Use **[Azure-Company-01-Access-And-Resources.md](./Azure-Company-01-Access-And-Resources.md)**
> through **[Azure-Company-07-Webapp-Connect.md](./Azure-Company-07-Webapp-Connect.md)**
> — a step-by-step series that takes you all the way from first sign-in to
> the webapp running against the real Azure API, written so someone with no
> prior Azure experience can follow it verbatim. This checklist is the
> fast, at-a-glance companion to that series — use it to track status once
> you've already been through the guided docs once, or as a quick reference
> mid-session.

**Purpose:** a checklist to bring to the other PC for the real work on the
**company** Azure portal/subscription. The company's Azure admin creates the
resources — you did **not** name them, so this doc never guesses a real
name. Every resource name below is a **placeholder** in `<angle-brackets>`.
The first thing you do for each resource is find it in the portal and write
its real name into the blank next to the placeholder. After that, every
later reference to that resource in this doc means "the name you wrote in
that blank," not the placeholder text itself.

**Do not write the resource group name into this file.** Use
`<resource-group>` everywhere; fill the real value only in your own head /
a private notes app if you need it, not here, since this doc may be shared.

Related background (already done, on your **personal** account, different
names, kept only for reference — do not copy names from these into company
work):
- [Azure-01-Account-Setup.md](./Azure-01-Account-Setup.md) → [Azure-07-Webapp-To-Azure.md](./Azure-07-Webapp-To-Azure.md) — personal dry run, phases 1–7
- [Azure-Company-Permission-Request.md](./Azure-Company-Permission-Request.md) — the request you sent asking the company to create these resources (with *proposed* names — the company may not have used them)
- [Azure-Products-Required.md](./Azure-Products-Required.md) — plain reference for what each Azure product is for

---

## 0. Placeholder legend

Fill these in as you find each thing. Keep this table as your single
source of truth — every section below refers back to these names.

| Placeholder                    | What it is                                              | Actual value (fill in on-site) |
| ------------------------------- | -------------------------------------------------------- | ------------------------------- |
| `<resource-group>`              | The resource group holding all Timefold resources        |                                  |
| `<region>`                      | Azure region the resources were created in                |                                  |
| `<subscription-id>`             | Company subscription GUID                                 |                                  |
| `<your-account>`                | The login you were given (email / UPN)                    |                                  |
| `<storage-account-name>`        | Storage account (Blob)                                    |                                  |
| `<blob-container-name>`         | Container inside the storage account                      |                                  |
| `<acr-name>`                    | Container Registry name (login server = `<acr-name>.azurecr.io`) |                          |
| `<aca-environment-name>`        | Container Apps environment                                 |                                  |
| `<aca-app-name>`                | Container Apps HTTP app (the API Controller host)          |                                  |
| `<aca-app-url>`                 | Public HTTPS URL of the ACA app                            |                                  |
| `<batch-account-name>`          | Batch account                                              |                                  |
| `<batch-pool-id>`               | Batch pool inside the Batch account                        |                                  |
| `<batch-job-id>`                | Standing job inside the Batch account (may not exist yet)  |                                  |
| `<user-assigned-identity-name>` | User-assigned Managed Identity attached to the Batch pool  |                                  |

---

## 1. Status board (update this as you go)

Nothing is done yet — everything starts at "Not started." Update the
Status column live during the session so you always know where you are if
you have to stop and resume.

| # | Item                                                   | Status        |
| - | ------------------------------------------------------- | ------------- |
| 1 | Sign in to company account, confirm subscription/tenant  | Not started   |
| 2 | Locate resource group, confirm access                    | Not started   |
| 3 | Subscription-level resource providers registered         | Not started   |
| 4 | Batch vCPU quota available                                | Not started   |
| 5 | Storage account + container located and verified         | Not started   |
| 6 | Container Registry located and verified                  | Not started   |
| 7 | Container Apps environment + app located and verified    | Not started   |
| 8 | Batch account + pool located and verified                 | Not started   |
| 9 | User-assigned Managed Identity located and verified        | Not started   |
| 10| RBAC — your user account (5 roles)                        | Not started   |
| 11| RBAC — ACA app's system-assigned identity (3 roles)        | Not started   |
| 12| RBAC — Batch pool's user-assigned identity (2 roles)        | Not started   |
| 13| Section 6 "information sheet" filled in completely         | Not started   |
| 14| Local `.env` / config files updated with real values        | Not started   |

Status values to use: `Not started` → `In progress` → `Found, name recorded`
→ `Verified` → `Blocked (see notes)`.

---

## 2. Step 0 — Sign in and orient yourself

1. Go to https://portal.azure.com and sign in with `<your-account>`.
2. Top-right avatar → confirm you're in the **company** tenant/subscription,
   not your personal one. Note the subscription name.
3. Portal search → `subscriptions` → click the company subscription → copy
   the **Subscription ID** into the placeholder table above.
4. Portal search → `resource groups` → look for the group the company set
   up for this project (ask whoever provisioned it if it's not obvious from
   the list — there may be only one you have access to, which makes this
   easy). Click into it.
5. Inside that resource group you should see a list of resources — this is
   your master list for section 4 below. Screenshot or note the resource
   **types** and **names** now; you'll use them for every section that
   follows.

If you only have Reader-level access and can't see the resource group list,
ask the admin who created the resources to send you the resource group name
directly (out of band — chat/email, not needed in this file).

### Optional — Azure CLI login (if you'll use CLI at all)

```bash
az login
az account show --output table
az account set --subscription "<subscription-id>"
```

Per [Azure-03-2-RoleAssignment-Workaround.md](./Azure-03-2-RoleAssignment-Workaround.md),
`az role assignment` commands were unreliable during the personal dry run.
**Plan to do all RBAC checks and assignments in the portal**, not CLI,
unless you've confirmed CLI works cleanly on this machine/account first.

---

## 3. Subscription-level checks (no resource names needed)

These aren't scoped to a specific resource, so there's nothing to look up —
just run/check them.

### 3a. Resource provider registration

```bash
az provider list --query "[?contains(['Microsoft.Storage','Microsoft.Authorization','Microsoft.App','Microsoft.OperationalInsights','Microsoft.ContainerRegistry','Microsoft.ManagedIdentity','Microsoft.Batch'], namespace)].{name:namespace, state:registrationState}" -o table
```

All 7 should read `Registered`. If any say `NotRegistered` and you get a
permission error trying to register them yourself:

```bash
az provider register --namespace Microsoft.<X>
```

— ask the admin to register that namespace; company subscriptions often
restrict this to admins.

Portal alternative: search `Subscriptions` → your subscription → left
sidebar **Resource providers** → search each `Microsoft.X` name → status
column → **Register** if needed.

### 3b. Batch vCPU quota

```bash
az batch location quotas show --location <region> -o table
```

Look for `DedicatedCoreQuotaPerVMFamily` or `LowPriorityCoreQuota` > 0 for
the VM family the pool uses. If both are 0, the pool can never scale above
zero nodes even if everything else is correct.

Portal alternative: search **Quotas** → **Compute** → filter by subscription
+ `<region>` → find the Batch rows.

If quota is 0, this needs an increase request (Portal → Quotas → pencil
icon → request) — ask the admin if you don't have permission to request it
yourself on the company subscription.

---

## 4. Per-resource verification

For each resource: find it, record its name in section 0, sanity-check the
key settings, then jump to section 5 for the matching RBAC check. Nothing
here needs you to create anything — these are all "should already exist"
checks. A **fallback create-it-yourself** procedure is at the very end
(section 7) in case something is genuinely missing.

### 4.1 Storage account + Blob container

**Find it:** Resource group → look for a resource of type **Storage
account**. Click in → left sidebar **Containers** → note the container name.

Record into section 0: `<storage-account-name>`, `<blob-container-name>`.

Quick checks (Overview / Configuration blades):
- [ ] Kind = `StorageV2`, Replication = some `LRS`/`ZRS`/`GRS` variant (any is fine functionally)
- [ ] Container `<blob-container-name>` exists and is **Private** (no anonymous access)
- [ ] "Allow Blob public access" = Disabled (Configuration blade)

You do **not** need to create this — only confirm it exists and you can see
it. RBAC (data-plane access) is a separate step — see 5.1.

### 4.2 Azure Container Registry (ACR)

**Find it:** Resource group → resource of type **Container registry**.

Record into section 0: `<acr-name>` (login server shown on the Overview
blade as `<acr-name>.azurecr.io`).

Quick checks:
- [ ] SKU is Basic/Standard/Premium (note which — affects storage limits, not function)
- [ ] Repositories blade — if the company pre-loaded images, you'll see
      `api-controller` and/or `timefold` repos here already. If empty,
      that's expected — you'll push images yourself later.

### 4.3 Container Apps environment + app

**Find it:** Resource group → resource of type **Container Apps
Environment**, and a separate resource of type **Container App**.

Record into section 0: `<aca-environment-name>`, `<aca-app-name>`,
`<aca-app-url>` (Overview blade → Application Url).

Quick checks:
- [ ] The Container App's Overview shows a public HTTPS URL you can open
- [ ] Left sidebar → **Identity** → System assigned → **On** (needed for RBAC in 5.2). If it's Off, you'll need to turn it on yourself — see the note below.

If System-assigned identity is **Off**:
1. Left sidebar → **Identity** → **System assigned** tab → toggle **Status = On** → **Save**.
2. This is a config change, not a "create resource" — safe to do yourself.

### 4.4 Batch account + pool

**Find it:** Resource group → resource of type **Batch account**. Inside
it, left sidebar → **Pools**.

Record into section 0: `<batch-account-name>`, `<batch-pool-id>`. If a
**Jobs** blade shows an existing job, record `<batch-job-id>` too — if none
exists, you (or the API Controller code) will create one later; note that
as a gap, not a blocker.

Quick checks:
- [ ] Pool exists, state is `Active`/`Steady`
- [ ] Pool's node count can be 0 (that's expected/healthy when idle)
- [ ] Pool → **Identity** — should list a **User-assigned** managed identity attached (this is `<user-assigned-identity-name>`, see 4.5)

### 4.5 User-assigned Managed Identity (Batch pool's identity)

**Find it:** Two ways —
- Resource group → resource of type **Managed Identity**, OR
- Batch account → Pool → **Identity** blade → the user-assigned identity listed there → click through to its resource page.

Record into section 0: `<user-assigned-identity-name>`.

On its Overview blade, note the **Object (principal) ID** — you'll need
this exact value if you ever do RBAC assignment via CLI (portal doesn't
need it, it looks the identity up by name).

---

## 5. RBAC — confirm or assign

For every row below: **first check if it's already assigned** (most likely,
since the company set this up); **only assign it yourself if it's
missing**. Portal steps for both are identical except for the final button.

### How to check what's assigned (do this for every resource below)

1. Open the resource (storage account / ACR / ACA app / Batch account).
2. Left sidebar → **Access control (IAM)**.
3. Tab → **Role assignments**.
4. Use the search box to filter by the role name, or by `<your-account>` /
   the identity name, and confirm the row exists.

### How to assign a missing role (portal — do not fight the CLI, see Azure-03-2)

1. On the same **Access control (IAM)** page → **+ Add** → **Add role assignment**.
2. **Role** tab → search the exact role name below → select → **Next**.
3. **Members** tab:
   - For your own account: **Assign access to** = `User, group, or service principal` → **+ Select members** → search `<your-account>` → select.
   - For the ACA app or Batch pool identity: **Assign access to** = `Managed identity` → **+ Select members** → pick the identity type (Container app / User-assigned) → pick `<aca-app-name>` or `<user-assigned-identity-name>`.
4. **Review + assign** → **Review + assign**.
5. Wait ~30 seconds, then re-check via the Role assignments tab (not CLI).

### 5.1 Your user account (`<your-account>`)

| Role                                | Scope                                | Status |
| ------------------------------------ | ------------------------------------- | ------ |
| **Reader**                           | `<resource-group>`                    | ☐ checked |
| **Storage Blob Data Contributor**    | `<storage-account-name>`              | ☐ checked |
| **AcrPush**                          | `<acr-name>`                          | ☐ checked |
| **Container Apps Contributor**       | `<aca-app-name>`                      | ☐ checked |
| **Azure Batch Account Contributor**  | `<batch-account-name>`                | ☐ checked |

`Owner` / `User Access Administrator` are not required — don't ask for them
and don't worry if you don't have them.

### 5.2 ACA app's system-assigned Managed Identity

(This is the identity that lives on `<aca-app-name>` itself — see 4.3 for
turning it on if it's currently off.)

| Role                                | Scope                                | Status |
| ------------------------------------ | ------------------------------------- | ------ |
| **Storage Blob Data Contributor**    | `<storage-account-name>`              | ☐ checked |
| **AcrPull**                          | `<acr-name>`                          | ☐ checked |
| **Azure Batch Account Contributor**  | `<batch-account-name>`                | ☐ checked |

### 5.3 Batch pool's user-assigned Managed Identity (`<user-assigned-identity-name>`)

| Role                                | Scope                                | Status |
| ------------------------------------ | ------------------------------------- | ------ |
| **Storage Blob Data Contributor**    | `<storage-account-name>`              | ☐ checked |
| **AcrPull**                          | `<acr-name>`                          | ☐ checked |

**Total: 10 role assignments across 3 identities.** If you find more or
fewer already in place than listed, that's fine — this table is the
*minimum needed for the app to run*; extra roles someone else added aren't
a problem, just note anything missing.

---

## 6. Information sheet (fill in and keep for the deployment step)

Once sections 0–5 are all checked, this is what you'll actually paste into
config files / `.env` files back on your dev machine. It's the same list as
section 5 of [Azure-Company-Permission-Request.md](./Azure-Company-Permission-Request.md#5-information-required-after-provisioning),
just as a blank form now that the real values exist.

1. Subscription ID: `____________________________`
2. Resource group name: `____________________________` *(write this only in your private notes, not shared copies of this doc)*
3. Storage account name: `____________________________`
4. Blob container name: `____________________________`
5. ACR name + login server: `____________________________` / `____________________________`
6. ACA environment name: `____________________________`
7. ACA app name: `____________________________`
8. ACA app public URL: `____________________________`
9. Batch account name + URL: `____________________________` / `____________________________`
10. Batch pool ID: `____________________________`
11. Batch job ID (if one exists): `____________________________`
12. User-assigned MI name + resource ID: `____________________________` / `____________________________`
13. Region used: `____________________________`
14. All 5 roles in section 5.1 confirmed: ☐
15. All 3 roles in section 5.2 confirmed: ☐
16. Both roles in section 5.3 confirmed: ☐
17. Batch vCPU quota confirmed > 0: ☐

---

## 7. Fallback — if something is genuinely missing

Only use this if section 4 turned up a **missing** resource, not just an
unfamiliar name. Ask the admin first — it may exist under a name/location
you haven't found yet, or under someone else's access. If it's confirmed
missing and you've been asked to create it yourself, the exact CLI commands
already exist in the personal dry-run phase docs — reuse the command, swap
in a name that fits the **company's naming convention** (ask what that is
rather than inventing one), and create it inside `<resource-group>`, not a
new group:

| Missing resource              | Reference for the create command |
| ------------------------------ | --------------------------------- |
| Storage account + container    | [Azure-02-Storage-Blob.md](./Azure-02-Storage-Blob.md) Steps 2–4 |
| ACR                             | [Azure-04-Container-Registry.md](./Azure-04-Container-Registry.md) Step 1 |
| ACA environment + app           | [Azure-03-Container-Apps.md](./Azure-03-Container-Apps.md) Steps 1–2 |
| Batch account + pool            | [Azure-05-Batch-First-Run.md](./Azure-05-Batch-First-Run.md) Steps 3–4 |
| User-assigned Managed Identity  | [Azure-05-Batch-First-Run.md](./Azure-05-Batch-First-Run.md) Step 1 |

After creating anything this way, go back to section 5 and assign its RBAC
roles — a newly created resource has none by default.
