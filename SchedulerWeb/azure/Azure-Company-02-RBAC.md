# Company Phase 2 — Confirm or assign permissions (RBAC)

**Goal of this phase:** every identity that needs to touch a resource
(you, the API Controller, the Batch pool) actually has permission to do so.
Most of these are probably already set up by the admin — this phase is
mostly **checking**, and only **assigning** the ones that are missing.

**Time:** ~20–30 minutes.
**Prereqs:** Phase 1 done — `~/azure-timefold-company-env.sh` has every real
name filled in.

---

## `Contributor` is NOT enough — read this before you check anything

If the admin gave you a general **`Contributor`** role instead of the
specific roles below, you will still fail on Storage and ACR. This is not a
misconfiguration — it's how Azure is designed:

- **Actions** (management plane — create/configure/delete the *resource
  itself*) are covered by `Contributor`. This includes creating/resizing
  Batch pools, submitting Batch tasks, and deploying Container Apps
  revisions — all of that works fine on `Contributor` alone.
- **DataActions** (data plane — touching the *data inside* the resource,
  e.g. reading/writing an actual blob, or `docker push`/`pull`) are
  **deliberately excluded** from `Contributor`. Storage and ACR both split
  their permissions this way on purpose, as a separate, revocable layer.
  `Contributor` — even `Owner` — does not imply data access.

So with only `Contributor` you can create the storage account but not
upload a blob to it, and you can create the ACR but not push an image to
it. You still need the specific roles in 2.1–2.3 below
(`Storage Blob Data Contributor`, `AcrPush`, `AcrPull`) — ask the admin to
add those explicitly; `Contributor` cannot be "used differently" to cover
them.

One more consequence: `Contributor` also excludes
`Microsoft.Authorization/roleAssignments/write`, so if that's all you have,
you **cannot assign these roles to yourself either** — only `Owner` or
`User Access Administrator` can. If your portal "Add role assignment"
button is greyed out or errors with "you do not have permissions," that's
why — ask the admin to add the specific roles for you.

---

## Why portal, not CLI

During the personal-account dry run, `az role assignment create` broke in a
way that survived every documented fix (see
[Azure-03-2-RoleAssignment-Workaround.md](./Azure-03-2-RoleAssignment-Workaround.md)).
**Do every check and every assignment in this phase through the Azure
Portal.** It's ~5 clicks and a couple minutes per role — slower per-click
than a CLI command that works, but zero risk of losing time to a CLI cache
bug on an unfamiliar company machine.

---

## The pattern you'll repeat for every row below

**To check if a role is already assigned:**
1. Open the resource (the one in the "Scope" column).
2. Left sidebar → **Access control (IAM)**.
3. Tab → **Role assignments**.
4. Use the search box — type the role name, or the identity's name/email —
   and see if a matching row appears.

**To assign a role that's missing:**
1. Same **Access control (IAM)** page → **+ Add** → **Add role assignment**.
2. **Role** tab → type the exact role name from the table → click it → **Next**.
3. **Members** tab:
   - Assigning to **yourself**: **Assign access to** = `User, group, or service principal` → **+ Select members** → search your email → click it → **Select**.
   - Assigning to the **ACA app's identity**: **Assign access to** = `Managed identity` → **+ Select members** → **Managed identity** dropdown = `Container app` → pick `$ACA_APP` → **Select**.
   - Assigning to the **Batch pool's identity**: **Assign access to** = `Managed identity` → **+ Select members** → **Managed identity** dropdown = `User-assigned managed identity` → pick `$MI_NAME` → **Select**.
4. **Review + assign** → **Review + assign**.
5. Wait ~30 seconds, then go back to **Role assignments** and confirm the row now appears. Always verify in the portal, not CLI.

---

## 2.1 — Your own account (5 roles)

Source your env script first so you have the real names to search for:
```bash
source ~/azure-timefold-company-env.sh
echo "RG=$RG  ST=$ST  ACR=$ACR  ACA_APP=$ACA_APP  BATCH=$BATCH"
```

| # | Role                                | Open this resource...           | Scope (what you'll pick in "Add role assignment") | Why you need it |
| - | ------------------------------------ | -------------------------------- | ---------------------------------------------------- | ----------------- |
| 1 | **Reader**                           | Resource group `$RG`             | The resource group itself                            | See resources in portal / CLI |
| 2 | **Storage Blob Data Contributor**    | Storage account `$ST`            | The storage account                                   | Upload/download/inspect blobs for debugging |
| 3 | **AcrPush**                          | Container registry `$ACR`        | The registry                                          | Push Docker images (Phase 4, 6) |
| 4 | **Container Apps Contributor**       | Container App `$ACA_APP`         | The ACA app                                           | Deploy new revisions (Phase 6) |
| 5 | **Azure Batch Account Contributor**  | Batch account `$BATCH`           | The Batch account                                     | Create/inspect/cancel tasks (Phase 5) |

Check all 5 first. For any missing, assign using the pattern above ("Assigning to yourself").

You do **not** need `Owner` or `User Access Administrator` — if you don't
have them, that's correct, not a gap.

## 2.2 — ACA app's system-assigned Managed Identity (3 roles)

This identity lives *inside* `$ACA_APP` itself. If Phase 1 found it turned
off, turn it on first:

1. Portal → Container App `$ACA_APP` → left sidebar **Identity**.
2. **System assigned** tab → **Status** = **On** → **Save** (skip this if it's already On).

| # | Role                                | Open this resource...     | Why the API Controller needs it |
| - | ------------------------------------ | --------------------------- | ---------------------------------- |
| 1 | **Storage Blob Data Contributor**    | Storage account `$ST`       | Reads/writes input, output, status blobs |
| 2 | **AcrPull**                          | Container registry `$ACR`   | Pulls the API Controller's own image on deploy/cold-start |
| 3 | **Azure Batch Account Contributor**  | Batch account `$BATCH`      | Creates/terminates Batch tasks when a run is submitted |

Assign using "Assigning to the ACA app's identity" from the pattern above.

## 2.3 — Batch pool's user-assigned Managed Identity (2 roles)

This is `$MI_NAME` from Phase 1.

| # | Role                                | Open this resource...     | Why the compute node needs it |
| - | ------------------------------------ | --------------------------- | -------------------------------- |
| 1 | **Storage Blob Data Contributor**    | Storage account `$ST`       | Compute nodes read input YAMLs, write output YAML + status.json |
| 2 | **AcrPull**                          | Container registry `$ACR`   | Compute nodes pull the Timefold solver image |

Assign using "Assigning to the Batch pool's identity" from the pattern above.

---

## Record what you found

Keep a private note (not in this shared doc) of which of the 10 were
already assigned vs. which you assigned yourself — useful if something
fails later and you need to know whether a permission is the suspect.

| Section | Confirmed already assigned | Assigned by you today |
| ------- | --------------------------- | ----------------------- |
| 2.1 (5 roles)  |   |   |
| 2.2 (3 roles)  |   |   |
| 2.3 (2 roles)  |   |   |

---

## What you should have at the end of this phase

- [ ] All 5 roles in 2.1 confirmed present on your account
- [ ] ACA app's system-assigned identity is On, all 3 roles in 2.2 confirmed present
- [ ] All 2 roles in 2.3 confirmed present on `$MI_NAME`
- [ ] Every check was done via **Access control (IAM) → Role assignments**, not CLI

Next: [Azure-Company-03-Storage-Verify.md](./Azure-Company-03-Storage-Verify.md)
— prove your own Blob access actually works end-to-end.

---

## Troubleshooting

| Symptom                                                        | Cause                                            | Fix |
| ---------------------------------------------------------------- | -------------------------------------------------- | ----- |
| Search box in "Add role assignment" finds no matching role name | Typo, or you're on the wrong resource type's IAM page | Re-check exact spelling against the tables above; role lists differ by resource type |
| "You do not have permissions to add role assignment"             | You need `Owner` / `User Access Administrator` on that scope, which you likely don't have | Ask the admin to assign it, or to grant you temporary elevated access to do it yourself |
| Managed identity doesn't show up when searching in "Select members" | Wrong identity type selected in the dropdown       | ACA app's identity = `Container app` type; Batch pool's identity = `User-assigned managed identity` type — don't mix them up |
| Role assignment appears immediately in the Add flow but a later `az storage` command still fails with `AuthorizationPermissionMismatch` | Propagation delay (30–60s)                         | Wait a minute and retry; don't re-assign, that just creates a duplicate |
