# Phase 3.2 — Role assignment via portal (CLI cache workaround)

This doc captures the role-assignment error that survived even the Phase 2.2
fixes, and the **permanent operating workaround**: do role assignments in the
portal, not the CLI, for the rest of the project.

> **Read Phase 2.2 first** if you haven't — that covers the simpler cases
> (empty variables, provider not registered, principal-type race). This doc
> covers the deeper case where those fixes don't help.

---

## The symptom that's beyond Phase 2.2

After registering `Microsoft.Authorization`, setting `--assignee-principal-type`
explicitly, AND passing `--subscription` on the command, you still get:

```
ERROR: (MissingSubscription) The request did not have a subscription or a
valid tenant level resource provider.
```

Diagnostic that confirms the deeper cause:
```bash
az account show --query "{name:name, id:id}" -o table
```
If this prints `Name` only (no `Id` column / Id is null), your CLI's local
account cache has lost the subscription id binding. `az role assignment`
specifically depends on that cache; other commands (`az storage`, `az group`,
`az containerapp`) work because they get the subscription from the resource
ID in `--scope` or `--resource-group`.

---

## The workaround we adopted — portal for all role assignments

Role assignment in the portal is 5 clicks and ~2 minutes. **Do this for every
role assignment in this project from now on**:

1. Open the resource that's receiving access (storage account, ACR, Batch
   account — whoever owns the resource you're granting access TO).
2. Left sidebar → **Access control (IAM)**
3. Top → **+ Add** → **Add role assignment**
4. **Role** tab → search the role name → select → **Next**
5. **Members** tab:
   - **Assign access to:**
     - **User, group, or service principal** for your own account
     - **Managed identity** for ACA app, Batch pool, etc.
   - **+ Select members** → pick the right subscription and the right identity
     - For Managed identities the dropdown asks for the identity *type*
       (Container app, User-assigned, Batch account, etc.) — pick the right one
6. **Review + assign** → **Review + assign**
7. Wait ~30 s for propagation, then **verify via portal** (NOT CLI):
   - Go back to Access control (IAM) → **Role assignments** tab
   - Find the role → expand → both members should be listed

That's it. Identical end-state to CLI; Azure doesn't track which client did it.

---

## Why we accept this trade-off

- **Role assignment is a one-time-per-resource operation.** This project will
  have ~5–8 role assignments total across all phases. 5 portal clicks × 8
  assignments = 40 clicks vs hours of CLI debugging. The math is clear.
- **Portal works on every machine.** No CLI cache state, no token rot, no
  account-id-null surprises. The company subscription will probably have
  similar quirks on first use; portal sidesteps them all.
- **The audit trail is identical.** `Activity log` and `Access control (IAM)`
  show the assignment regardless of which client created it.

---

## When (and how) to fix the CLI later

You don't need this for the project to work. But when you have an idle 10
minutes and want a clean CLI:

```bash
# Full reset — wipes the Azure CLI's local cache
az logout
az account clear

# On Windows / Git Bash:
rm -rf ~/.azure
# (this folder holds the cached token + account info; deleting forces a fresh login)

# Fresh login
az login
az account set --subscription "<your-subscription-id>"

# The smoke test
az account show --query "{name:name, id:id, tenantId:tenantId}" -o table
```

All three columns (Name, Id, TenantId) should print real values. If `Id` is
still null after the reset, your AAD account → subscription link is broken
at the Azure level — only Microsoft Support can fix it, and it has no impact
on day-to-day work.

After a successful reset, retry your last failing role-assignment command in
CLI to confirm it works. If it does, you can switch back to CLI for future
assignments; if not, just stay with the portal.

---

## TL;DR

| Question                                          | Answer                                                            |
| ------------------------------------------------- | ----------------------------------------------------------------- |
| CLI role assignment still fails after Phase 2.2 fixes? | Use the portal. Same result, 2 minutes.                       |
| How do I verify a role is assigned?               | Portal → Storage account → Access control (IAM) → Role assignments tab. Both your user and any MI should be listed under the role. |
| Should I fix the CLI right now?                   | No. Use portal for the rest of the project. Fix CLI at leisure with the cache reset above. |
| Will this hurt the company subscription setup later? | No. The portal workaround works identically there. Document it in the company access doc (Phase 8) and move on. |
