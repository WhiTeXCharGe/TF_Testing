# Phase 2.2 — Fixes & Provider Registration (read once, save the day later)

This doc captures the two errors we hit during Phase 2 setup and the
permanent fixes. **Run the "do this once per new subscription" section
right after Phase 1** on any future subscription (yours, your boss's, the
company's) and you'll never see either error again.

---

## The two errors in plain English

### Error A — `SubscriptionNotFound` when creating any Azure resource

```
ERROR: (SubscriptionNotFound) Subscription <guid> was not found.
```

**Misleading name.** Your subscription is fine. The actual problem: each
Azure *service* (Storage, Container Apps, Container Registry, Batch, etc.)
is a "Resource Provider" that has to be **explicitly turned on** in your
subscription before you can create resources of that type. Pay-as-you-go
and Free Trial accounts often start with most providers **NotRegistered**.

### Error B — `MissingSubscription` when running `az role assignment create`

```
ERROR: (MissingSubscription) The request did not have a subscription or a
valid tenant level resource provider.
```

Same root cause, different service. The role-assignment API lives in the
`Microsoft.Authorization` resource provider — also has to be registered.

---

## Do this ONCE per new subscription (5 minutes)

Right after Phase 1, before touching anything else, register every provider
you'll need across the whole project. Each call returns immediately; the
actual registration happens in the background and takes 1–3 minutes per
provider.

```bash
# Storage (Blob, queues, tables, files)
az provider register --namespace Microsoft.Storage

# Role assignments / RBAC
az provider register --namespace Microsoft.Authorization

# Container Apps + its required Log Analytics dependency
az provider register --namespace Microsoft.App
az provider register --namespace Microsoft.OperationalInsights

# Container Registry
az provider register --namespace Microsoft.ContainerRegistry

# Azure Batch (for the Timefold compute later)
az provider register --namespace Microsoft.Batch

# Managed Identities (used by ACA + Batch pool)
az provider register --namespace Microsoft.ManagedIdentity
```

Then wait for the two we need immediately to finish registering:

```bash
for ns in Microsoft.Storage Microsoft.Authorization; do
  until [ "$(az provider show --namespace $ns --query registrationState -o tsv)" = "Registered" ]; do
    echo "waiting for $ns..."; sleep 5
  done
  echo "$ns is Registered."
done
```

(The others — Microsoft.App, Microsoft.Batch, etc. — finish in the
background; we'll wait for them when we actually need them in Phase 3+.)

### Sanity check: confirm everything's registered

```bash
az provider list \
  --query "[?contains(['Microsoft.Storage','Microsoft.Authorization','Microsoft.App','Microsoft.OperationalInsights','Microsoft.ContainerRegistry','Microsoft.Batch','Microsoft.ManagedIdentity'], namespace)].{name:namespace, state:registrationState}" \
  -o table
```

Should show all 7 as `Registered` (or `Registering` for the ones still finishing).

---

## RBAC role assignment — gotchas and the portal fallback

### Gotcha 1: empty Bash variables
When you open a new Git Bash window, ALL variables (`$ST`, `$ST_ID`,
`$USER_OID`, etc.) are wiped. If you re-run a CLI command that uses them
and gets `MissingSubscription` or weird "name cannot be empty" errors,
**always check the variables first** before assuming a permissions problem:

```bash
echo "RG=[$RG]  ST=[$ST]  USER_OID=[$USER_OID]  ST_ID=[$ST_ID]"
```

If any brackets are empty, re-run the variable-setting commands.

### Gotcha 2: principal-type race
Even with everything set, `az role assignment create --assignee <oid>`
sometimes fails on fresh accounts because the AAD lookup of "is this OID
a User, Group, or ServicePrincipal?" hasn't propagated yet. Force the
classification:

```bash
az role assignment create \
  --assignee-object-id "$USER_OID" \
  --assignee-principal-type User \
  --role "Storage Blob Data Contributor" \
  --scope "$ST_ID"
```

### Gotcha 3: when CLI just won't, use the portal once
RBAC assignment is a one-time setup. If the CLI keeps fighting you for any
reason, do it in the portal — same effect, same audit trail:

1. https://portal.azure.com → search **Storage accounts** → click your account
2. Left sidebar → **Access control (IAM)**
3. Top bar → **+ Add** → **Add role assignment**
4. **Role** tab → search `Storage Blob Data Contributor` → select → Next
5. **Members** tab:
   - Assign access to: `User, group, or service principal`
   - Click **+ Select members** → type your email → click your entry → **Select**
6. **Review + assign** → **Review + assign**
7. Wait ~30 s for propagation, then in Git Bash:
   ```bash
   az storage container list --account-name $ST --auth-mode login -o table
   ```
   No error = role is working.

---

## Keep variables across sessions — the env script

Bash variables die with the terminal. Save them in a tiny sourceable script
and you'll never re-type them again:

```bash
cat > ~/azure-timefold-env.sh <<'EOF'
# Source this at the start of any Azure session:
#   source ~/azure-timefold-env.sh
export RG=rg-timefold-dev
export LOC=eastus
export ST=sttimefolddevseiya        # <-- edit if you used a different name
export CONTAINER=timefold

# Re-fetched fresh each session (handles token rotation cleanly)
export USER_OID=$(az ad signed-in-user show --query id -o tsv)
export ST_ID=$(az storage account show --name $ST --resource-group $RG --query id -o tsv 2>/dev/null)

echo "Loaded RG=$RG  ST=$ST"
[ -z "$ST_ID" ] && echo "WARN: ST_ID empty — storage account not found yet (OK for Phase 1)."
EOF
```

Then any time you open a new terminal:
```bash
source ~/azure-timefold-env.sh
```

> The 2>/dev/null on the ST_ID line silently tolerates "storage doesn't
> exist yet" so you can also source this script during Phase 1 before the
> storage account is created.

When we add more resources (ACA app, Batch pool, ACR), we'll add their IDs
to this same script. It'll grow into a small per-session bootstrap.

---

## What this saves you later

When you bring on the company subscription:

1. Brand-new subscription = same providers NotRegistered.
2. Run the registration block from this doc once.
3. The Phase 2 commands then work first try.

When you write the company access-request doc (Phase 8), this list of
providers is **exactly what you'll ask the company's Azure admin to enable**
on the prod subscription. You're already documenting it.

---

## TL;DR

| Problem                                                | Fix                                                                       |
| ------------------------------------------------------ | ------------------------------------------------------------------------- |
| `SubscriptionNotFound` on any `az <service> create`    | Register the provider for that service (`az provider register --namespace Microsoft.<X>`) |
| `MissingSubscription` on role assignment               | Register `Microsoft.Authorization`; use `--assignee-object-id` + `--assignee-principal-type` |
| Variables empty in new terminal                        | `source ~/azure-timefold-env.sh`                                          |
| CLI keeps fighting RBAC                                | Do it once in the portal (Access control → Add role assignment)           |
