# Company Phase 3 — Prove Blob Storage works for you

**Goal of this phase:** upload a real file to the company's storage account,
read it back, and generate a link that a browser can open. If all three
work, you know your account's Blob permission (Phase 2) is correctly wired
before you build anything that depends on it.

**Time:** ~15 minutes.
**Prereqs:** Phase 1 + Phase 2 done. `~/azure-timefold-company-env.sh`
sourced, and role #2 in section 2.1 (Storage Blob Data Contributor)
confirmed.

---

## Step 1 — Source your env

```bash
source ~/azure-timefold-company-env.sh
echo "ST=$ST  CONTAINER=$CONTAINER"
```

## Step 2 — Confirm you can list the container

```bash
az storage container list --account-name "$ST" --auth-mode login -o table
```

You should see `$CONTAINER` in the list, no error. If you get
`AuthorizationPermissionMismatch`, your role from Phase 2.1 hasn't
propagated yet or wasn't actually assigned — wait a minute, re-check in the
portal, then retry.

> `--auth-mode login` uses your signed-in AAD identity instead of an account
> key. Never use account keys on the company subscription — they're a
> shared secret, not tied to your identity, and harder to audit/revoke.

## Step 3 — Upload a test file

```bash
LOCAL_YAML=/c/Users/Seiya/Desktop/work/Timefold/web/Timefold/src/main/resource/EnvConfig.yaml

az storage blob upload \
  --account-name "$ST" \
  --container-name "$CONTAINER" \
  --auth-mode login \
  --name "input/company-smoke-test/EnvConfig.yaml" \
  --file "$LOCAL_YAML" \
  --overwrite
```

## Step 4 — Confirm it landed

```bash
az storage blob list \
  --account-name "$ST" \
  --container-name "$CONTAINER" \
  --auth-mode login \
  --prefix "input/company-smoke-test/" \
  --query "[].{name:name, size:properties.contentLength}" \
  -o table
```

Should show one row.

## Step 5 — Download it back

```bash
az storage blob download \
  --account-name "$ST" \
  --container-name "$CONTAINER" \
  --auth-mode login \
  --name "input/company-smoke-test/EnvConfig.yaml" \
  --file "/tmp/downloaded-smoke-test.yaml"

head -10 /tmp/downloaded-smoke-test.yaml
```

If you see real YAML content, the round trip works.

## Step 6 — Generate a SAS URL and open it in a browser

This is the same mechanism the real API Controller will use for the
"download result" feature — proving it here first isolates any problem to
Blob/permissions rather than to code you haven't written yet.

```bash
EXPIRY=$(date -u -d '+1 hour' '+%Y-%m-%dT%H:%MZ')

az storage blob generate-sas \
  --account-name "$ST" \
  --container-name "$CONTAINER" \
  --name "input/company-smoke-test/EnvConfig.yaml" \
  --permissions r \
  --expiry "$EXPIRY" \
  --auth-mode login \
  --as-user \
  --https-only \
  --full-uri \
  --output tsv
```

Paste the printed URL into a browser. You should see the YAML content or a
download prompt.

## Step 7 — Clean up the smoke test blob

Don't leave test data lying around in the company account:

```bash
az storage blob delete \
  --account-name "$ST" \
  --container-name "$CONTAINER" \
  --auth-mode login \
  --name "input/company-smoke-test/EnvConfig.yaml"
```

---

## What you should have at the end of this phase

- [ ] `az storage container list` shows `$CONTAINER` with no auth error
- [ ] Test upload succeeded
- [ ] Test download round-tripped real content
- [ ] SAS URL opened successfully in a browser
- [ ] Smoke-test blob deleted afterward

Next: [Azure-Company-04-Register-Timefold-Image.md](./Azure-Company-04-Register-Timefold-Image.md)
— build the Timefold solver's Docker image and push it to the company's
Container Registry.

---

## Troubleshooting

| Symptom                                                          | Cause                                              | Fix |
| ------------------------------------------------------------------ | ----------------------------------------------------- | ----- |
| `AuthorizationPermissionMismatch`                                 | Role not propagated yet, or not actually assigned    | Re-check Phase 2.1 role #2 in the portal; wait 60s and retry |
| `--auth-mode login` errors despite being signed in                | Wrong tenant active                                   | `az account show` → check tenant → `az login --tenant <tenantId>` if wrong |
| SAS URL returns `AuthenticationFailed` in the browser              | Expired, or PC clock is off                            | Regenerate with a longer `--expiry`; check your system clock |
| `date -d '+1 hour'` fails                                          | Not GNU date (rare on Git Bash)                        | Use `date -u -v+1H '+%Y-%m-%dT%H:%MZ'` (macOS-style) as a fallback |
