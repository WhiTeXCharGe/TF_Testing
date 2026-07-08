# Company Phase 4 — Build the Timefold image and push it to the registry

**Goal of this phase:** the Docker image built from `web/Timefold/` (the
Java solver you've been testing locally) is pushed into the company's
Container Registry (`$ACR`), ready for Azure Batch to pull in Phase 5.

This is the "register docker to it" step — "it" being the Azure Container
Registry.

**Time:** ~20 minutes (mostly upload time, depends on connection speed).
**Prereqs:** Phase 1–3 done. Docker Desktop installed and running. Role
`AcrPush` from Phase 2.1 confirmed on `$ACR`.

---

## Step 1 — Source env, confirm Docker is running

```bash
source ~/azure-timefold-company-env.sh
echo "ACR=$ACR  ACR_LOGIN=$ACR_LOGIN"

docker --version
```

If `docker --version` fails, open Docker Desktop and wait for the whale
icon in the system tray to go steady, then retry.

## Step 2 — Authenticate Docker to the company ACR

```bash
az acr login --name "$ACR"
```

Expected output ends with `Login Succeeded`. This exchanges your signed-in
AAD session for a short-lived Docker credential — no password to copy or
store anywhere.

If this fails with a permission error, go back to
[Azure-Company-02-RBAC.md §2.1](./Azure-Company-02-RBAC.md#21--your-own-account-5-roles)
and confirm `AcrPush` is actually assigned to you on `$ACR`.

## Step 3 — Build the image from `web/Timefold/`

This is the exact Dockerfile you've already been using locally via
`docker compose` in that folder — we're just tagging the build for the
company registry instead of a local-only name.

```bash
cd /c/Users/Seiya/Desktop/work/Timefold/web/Timefold

TAG=v1
docker build -t "$ACR_LOGIN/timefold:$TAG" .
```

First build takes 5–10 minutes (Maven downloads dependencies the first
time). If `pom.xml` / `docker/pom-standalone.xml` haven't changed since your
last local build, Docker reuses cached layers and this finishes in seconds.

Confirm the image exists locally under its new name:
```bash
docker images "$ACR_LOGIN/timefold"
```

## Step 4 — Push to the company registry

```bash
docker push "$ACR_LOGIN/timefold:$TAG"
```

Upload time depends on your connection — typically a few minutes for the
~280 MB image. You'll see a `digest: sha256:...` line per layer as it
finishes.

## Step 5 — Confirm it's really there

```bash
az acr repository list --name "$ACR" -o table
az acr repository show-tags --name "$ACR" --repository timefold -o table
az acr repository show --name "$ACR" --image "timefold:$TAG" \
  --query "{name:name, tag:tag, size:size, lastUpdated:lastUpdateTime}" -o table
```

You should see repository `timefold`, tag `v1`, size around 280 MB.

## Step 6 — Save the image reference to your env script

Phase 5 (Batch) and Phase 6 (API Controller) both need the full image name.

```bash
cat >> ~/azure-timefold-company-env.sh <<EOF

# Phase 4 additions — Timefold solver image
export SOLVER_IMAGE="$ACR_LOGIN/timefold:$TAG"
EOF
source ~/azure-timefold-company-env.sh
echo "SOLVER_IMAGE=$SOLVER_IMAGE"
```

## Step 7 — (Optional) Prove the pull direction works too

Delete your local copy and pull it back down — proves the round trip an
Azure Batch node will do later:

```bash
docker rmi "$ACR_LOGIN/timefold:$TAG"
docker pull "$ACR_LOGIN/timefold:$TAG"
```

---

## What you should have at the end of this phase

- [ ] `az acr login --name $ACR` succeeds
- [ ] `docker build` from `web/Timefold/` succeeds locally
- [ ] `docker push` completes with no errors
- [ ] `az acr repository show-tags` lists `v1` under `timefold`
- [ ] `$SOLVER_IMAGE` saved in your env script

Next: [Azure-Company-05-Batch-Setup-And-Run.md](./Azure-Company-05-Batch-Setup-And-Run.md)
— point the Batch pool at this image and run a real end-to-end solve.

---

## Troubleshooting

| Symptom                                                       | Cause                                              | Fix |
| ---------------------------------------------------------------- | ------------------------------------------------------ | ----- |
| `az acr login` fails with a permission error                     | `AcrPush` missing on `$ACR`                             | Go back to Phase 2.1, assign/confirm it, wait 60s, retry |
| `denied: requested access to the resource is denied` on push     | Your `az acr login` token expired (valid ~24h)          | Re-run `az acr login --name "$ACR"` |
| `docker build` fails with `Non-resolvable parent POM`            | You're not running the command from `web/Timefold/`     | `cd` into that exact folder first — it's where the Dockerfile and `docker/pom-standalone.xml` live |
| `docker push` hangs or is very slow                              | Large image over a slow/unstable connection             | Be patient, use a wired connection if possible; push resumes automatically if interrupted |
| `repository name must be lowercase`                              | Capital letters somewhere in `$ACR_LOGIN` or the tag     | Registry names are always lowercase; double-check your env script has no typos |
