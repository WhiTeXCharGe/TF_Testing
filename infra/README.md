# infra/ — Azure deployment (for later)

> Not needed yet. The whole stack runs locally without Azure — see
> `server/MOCK_AZURE.md` (`npm run dev:mock`). Use this once local testing of
> every feature is done.

## What `deploy.sh` creates

| Resource | Purpose |
|---|---|
| Resource group | container for everything below |
| Container Registry (ACR) | holds the one image (`ROLE` env picks aca1/aca2) |
| Storage account + `sessions` container | session state (`baseline` / `log` / `meta` / `status` blobs) |
| Key Vault | source of truth for `internal-key` + `blob-conn` |
| Container Apps environment | shared runtime |
| `ca-gantt-aca2` | live relay — `minReplicas 0`, sticky sessions, external ingress `:4010` |
| `ca-gantt-aca1` | session API — `minReplicas 1`, external ingress `:4000` |

## Run it

```bash
az login
az account set -s <subscription>
export SUFFIX=abc123           # short, globally unique (ACR + storage names)
export WEB_ORIGIN=https://<static-web-apps-url-or-custom-domain>
./infra/deploy.sh              # from the repo root (build context is ./server)
```

Everything else has a default (`infra/deploy.sh` top block) and can be overridden by env var.

## After the backend is up

1. Build the web client with `VITE_ACA1_URL=https://<aca1-fqdn>` and deploy to
   Azure Static Web Apps (Free tier).
2. Set `WEB_ORIGIN` to the Static Web Apps URL and re-run:
   ```bash
   az containerapp update -n ca-gantt-aca1 -g rg-gantt-collab --set-env-vars WEB_ORIGIN=<url>
   az containerapp update -n ca-gantt-aca2 -g rg-gantt-collab --set-env-vars WEB_ORIGIN=<url>
   ```
3. Custom domain + managed cert on ACA1 and the SWA (`az containerapp hostname add/bind`).

## Redeploys

`deploy-collab.yml` (GitHub Actions) rebuilds the image and rolls both apps on
push. Needs repo secret `AZURE_CREDENTIALS` (a service principal with
Contributor on the resource group).

## Cost (single region, internal use)

ACA1 one small always-on replica + ACA2 mostly at zero + Blob (cents) +
Static Web Apps (free) ≈ **$5–20/mo**. Scale-out later needs a Socket.IO Redis
adapter for ACA2 (see the design doc §6.1).
