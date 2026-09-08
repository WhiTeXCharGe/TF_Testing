# Mock Azure stack (ACA1 + ACA2 + Blob)

Runs the online-collaboration backend locally, with the exact code that deploys
to Azure Container Apps. No Azure account or network access needed.

## What runs

| Piece | Local | Azure |
|---|---|---|
| **ACA1** – session API (list / create / open / delete) | `http://localhost:4000` | Container App, always on |
| **ACA2** – live relay (Socket.IO) + `/internal/*` control plane | `http://localhost:4010` | Container App, scale-to-zero |
| **Storage** – `sessions/<id>/{meta,status,baseline,log}.json` | `server/mock-blob/` folder | Azure Blob |
| **Web client** | Vite `http://localhost:5173` | Azure Static Web Apps |

One image / one codebase; `ROLE=aca1｜aca2｜local` selects behaviour (`src/config.ts`).

## Run it

### Option A — node (no Docker)

```bash
# from GanttChartEditor/
npm run dev:mock
```

Starts ACA1 (:4000), ACA2 (:4010) and the web client (:5173), all sharing
`server/mock-blob/`. Ctrl-C stops all three.

### Option B — Docker Compose (ACA1 + ACA2 only)

```bash
# from GanttChartEditor/server/
docker compose -f docker-compose.mock.yml up --build
```

Then run the web client separately with `npm run dev` if you need it.

## Poke at it

- **Postman:** import `server/postman/OnlineCollabAzure.postman_collection.json`
  (`baseUrl` = `http://localhost:4000`). Covers create (JSON + 2-YAML upload),
  list, get, open, delete. The create requests stash `sessionId` / `ownerToken`
  into collection variables for the later requests.
- **Lock / unlock** are Socket.IO events on ACA2 (`lock` / `unlock`, owner only,
  after joining), not HTTP — so they are not in the Postman collection. The full
  path (create → 2 clients → edit → lock → unlock → last-leave flush → re-open
  replay → delete) is covered by `src/__integration__/localFlow.test.ts`.

## Config knobs (env)

| Var | Default | Notes |
|---|---|---|
| `ROLE` | `local` | `aca1` \| `aca2` \| `local` |
| `PORT` | 3010/4000/4010 | per role; `0` = ephemeral |
| `STORAGE` | `memory` (local) / `fs` | `fs` \| `blob` \| `memory` |
| `MOCK_BLOB_DIR` | `../mock-blob` | folder for `STORAGE=fs` |
| `INTERNAL_KEY` | – | required for `aca1`/`aca2`; shared ACA1↔ACA2 secret |
| `ACA2_URL` | `http://localhost:4010` | ACA1 → ACA2 base URL |
| `PUBLIC_RELAY_URL` | `http://localhost:<port>` | URL ACA2 records for clients to connect to |
| `WEB_ORIGIN` | – | CORS allow-origin for `aca1`/`aca2` (null → LAN allowlist) |
