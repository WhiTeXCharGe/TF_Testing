# Company Phase 7 — Connect the webapp to Azure

**Goal of this phase:** the React webapp you already use every day talks to
the real `$APP_URL` (the ACA app from Phase 6) instead of your local
`service/` mock on `localhost:3001`. No application code changes — this is
purely a config switch, because
[webapp/src/services/runService.ts](../src/services/runService.ts) already
calls whatever URL is in `APP_CONFIG.apiBaseUrl`.

**Time:** ~10 minutes.
**Prereqs:** Phase 6 done — `https://$APP_URL/health` returns `{"ok":true}`.

---

## Step 1 — Point the webapp at the real API

`webapp/.env` (if present on this PC) is your **local-only default** —
it's git-ignored, so it may not exist at all on the other PC. Either way,
use `.env.local` for the Azure switch: Vite loads it *after* `.env` and
lets it override, and it's easy to delete to instantly go back to local
mode.

```bash
cd /c/Users/Seiya/Desktop/work/Timefold/web/webapp
source ~/azure-timefold-company-env.sh

cat > .env.local <<EOF
VITE_API_BASE_URL=https://$APP_URL
EOF

cat .env.local
```

> Vite only reads env files at dev-server **startup**. If `npm run dev` is
> already running, stop it (Ctrl+C) and restart after creating/editing this
> file.

## Step 2 — Start the webapp

```bash
npm install    # only needed the first time on this PC
npm run dev
```

Open the printed local URL (typically http://localhost:5173) in a browser.

## Step 3 — Confirm which backend is active

Check the browser's dev tools console, or just trust the network tab: once
you submit a run (Step 4), every solver-related request should go to
`https://$APP_URL/...`, not `localhost:3001`.

If the webapp's UI has a "Local vs Azure" indicator, it should reflect
Azure now that `VITE_API_BASE_URL` is set (the `solverEnabled` flag in
[webapp/src/hooks/useRuns.ts](../src/hooks/useRuns.ts) turns on automatically
whenever this variable is non-empty — nothing else to configure for that).

## Step 4 — Run the full click-through test

1. **New Run** → attach both `EnvConfig.yaml` and `Schedule.yaml` → submit.
2. Open the browser's Network tab → confirm the `POST .../runSolver` request
   went to `https://$APP_URL/runSolver` and returned `202` with a `runId`.
3. Watch the run's status update — it should move
   `Submitted → Running → Completed` as Azure Batch solves it (same timing
   as Phase 5's manual test).
4. **Show Result** / download action → confirm the file downloads and opens
   correctly (this hits `GET /download/:runId`, which streams straight from
   Blob).
5. Try **Delete** (or cancel, if the UI exposes it mid-run) → confirm the
   run disappears and, if you check the portal, the corresponding blobs
   under `input/`, `output/`, `status/` for that run are gone.

## Step 5 — Confirm in the Azure Portal too

Storage account `$ST` → Containers → `$CONTAINER` → you should see
`input/<runId>/` (and, once complete, `output/<runId>/`) appear and
disappear as you create/delete runs from the webapp — real proof the UI is
driving real Azure resources, not a local mock.

## Step 6 — Switching back to local mode later

If you need to go back to testing against the local `service/` mock:
```bash
rm /c/Users/Seiya/Desktop/work/Timefold/web/webapp/.env.local
```
Restart `npm run dev` — the webapp falls back to whatever `VITE_API_BASE_URL`
is in `.env` (or local-only mode if that's also unset).

---

## What you should have at the end of this phase

- [ ] `webapp/.env.local` contains `VITE_API_BASE_URL=https://$APP_URL`
- [ ] `npm run dev` restarted after creating/editing that file
- [ ] New Run → real Blob upload confirmed (Network tab + portal)
- [ ] Status polling shows real progress from Azure Batch
- [ ] Download returns the real solved YAML
- [ ] Delete removes the run's blobs in Azure

**You now have a complete, real, end-to-end demo:** webapp → API Controller
(ACA) → Blob Storage + Azure Batch (Timefold solver) → back to the webapp.
Every piece is the company's actual Azure account, not a local mock.

---

## Troubleshooting

| Symptom                                                         | Cause                                                              | Fix |
| ------------------------------------------------------------------ | --------------------------------------------------------------------- | ----- |
| Webapp still calls `localhost:3001`                                 | Dev server wasn't restarted after creating `.env.local`                | Stop with Ctrl+C, run `npm run dev` again |
| New Run upload fails with a CORS error in the browser console        | Very unlikely — the API enables CORS for all origins                    | Hard refresh (Ctrl+Shift+R); check `az containerapp logs show` on the API side for what it actually received |
| Status stays on "Submitted" forever                                  | Batch task never actually got created, or the pool has 0 nodes and isn't scaling up | Check Phase 5/6 troubleshooting; `az batch pool show` to confirm node count is increasing |
| Download button does nothing / errors                                | Status isn't actually `Completed` yet, or `output/<runId>/result_Schedule.yaml` is missing | Check `/status/<runId>` directly via curl first, isolate webapp vs. API vs. Batch |
| Everything worked once, now a new run silently fails                 | Pool scaled to 0 between sessions and needs to scale back up (cold start), or the vCPU quota got used up by another run | Check `az batch pool show` node counts; give it a few minutes for autoscale/low-priority provisioning |
