# Phase 7 — Webapp wired to Azure

**Goal of this phase:** flip the React webapp's calls from the local Vite
middleware to your live Azure API Controller. Same UI, same code, just a
different base URL.

By the end:
- `npm run dev` in webapp → all New Run / Show Result / Delete actions hit
  Azure, not local disk
- Files appear in real Blob Storage
- Status polls work, downloads via real SAS URLs

**Time:** ~20 minutes
**Cost:** $0 extra
**Prereqs:** Phase 6 done. Your API is live at `https://$APP_URL/health → {"ok":true}`.

---

## What changes (minimally)

The webapp already has `APP_CONFIG.apiBaseUrl` wired through
`src/config/appConfig.ts` — when set, the runStore should prefix every
endpoint URL with it. We just need to:

1. Set `VITE_API_BASE_URL` in a webapp `.env` file
2. Update `src/services/runStore.ts` to use it as the prefix
3. (optional) Show a small "Connected to: …" badge so we can see at a glance

That's it. The endpoint shapes are identical (POST /runSolver, GET /status,
etc.) so the existing UI components keep working.

---

## Step 1 — Add VITE_API_BASE_URL to the webapp

In Git Bash:

```bash
cd /c/Users/Seiya/Desktop/work/Timefold/web/webapp

# Recall the URL from the env script
source ~/azure-timefold-env.sh
echo "API base: https://$APP_URL"
```

Create `web/webapp/.env.local` (git-ignored by Vite's default):

```bash
cat > .env.local <<EOF
VITE_API_BASE_URL=https://$APP_URL
EOF

cat .env.local
```

> Vite only reads env vars prefixed with `VITE_` and only at dev-server startup.
> If you change this file, restart `npm run dev`.

---

## Step 2 — Make runStore.ts use the base URL when set

Open `web/webapp/src/services/runStore.ts`. Find the top:

```javascript
const API = {
  runs:   '/api/runs',
  upload: '/api/upload',
  run:    (id) => `/api/run/${encodeURIComponent(id)}`,
  output: (id) => `/api/run/${encodeURIComponent(id)}/output`,
};
```

Replace with this Azure-aware version:

```typescript
import { APP_CONFIG } from '@/config/appConfig';

// When VITE_API_BASE_URL is set, talk to the real API (e.g. https://ca-tf-api...).
// When empty, fall back to the local Vite middleware (/api/...).
const BASE = APP_CONFIG.apiBaseUrl?.replace(/\/$/, '') ?? '';

// Azure endpoint shapes from Azure-06; local-middleware shapes from vite.config.ts.
// They share the SAME paths so we only swap the prefix.
const API = {
  runs:    `${BASE}/api/runs`,                            // local only — Azure has /status/{id} per-run instead
  upload:  BASE ? `${BASE}/runSolver`     : '/api/upload',
  status:  (id: string) => BASE ? `${BASE}/status/${encodeURIComponent(id)}` : `${BASE}/api/run/${encodeURIComponent(id)}`,
  download:(id: string) => BASE ? `${BASE}/download/${encodeURIComponent(id)}` : `${BASE}/api/run/${encodeURIComponent(id)}/output`,
  cancel:  (id: string) => `${BASE}/cancel/${encodeURIComponent(id)}`,
  del:     (id: string) => BASE ? `${BASE}/run/${encodeURIComponent(id)}`     : `${BASE}/api/run/${encodeURIComponent(id)}`,
  mockComplete: (id: string) => `${BASE}/mock-complete/${encodeURIComponent(id)}`,   // dev-only convenience
};
```

The rest of the file's functions (`fetchRuns`, `uploadRun`, `deleteRun`,
`checkOutput`, `resetRuns`) use the API object — they don't need to know
which backend they're hitting.

---

## Step 3 — `fetchRuns()` adjustment for Azure

The local middleware exposes `GET /api/runs` returning the whole runs list.
The Azure API doesn't — it only knows per-run status. Two clean options:

### Option A (simplest for the demo) — keep a localStorage list of runIds
The webapp remembers which runs it created and fetches each one's status
individually:

```typescript
const LOCAL_KEY = 'azure_runIds';
function readLocalIds(): string[] {
  try { return JSON.parse(localStorage.getItem(LOCAL_KEY) ?? '[]'); } catch { return []; }
}
function addLocalId(id: string): void {
  const ids = readLocalIds();
  if (!ids.includes(id)) localStorage.setItem(LOCAL_KEY, JSON.stringify([id, ...ids]));
}
function removeLocalId(id: string): void {
  const ids = readLocalIds().filter(x => x !== id);
  localStorage.setItem(LOCAL_KEY, JSON.stringify(ids));
}

export async function fetchRuns(): Promise<Run[]> {
  if (!BASE) {
    // Local mode — use the old runs.json endpoint
    const res = await fetch(API.runs, { cache: 'no-store' });
    const data = await res.json() as RunsResponse;
    return (data.runs ?? []).sort((a, b) => b.solveDate.localeCompare(a.solveDate));
  }

  // Azure mode — fetch each known run's status
  const ids = readLocalIds();
  const runs = await Promise.all(ids.map(async id => {
    try {
      const res = await fetch(API.status(id), { cache: 'no-store' });
      if (!res.ok) return null;
      const s = await res.json();
      return {
        id, solveDate: s.startedAt, label: 'Azure run',
        folderPath: `azure://${id}/`,
        inputEnvName: s.savedEnvPath?.split('/').pop() ?? 'EnvConfig.yaml',
        inputSchedName: s.savedSchedPath?.split('/').pop() ?? 'Schedule.yaml',
        inputDir: null,
        output: s.status === 'Completed' ? 'ready' : 'none',
        outputHasYaml: s.status === 'Completed',
        originalEnvPath:   s.originalEnvPath,
        originalSchedPath: s.originalSchedPath,
        savedEnvPath:      s.savedEnvPath,
        savedSchedPath:    s.savedSchedPath,
        savedOutputPath:   s.output,
      } as Run;
    } catch { return null; }
  }));
  return runs.filter((r): r is Run => r !== null).sort((a, b) => b.solveDate.localeCompare(a.solveDate));
}
```

Update `uploadRun()` to also save the new id and `deleteRun()` to forget it:

```typescript
export async function uploadRun(p: UploadPayload): Promise<Run> {
  // ... existing FormData build ...
  const data = await res.json() as UploadResponse;
  if (BASE) addLocalId(data.run.id ?? data.runId);    // remember it for fetchRuns
  return data.run;
}

export async function deleteRun(id: string): Promise<void> {
  // ... existing fetch ...
  removeLocalId(id);
}
```

### Option B (cleaner long-term) — add `GET /runs` to the API later
List blobs under `status/` to enumerate runIds. Not needed for the demo.

For tomorrow, **Option A is plenty**.

---

## Step 4 — Tiny visual indicator (optional but nice for the demo)

Add to `src/components/layout/Topbar.tsx` (or wherever the top bar lives):

```tsx
import { APP_CONFIG } from '@/config/appConfig';

// somewhere in the render
{APP_CONFIG.apiBaseUrl
  ? <span style={{ background: '#e8f0fe', color: '#1a73e8', padding: '2px 8px', borderRadius: 10, fontSize: 11, fontWeight: 600 }}>
      Azure
    </span>
  : <span style={{ background: '#f1f3f4', color: '#5f6368', padding: '2px 8px', borderRadius: 10, fontSize: 11, fontWeight: 600 }}>
      Local
    </span>}
```

So during the demo you can point at the badge and say "now we're talking to Azure."

---

## Step 5 — Restart dev server and test

```bash
# Make sure the .env.local change is picked up
cd /c/Users/Seiya/Desktop/work/Timefold/web/webapp
npm run dev
```

Open http://localhost:5173 in your browser. The badge should say **Azure**.

Then:

1. **Click New Run** → drag both YAML files → optionally paste original paths → Submit
2. Check the storage portal: `input/<runId>/` should have your YAMLs
3. Hover the EnvConfig/Schedule chips → popup shows original + saved paths
4. **For the demo, simulate completion via curl in a side terminal:**
   ```bash
   curl -X POST https://$APP_URL/mock-complete/<runId>
   ```
   (or build a tiny "Mock complete" button in the webapp if you want to make it
   one-click — but this is fine for tomorrow)
5. **Click Show Result** → the API returns a SAS URL → editor placeholder dialog opens. The savedOutputPath shows in the result column box.
6. **Click Delete** → confirm → blobs in storage are gone

---

## What you should have at the end of Phase 7

- [ ] `web/webapp/.env.local` contains `VITE_API_BASE_URL=https://$APP_URL`
- [ ] `runStore.ts` uses `BASE` prefix and `azure_runIds` localStorage list
- [ ] Topbar shows **Azure** badge
- [ ] New Run uploads to Azure Blob (verify in portal)
- [ ] Status fetch works (Submitted by default)
- [ ] Mock-complete via curl flips it to Completed
- [ ] Show Result returns a SAS URL the browser can download from
- [ ] Delete removes blobs from Azure

You now have a real end-to-end demo: **webapp → Azure API → Blob Storage**.
The Batch piece is the only thing still mocked, and the leader will see
exactly how it slots in (one more call from the API to Batch, no UI changes).

---

## What's next

- **Phase 8** (later, on personal) — when Batch quota is approved, add the
  Batch task creation in `/runSolver`, remove `/mock-complete`, run a real
  solve through the full stack.
- **Company environment** — see [`Azure-Company-Workflow.md`](./Azure-Company-Workflow.md)
  and [`Azure-Company-Permission-Request.md`](./Azure-Company-Permission-Request.md)
  for the company-PC version of all this.

---

## Troubleshooting

| Symptom                                                        | Cause                                                                    | Fix                                                                          |
| -------------------------------------------------------------- | ------------------------------------------------------------------------ | ---------------------------------------------------------------------------- |
| Badge stays "Local" after editing `.env.local`                  | Vite didn't restart                                                      | Stop `npm run dev` with Ctrl+C, restart                                      |
| New Run upload fails with CORS error in browser console        | API needs CORS for your origin                                           | API already enables `cors()` for all origins (demo). If still failing, browser may be caching — hard refresh (Ctrl+Shift+R) |
| 401 / 403 on /runSolver                                        | ACA MI doesn't have Blob role                                            | Portal: storage → IAM → assign `Storage Blob Data Contributor` to ca-tf-api  |
| Show Result returns "no output" forever                        | You never called `/mock-complete`                                         | Run the curl from Step 5; refresh the webapp                                 |
| `fetchRuns` returns empty list even after uploads              | localStorage `azure_runIds` not updated                                  | Check the `uploadRun` patch added `addLocalId(...)` — without it new runs don't appear in the list |
| Browser downloads a JSON file instead of the YAML when clicking SAS | `Content-Type` confusion                                              | Open the SAS URL in a new tab and "Save as" — the bytes are the right YAML  |
