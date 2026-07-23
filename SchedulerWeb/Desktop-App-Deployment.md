# SchedulerWeb Desktop App — Build & Deploy Notes

This covers the Electron desktop build specifically: how the Azure connection
gets baked in, what a machine needs to actually run it, and how it finds
GanttChartEditor. For the underlying Azure architecture itself, see
[Azure.md](Azure.md); for the actual "point the webapp at Azure" steps, see
[../azure/Azure-Company-07-Webapp-Connect.md](../azure/Azure-Company-07-Webapp-Connect.md)
— this doc just explains how those same steps apply to the packaged app.

---

## 1. The one thing that trips people up: build-time vs runtime

`VITE_API_BASE_URL` (which backend the app talks to) is read from `.env` /
`.env.local` **only while running `vite build`**. Vite bakes it into the
compiled JS as a literal string — there is no way to change it after the
`.exe` is built short of rebuilding. Packaging does not make this dynamic.

This means two completely different machines are involved, with two
completely different requirements:

| | Needs Azure **permission**? | Needs Azure **network access**? |
|---|---|---|
| **Build machine** (runs `npm run electron:build`) | No — it only needs to know the *URL string* (`https://<APP_URL>`), not credentials or connectivity. It never calls Azure during the build. | No |
| **Run machine** (the PC/company that opens the `.exe`) | No — the app makes plain HTTPS calls, same as a browser | **Yes** — it needs outbound network access to that URL, same as the "other PC" that already works |

So: **this PC not having Azure permissions is not a blocker for building an
Azure-pointed `.exe`** — you just need the correct `$APP_URL` string (get it
from whoever ran the Azure-Company-07 phase, or from the Azure Portal →
Container Apps → your API Controller app → the URL shown there). Building
itself never touches Azure.

What *would* be a real blocker: if the target PC's network (firewall/VPN/
proxy) can't reach that URL at runtime. That's the same requirement the
already-working company PC already satisfies — if that PC works today, a
`.exe` built with the same URL will work on it too.

## 2. Building a build that talks to Azure

By default this repo's `.env` points at `http://localhost:3001` (the local
Docker orchestrator) — a build made without changing anything runs in
**local-only mode**, and the solver/download buttons that call Azure won't
do anything (`solverEnabled` in `useRuns.ts` is only true when
`VITE_API_BASE_URL` is non-empty).

To build a `.exe` that talks to the real Azure backend:

```bash
cd SchedulerWeb

cat > .env.local <<EOF
VITE_API_BASE_URL=https://<APP_URL>
EOF

npm run electron:build
```

`.env.local` overrides `.env` and is git-ignored, so it's safe to create
per-machine without affecting anyone else's local setup. The resulting
`.exe` (in `release/win-unpacked/` or the NSIS installer once Developer Mode
lets that step run) has that URL permanently baked in — no `.env` file needs
to travel with it, and there's nothing to configure on the machine you hand
it to.

To confirm which mode a given build is in without reading source: open the
app, submit a run, and check whether it calls `localhost:3001` or the real
`https://<APP_URL>` — same check as Step 3 in Azure-Company-07.

## 3. What the run machine actually needs

Nothing beyond the extracted folder and, if Azure mode, network reachability
to `$APP_URL`. Specifically **not** required on that machine: Node.js, npm,
git, or this source repo. The `.exe` bundles its own backend (`localApi/`)
and Chromium runtime.

On first launch it creates a `local/` folder next to the `.exe`
(`runs.json`, `<runId>/input/`, `<runId>/output/`) — this is where uploaded
files and downloaded solver output actually land now (see the main fix:
previously Azure-mode downloads only ever reached the browser's Downloads
folder). Nothing needs pre-creating; it's lazy.

## 4. Distribution layout — so the two apps find each other

Extract (or copy `release/win-unpacked/`) both apps as **sibling folders
under one common parent**, e.g.:

```
SomeFolder/
  GanttChartEditor/
    GanttChartEditor.exe
  SchedulerWeb/
    Timefold Scheduler.exe
```

Both apps scan their own parent folder for the other's `.exe` by name the
first time a cross-app action (計画管理ツールへ送信 / 結果を表示) is used —
no manual "locate the file" step needed as long as they're laid out this
way. It doesn't matter what the common parent folder is named, only that
both app folders sit directly inside it.

If they're *not* laid out this way (e.g. installed separately, or on
different drives), the app falls back to asking once via a native file
picker, then remembers that choice for next time.

## 5. Quick troubleshooting

- **Solver/download buttons do nothing** → build was made without
  `VITE_API_BASE_URL` set (local-only mode). Rebuild per §2.
- **"計画管理ツールへ送信" can't find the other app** → the two `.exe`
  folders aren't siblings under a common parent (§4). Either move them, or
  let the native picker prompt find it once — it'll remember for next time.
- **Nothing appears to save** → check the `local/` folder actually exists
  next to the `.exe` and isn't blocked by antivirus/read-only permissions on
  wherever the app was extracted to.
