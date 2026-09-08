# Online Collaboration on Azure (Container Apps) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Date:** 2026-09-08 · **Owner:** (you) · **Builds on:** `GanttChartEditor_OnlineCollabAzure_Design20260904.md`, `GanttChartEditor_LiveCollabEdit_Design20260826.md`

**Goal:** Move the LAN-only live-collaboration relay to a two-service Azure Container Apps design (an always-on **ACA1** session API + a scale-to-zero **ACA2** live relay + Blob storage), reachable from anywhere through a web session list instead of a share link, and runnable in full as a local mock when Azure is unreachable.

**Architecture:** One `server/` codebase, one container image, `ROLE` env var selects behaviour — `aca1` (HTTP session API, no sockets), `aca2` (Socket.IO relay, today's code + persistence), or `local` (both on one port — today's Electron/LAN mode, unchanged). Session state (`baseline` + ordered `log`) lives in storage behind a `StorageClient` interface with `fs` and `blob` implementations, so the same code runs against a local folder or Azure Blob. ACA1 orchestrates ACA2 over internal HTTP; clients talk Socket.IO straight to ACA2.

**Tech Stack:** Node 20 + TypeScript (ESM, NodeNext), Express, Socket.IO 4, `js-yaml`, `zod`, `multer`, Vitest + `supertest` + `socket.io-client` (tests), Docker + Docker Compose, Postman (manual API checks). Azure (phase 7 only): Container Apps, Container Registry, Blob Storage, Key Vault, Static Web Apps, `az` CLI, GitHub Actions.

> **Status (2026-09-08):** Phases 1–2 **implemented** on branch `online-collab-aca` (16 commits). Server: **111 vitest** green incl. the full local integration flow + an Azure Blob `StorageClient` verified against the fs contract via Azurite. Client (`GanttChartEditor_OnlineCollabAzure_ACA_ClientPlan20260908.md`): **174 jest** green, `vite build` clean, session list / create-by-YAML / open / owner lock wired. Verified end-to-end against `npm run dev:mock`: two browser tabs join one session via ACA1→ACA2, both render the replayed baseline, presence shows "2人が参加中", no console errors. Also ready but not run: `infra/deploy.sh` + `.github/workflows/deploy-collab.yml` (Phase 4 — needs Azure access). Not verified: `docker compose` (Docker not installable on the dev machine — virtualization/App-Control locked). `local` Electron/LAN mode unchanged. Remaining: Phase 3 security limits (Appendix B), Phase 4 Azure deploy (Appendix C), Phase 5 docs (Appendix D).

## Global Constraints

- Node **20**, TypeScript **~5.5.3**, `"type": "module"`, `module`/`moduleResolution` **NodeNext** — every relative import ends in `.js`.
- Server tests: **Vitest**, files named `*.test.ts` under `server/src/**`, `environment: 'node'` (see `server/vitest.config.ts`). Never import test files from `dist/`.
- `server/dist/` is shipped verbatim inside the packaged Electron app — no test/devDependency imports may leak into non-test `src/` files.
- The **event-sourced model is unchanged**: server stores and orders actions, never interprets them; the client `reducer.ts` is the single source of truth. Do not add action-type-specific logic to the server.
- **Roles** are `'edit' | 'view'`. **Session status** is exactly `'open' | 'lock' | 'close'`.
- The existing `local` mode (Electron packaged + LAN join) must keep working byte-for-byte — all new behaviour is gated behind `ROLE=aca1` / `ROLE=aca2` / `STORAGE=…`.
- Frequent commits: one per task minimum, Conventional Commits (`feat:`, `fix:`, `test:`, `chore:`).
- Commit message trailer: `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>`.

---

## Team summary (phases + tools)

**Phase 0 — Design**
- Design — this doc (Markdown)
- Check Azure spec (ACA / Blob / ACR limits) — Azure docs, `az` CLI

**Phase 1 — Mock / local** *(this plan, Tasks 1–13)*
- `StorageClient` interface + local-folder impl standing in for Blob — Node `fs`, Vitest
- ACA2 persistence: load baseline + edit history from storage, flush on idle / last leave — TypeScript
- ACA2 lock / unlock via Socket.IO (owner must join first), `session-status` broadcast — Socket.IO
- ACA2 internal endpoints: `activate` / `live` / `evict` — Express, shared internal key
- ACA1 session API: list / create (2-YAML upload **or** desktop JSON) / open / delete — Express, `multer`, `js-yaml`, `zod`
- `status.json` state machine `open` / `lock` / `close` — TypeScript
- One image, `ROLE` env var picks ACA1 vs ACA2 vs local — `process.env`
- Dockerfile + Docker Compose (both services + shared folder) — Docker
- Postman collection (create / list / open / lock / unlock / delete) — Postman
- Full local run-through integration test — Vitest + `socket.io-client`
- `dev:mock` script (ACA1 + ACA2 + web together) — `concurrently`

**Phase 2 — Web client** *(follow-on plan, Appendix A)*
- Session-list landing page, status badges — React + TypeScript
- Create session by uploading 2 YAML — React `<input type=file>`, `FormData`
- Owner lock / unlock button (after joining) — `socket.io-client`
- Socket connects to the ACA2 URL returned by ACA1 — edit `collabService.ts`
- Remove old share-link / LAN-IP UI — edit `SessionDialog.tsx`
- Locked session = read-only banner — reuse existing viewer-gating
- Handle ACA2 cold-start latency — retry/poll on `open`
- Tests — Jest + Testing Library, Cypress

**Phase 3 — Security / limits** *(follow-on plan, Appendix B)*
- Rate-limit create (10/hr/IP) — `express-rate-limit`
- Max sessions / max participants / max upload size — TypeScript guards, `multer` limits
- Internal key enforced on every ACA2 `/internal/*` — Express middleware
- Owner token stored hashed — Node `crypto`
- Action payload cap 1 MB — Socket.IO `maxHttpBufferSize`
- Secrets via env / Key Vault ref — `process.env`, Azure Key Vault

**Phase 4 — Azure deploy** *(follow-on plan, Appendix C — needs Azure access)*
- Provision RG / ACA env / ACR / Blob / Key Vault — `az` CLI
- Build + push one image — `az acr build`
- Deploy ACA1 (`minReplicas 1`) + ACA2 (`minReplicas 0`, `maxReplicas N`, sticky) — `az containerapp`
- Static Web Apps for the web client — Azure Static Web Apps
- Custom domain + managed cert — `az containerapp hostname`
- CI deploy — GitHub Actions
- Flip `STORAGE=fs` → `STORAGE=blob` — `@azure/storage-blob`
- Smoke test on Azure — Postman / `curl`

**Phase 5 — Docs** *(follow-on)*
- Update presentation diagrams (ACA1/ACA2 split, Lock, no share link) — Markdown + Mermaid
- Operator runbook (cost, freeze/kill a session, scale later) — Markdown

---

## Design reference

### Services (one image, `ROLE` selects behaviour)

| `ROLE` | Listens | Does | Azure |
|---|---|---|---|
| `aca1` | HTTP `:PORT` | Session API. Lists sessions, creates them, checks `status.json`, wakes ACA2, deletes sessions. No sockets. | Container App, `minReplicas 1` |
| `aca2` | HTTP + Socket.IO `:PORT` | Live relay (today's `collabSocket`) + persistence + `/internal/*`. | Container App, `minReplicas 0`, `maxReplicas N`, session affinity |
| `local` (default) | HTTP + Socket.IO `:3010` | Everything on one port — current Electron/LAN behaviour, unchanged. | n/a |

### Storage layout (one prefix per session)

| Key | Contents | Written by |
|---|---|---|
| `sessions/<id>/meta.json` | `{ id, name, createdAt, ownerTokenHash }` — immutable | ACA1 on create |
| `sessions/<id>/status.json` | `{ status: 'open'\|'lock'\|'close', relayInstance: string\|null, relayUrl: string\|null, lastActivityAt: number }` | ACA1 on create; ACA2 thereafter |
| `sessions/<id>/baseline.json` | `{ schedule, envConfig, currentView }` | ACA1 on create |
| `sessions/<id>/log.json` | `LoggedAction[]` (ordered, `seq` from 0) | ACA2 on flush |

### Status machine

```
             ACA1 /open  ─────────────►  ACA2 activate ──► status = open
close ───────────────────────────────────────────────────────────────►  open
  ▲                                                                       │
  │  ACA2: last participant leaves → flush log + baseline, status=close    │
  └───────────────────────────────────────────────────────────────────────┘
                                                                          │
                          owner joins, emits `lock`  ──► status = lock ◄───┤
                          owner emits `unlock`        ──► status = open  ───┘

close + lastActivityAt older than ABSOLUTE_MAX (8 h)  ──► ACA1 sweep deletes the prefix
```

- On `POST /api/sessions/:id/open`: `close` → ACA1 calls ACA2 `activate`; `open`/`lock` → ACA1 confirms ACA2 still live (calls `live`; re-activates if the replica died), returns the recorded `relayUrl`.
- `lock`/`unlock` are **Socket.IO events on ACA2**, owner-only — the owner must be joined. ACA2 writes `status.json` and broadcasts `session-status`.
- While `lock`: every inbound `action` is dropped (including the owner's); joiners still get `sync-init`, `presence`, and live `action` replays; presence is kept.

### HTTP + Socket API

**ACA1**

| Method | Path | Body / headers | Returns |
|---|---|---|---|
| `POST` | `/api/sessions` | `application/json` `{ name, schedule, envConfig, currentView }` **or** `multipart/form-data` files `schedule`, `envConfig` (+ field `name`) | `{ ok, sessionId, ownerToken }` |
| `GET` | `/api/sessions` | — | `{ ok, sessions: SessionSummary[] }` |
| `GET` | `/api/sessions/:id` | — | `{ ok, session: SessionSummary }` or 404 |
| `POST` | `/api/sessions/:id/open` | — | `{ ok, sessionId, relayUrl, status }` |
| `DELETE` | `/api/sessions/:id` | header `x-owner-token` | `{ ok }` or 403 |
| `GET` | `/api/health` | — | `{ ok, role: 'aca1' }` |

**ACA2**

| Method | Path | Guard | Purpose |
|---|---|---|---|
| `POST` | `/internal/sessions/:id/activate` | `x-internal-key` | Load `baseline`+`log` into memory (idempotent), write `status.json` (`open` unless already `lock`) with this replica's `relayInstance`/`relayUrl`, return `{ ok, relayUrl, status }` |
| `GET` | `/internal/sessions/:id/live` | `x-internal-key` | `{ ok, active, participantCount, status }` |
| `POST` | `/internal/sessions/:id/evict` | `x-internal-key` | Flush + drop from memory + `status.json` → `close` |
| `GET` | `/api/health` | — | `{ ok, role: 'aca2', instance }` |
| Socket.IO | `/collab/socket.io` | — | see below |

**Socket.IO events** (`*` = new/changed)

| Direction | Event | Payload |
|---|---|---|
| client → server | `join` | `{ sessionId, name, role, ownerToken? }` * (`ownerToken` added) |
| client → server | `action` | `{ type, payload }` (dropped when `status = lock`) * |
| client → server | `leave` | — |
| client → server | `lock` * | — (owner only) |
| client → server | `unlock` * | — (owner only) |
| server → client | `sync-init` | `{ ok, name, baseline, actions, participants, status }` * (`status` added) |
| server → client | `action` | `{ type, payload }` |
| server → client | `presence` | `SessionParticipant[]` |
| server → client | `session-status` * | `{ status: 'open' \| 'lock' }` |

### Shared types

```ts
// server/src/collab/types.ts  (new — extracted so aca1 + aca2 share them)
export type SessionStatus = 'open' | 'lock' | 'close';

export interface SessionBaseline {
  schedule: unknown;
  envConfig: unknown;
  currentView: 'worker' | 'device';
}

export interface LoggedAction { seq: number; type: string; payload: unknown; }

export interface SessionMeta {
  id: string;
  name: string;
  createdAt: number;
  ownerTokenHash: string; // sha256 hex of the ownerToken
}

export interface SessionStatusRecord {
  status: SessionStatus;
  relayInstance: string | null;
  relayUrl: string | null;
  lastActivityAt: number;
}

export interface SessionSummary {
  id: string;
  name: string;
  status: SessionStatus;
  createdAt: number;
  lastActivityAt: number;
  participantCount: number | null; // null when ACA2 is unreachable / asleep
}
```

### File structure

```
GanttChartEditor/
  server/
    Dockerfile                          (new)  build the single image
    .dockerignore                       (new)
    docker-compose.mock.yml             (new)  aca1 + aca2 + shared ./mock-blob volume
    postman/OnlineCollabAzure.postman_collection.json  (new)
    src/
      config.ts                         (new)  read + validate all env → typed config
      collab/
        types.ts                        (new)  shared types (above)
        storage/
          storageClient.ts              (new)  StorageClient interface
          fsStorage.ts                  (new)  folder-backed impl
          blobStorage.ts                (new)  Azure Blob impl (thin; used in Phase 4)
          index.ts                      (new)  makeStorage(config): StorageClient
        persistence.ts                  (new)  create/load/flush/delete session records via StorageClient
        sessionStore.ts                 (modify) add activateFromStorage / markDirty / flushDirty / evict / getLive / lock state
        collabSocket.ts                 (modify) ownerToken on join, lock/unlock events, drop actions when locked, flush+evict+status on last leave, session-status broadcast
        sessionStore.test.ts            (modify) async + persistence coverage
        collabSocket.test.ts            (modify) lock gating, last-leave flush
      internalAuth.ts                   (new)  x-internal-key middleware
      routes/
        internal.ts                     (new)  /internal/sessions/:id/{activate,live,evict}
        collab.ts                       (modify) mounted only in ROLE=local (create/name move to ACA1)
      aca1/
        app.ts                          (new)  express() factory for the session API (supertest-friendly)
        sessionApi.ts                   (new)  route handlers
        yamlIntake.ts                   (new)  parse + validate uploaded YAML → SessionBaseline
        aca2Client.ts                   (new)  typed fetch wrapper for ACA2 /internal/*
        sweep.ts                        (new)  absolute-TTL + orphan-open backstop sweep
        app.test.ts                     (new)
        yamlIntake.test.ts              (new)
      index.ts                          (modify) switch on config.role → local | aca1 | aca2
      __integration__/
        localFlow.test.ts               (new)  full run-through against both apps in-process
  package.json                          (modify) add "dev:mock"
```

---

## Task 1: Shared types module

**Files:**
- Create: `GanttChartEditor/server/src/collab/types.ts`
- Modify: `GanttChartEditor/server/src/collab/sessionStore.ts` (import the shared `SessionBaseline`, `LoggedAction` instead of re-declaring)

**Interfaces:**
- Produces: `SessionStatus`, `SessionBaseline`, `LoggedAction`, `SessionMeta`, `SessionStatusRecord`, `SessionSummary` (exact shapes in Design reference → Shared types).

- [ ] **Step 1: Write the failing test**

Create `GanttChartEditor/server/src/collab/types.test.ts`:

```ts
import { describe, it, expect } from 'vitest';
import type { SessionStatus, SessionBaseline, LoggedAction } from './types.js';

describe('shared collab types', () => {
  it('SessionStatus is one of the three literals', () => {
    const values: SessionStatus[] = ['open', 'lock', 'close'];
    expect(values).toHaveLength(3);
  });

  it('a baseline and an action are structurally usable', () => {
    const baseline: SessionBaseline = { schedule: {}, envConfig: {}, currentView: 'worker' };
    const action: LoggedAction = { seq: 0, type: 'SET_SCHEDULE', payload: { a: 1 } };
    expect(baseline.currentView).toBe('worker');
    expect(action.seq).toBe(0);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd GanttChartEditor/server && npx vitest run src/collab/types.test.ts`
Expected: FAIL — `Cannot find module './types.js'`.

- [ ] **Step 3: Write minimal implementation**

Create `GanttChartEditor/server/src/collab/types.ts` with exactly the six exports from the Design reference (`SessionStatus`, `SessionBaseline`, `LoggedAction`, `SessionMeta`, `SessionStatusRecord`, `SessionSummary`).

- [ ] **Step 4: Update `sessionStore.ts` to consume the shared types**

In `GanttChartEditor/server/src/collab/sessionStore.ts` replace the local `SessionBaseline` and `LoggedAction` declarations with `import type { SessionBaseline, LoggedAction } from './types.js';`. Keep `SessionParticipant` and `CollabSession` local for now. Re-export `SessionBaseline` so existing importers (`routes/collab.ts`) do not break: `export type { SessionBaseline, LoggedAction } from './types.js';`.

- [ ] **Step 5: Run the full server test suite**

Run: `cd GanttChartEditor/server && npx vitest run`
Expected: PASS — all existing tests plus `types.test.ts`.

- [ ] **Step 6: Commit**

```bash
git add GanttChartEditor/server/src/collab/types.ts GanttChartEditor/server/src/collab/types.test.ts GanttChartEditor/server/src/collab/sessionStore.ts
git commit -m "feat(server): extract shared collab types for aca1/aca2"
```

---

## Task 2: `StorageClient` interface + `fsStorage` implementation

**Files:**
- Create: `GanttChartEditor/server/src/collab/storage/storageClient.ts`
- Create: `GanttChartEditor/server/src/collab/storage/fsStorage.ts`
- Create: `GanttChartEditor/server/src/collab/storage/fsStorage.test.ts`

**Interfaces:**
- Produces:
  ```ts
  export interface StorageClient {
    getJson<T>(key: string): Promise<T | null>;      // null when the key is absent
    putJson(key: string, value: unknown): Promise<void>;  // creates parent dirs; atomic write
    listPrefix(prefix: string): Promise<string[]>;   // full keys, sorted, "" prefix = everything
    delete(key: string): Promise<void>;              // no-op when absent
    deletePrefix(prefix: string): Promise<void>;     // removes every key under the prefix
  }
  export function createFsStorage(rootDir: string): StorageClient;
  ```
- A "key" is always `/`-joined segments (e.g. `sessions/abc/meta.json`); `fsStorage` maps it under `rootDir`.

- [ ] **Step 1: Write the failing test**

Create `fsStorage.test.ts`:

```ts
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createFsStorage } from './fsStorage.js';
import type { StorageClient } from './storageClient.js';

let root: string;
let storage: StorageClient;

beforeEach(async () => {
  root = await mkdtemp(join(tmpdir(), 'gantt-fsstore-'));
  storage = createFsStorage(root);
});
afterEach(() => rm(root, { recursive: true, force: true }));

describe('fsStorage', () => {
  it('returns null for a missing key', async () => {
    expect(await storage.getJson('sessions/x/meta.json')).toBeNull();
  });

  it('round-trips JSON, creating parent directories', async () => {
    await storage.putJson('sessions/x/meta.json', { id: 'x', n: 1 });
    expect(await storage.getJson('sessions/x/meta.json')).toEqual({ id: 'x', n: 1 });
  });

  it('overwrites an existing key', async () => {
    await storage.putJson('k.json', { v: 1 });
    await storage.putJson('k.json', { v: 2 });
    expect(await storage.getJson('k.json')).toEqual({ v: 2 });
  });

  it('lists keys under a prefix, sorted, full paths', async () => {
    await storage.putJson('sessions/b/meta.json', {});
    await storage.putJson('sessions/a/meta.json', {});
    await storage.putJson('sessions/a/log.json', []);
    expect(await storage.listPrefix('sessions/')).toEqual([
      'sessions/a/log.json', 'sessions/a/meta.json', 'sessions/b/meta.json',
    ]);
  });

  it('delete removes one key and is a no-op when absent', async () => {
    await storage.putJson('k.json', { v: 1 });
    await storage.delete('k.json');
    await storage.delete('k.json');
    expect(await storage.getJson('k.json')).toBeNull();
  });

  it('deletePrefix removes the whole subtree', async () => {
    await storage.putJson('sessions/a/meta.json', {});
    await storage.putJson('sessions/a/log.json', []);
    await storage.deletePrefix('sessions/a');
    expect(await storage.listPrefix('sessions/a')).toEqual([]);
  });

  it('rejects keys that escape the root', async () => {
    await expect(storage.putJson('../evil.json', {})).rejects.toThrow();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd GanttChartEditor/server && npx vitest run src/collab/storage/fsStorage.test.ts`
Expected: FAIL — modules not found.

- [ ] **Step 3: Write `storageClient.ts`**

```ts
export interface StorageClient {
  getJson<T>(key: string): Promise<T | null>;
  putJson(key: string, value: unknown): Promise<void>;
  listPrefix(prefix: string): Promise<string[]>;
  delete(key: string): Promise<void>;
  deletePrefix(prefix: string): Promise<void>;
}
```

- [ ] **Step 4: Write `fsStorage.ts`**

```ts
import { mkdir, readFile, writeFile, rename, rm, readdir, unlink } from 'node:fs/promises';
import { dirname, join, resolve, sep } from 'node:path';
import { randomUUID } from 'node:crypto';
import type { StorageClient } from './storageClient.js';

export function createFsStorage(rootDir: string): StorageClient {
  const root = resolve(rootDir);

  const toPath = (key: string): string => {
    const p = resolve(root, key);
    if (p !== root && !p.startsWith(root + sep)) throw new Error(`key escapes storage root: ${key}`);
    return p;
  };

  const walk = async (dir: string): Promise<string[]> => {
    let entries;
    try { entries = await readdir(dir, { withFileTypes: true }); }
    catch (err) { if ((err as NodeJS.ErrnoException).code === 'ENOENT') return []; throw err; }
    const out: string[] = [];
    for (const e of entries) {
      const full = join(dir, e.name);
      if (e.isDirectory()) out.push(...await walk(full));
      else out.push(full);
    }
    return out;
  };

  return {
    async getJson<T>(key: string): Promise<T | null> {
      try { return JSON.parse(await readFile(toPath(key), 'utf-8')) as T; }
      catch (err) { if ((err as NodeJS.ErrnoException).code === 'ENOENT') return null; throw err; }
    },
    async putJson(key: string, value: unknown): Promise<void> {
      const path = toPath(key);
      await mkdir(dirname(path), { recursive: true });
      const tmp = `${path}.${randomUUID()}.tmp`;
      await writeFile(tmp, JSON.stringify(value, null, 2), 'utf-8');
      await rename(tmp, path);
    },
    async listPrefix(prefix: string): Promise<string[]> {
      const base = toPath(prefix);
      const files = await walk(base.endsWith(sep) ? base : base);
      return files
        .map((f) => f.slice(root.length + 1).split(sep).join('/'))
        .filter((k) => k.startsWith(prefix.replace(/\/$/, '') === '' ? '' : prefix.replace(/\/+$/, '') ) || prefix === '')
        .sort();
    },
    async delete(key: string): Promise<void> {
      try { await unlink(toPath(key)); }
      catch (err) { if ((err as NodeJS.ErrnoException).code !== 'ENOENT') throw err; }
    },
    async deletePrefix(prefix: string): Promise<void> {
      await rm(toPath(prefix), { recursive: true, force: true });
    },
  };
}
```

> Note for the implementer: `listPrefix('sessions/')` and `listPrefix('sessions/a')` must both work. Simplest correct approach — resolve the prefix to a directory when it ends with `/`, otherwise to its parent dir then filter by the full key `startsWith(prefix)`. Adjust the filter until all `fsStorage.test.ts` cases pass; keep the return value sorted, `/`-joined, root-relative.

- [ ] **Step 5: Run the test to verify it passes**

Run: `cd GanttChartEditor/server && npx vitest run src/collab/storage/fsStorage.test.ts`
Expected: PASS (all 7 cases). Iterate on `listPrefix` if the two list cases disagree.

- [ ] **Step 6: Commit**

```bash
git add GanttChartEditor/server/src/collab/storage/
git commit -m "feat(server): add StorageClient interface and fs implementation"
```

---

## Task 3: `config.ts` — typed environment

**Files:**
- Create: `GanttChartEditor/server/src/config.ts`
- Create: `GanttChartEditor/server/src/config.test.ts`

**Interfaces:**
- Produces:
  ```ts
  export type Role = 'local' | 'aca1' | 'aca2';
  export interface AppConfig {
    role: Role;
    port: number;
    storage: { kind: 'fs'; rootDir: string } | { kind: 'blob'; connectionString: string; container: string };
    webOrigin: string | null;      // CORS allow-origin for aca1/aca2 cloud builds; null → fall back to LAN allowlist (local)
    internalKey: string;           // shared secret for ACA2 /internal/*
    aca2Url: string;               // aca1 → aca2 base URL, e.g. http://localhost:4010
    publicRelayUrl: string;        // the URL clients should open sockets to; written into status.json by aca2
    instanceId: string;            // this replica's id (CONTAINER_APP_REPLICA_NAME ?? HOSTNAME ?? uuid)
    absoluteSessionMaxMs: number;  // default 8h
    idleSweepMs: number;           // default 5min
    idleSessionTimeoutMs: number;  // default 30min
  }
  export function loadConfig(env = process.env): AppConfig;
  ```
- `loadConfig` throws a descriptive `Error` when `ROLE=aca2`/`aca1` and `INTERNAL_KEY` is unset, or `STORAGE=blob` without a connection string.

- [ ] **Step 1: Write the failing test**

```ts
import { describe, it, expect } from 'vitest';
import { loadConfig } from './config.js';

describe('loadConfig', () => {
  it('defaults to local + fs + port 3010', () => {
    const c = loadConfig({});
    expect(c.role).toBe('local');
    expect(c.port).toBe(3010);
    expect(c.storage).toEqual({ kind: 'fs', rootDir: expect.stringContaining('mock-blob') });
  });

  it('reads role and port', () => {
    const c = loadConfig({ ROLE: 'aca1', PORT: '4000', INTERNAL_KEY: 'k', ACA2_URL: 'http://localhost:4010' });
    expect(c.role).toBe('aca1');
    expect(c.port).toBe(4000);
  });

  it('throws when aca2 has no INTERNAL_KEY', () => {
    expect(() => loadConfig({ ROLE: 'aca2' })).toThrow(/INTERNAL_KEY/);
  });

  it('throws on unknown role', () => {
    expect(() => loadConfig({ ROLE: 'nope' })).toThrow(/ROLE/);
  });

  it('parses blob storage config', () => {
    const c = loadConfig({ ROLE: 'aca1', INTERNAL_KEY: 'k', ACA2_URL: 'x', STORAGE: 'blob', BLOB_CONNECTION_STRING: 'cs', BLOB_CONTAINER: 'sessions' });
    expect(c.storage).toEqual({ kind: 'blob', connectionString: 'cs', container: 'sessions' });
  });
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd GanttChartEditor/server && npx vitest run src/config.test.ts` → FAIL (no module).

- [ ] **Step 3: Implement `config.ts`**

Parse with plain code (no zod needed here). Defaults: `ROLE=local`, `PORT` = `3010` for local / required otherwise (default `4000` aca1, `4010` aca2 acceptable), `STORAGE=fs`, `MOCK_BLOB_DIR` default `resolve(process.cwd(), '../mock-blob')` **or** `resolve(rootDir of repo, 'mock-blob')` — use `resolve(process.cwd(), process.env.MOCK_BLOB_DIR ?? '../mock-blob')` and document it. `instanceId` = `env.CONTAINER_APP_REPLICA_NAME ?? env.HOSTNAME ?? randomUUID()`. `publicRelayUrl` = `env.PUBLIC_RELAY_URL ?? \`http://localhost:${port}\``. `aca2Url` default `http://localhost:4010`. TTL defaults: absolute `8*60*60*1000`, sweep `5*60*1000`, idle `30*60*1000`.

- [ ] **Step 4: Run to verify it passes**

Run: `cd GanttChartEditor/server && npx vitest run src/config.test.ts` → PASS.

- [ ] **Step 5: Commit**

```bash
git add GanttChartEditor/server/src/config.ts GanttChartEditor/server/src/config.test.ts
git commit -m "feat(server): add typed env config with role/storage/internal-key"
```

---

## Task 4: `persistence.ts` — session records via `StorageClient`

**Files:**
- Create: `GanttChartEditor/server/src/collab/persistence.ts`
- Create: `GanttChartEditor/server/src/collab/persistence.test.ts`

**Interfaces:**
- Consumes: `StorageClient` (Task 2); `SessionMeta`, `SessionStatusRecord`, `SessionBaseline`, `LoggedAction`, `SessionStatus` (Task 1).
- Produces:
  ```ts
  export function metaKey(id: string): string;      // `sessions/${id}/meta.json`
  export function statusKey(id: string): string;
  export function baselineKey(id: string): string;
  export function logKey(id: string): string;

  export function hashOwnerToken(token: string): string;  // sha256 hex

  export async function createSessionRecord(
    s: StorageClient,
    args: { name: string; baseline: SessionBaseline; ownerTokenHash: string },
  ): Promise<string>;  // returns new id (randomUUID); writes meta+baseline+status(close)+log([])

  export async function loadSessionRecord(
    s: StorageClient, id: string,
  ): Promise<{ meta: SessionMeta; baseline: SessionBaseline; log: LoggedAction[]; status: SessionStatusRecord } | null>;

  export async function writeStatus(
    s: StorageClient, id: string, patch: Partial<SessionStatusRecord>,
  ): Promise<SessionStatusRecord>;  // merges onto existing, always bumps lastActivityAt unless patch sets it

  export async function writeLog(s: StorageClient, id: string, log: LoggedAction[]): Promise<void>;

  export async function listSessionIds(s: StorageClient): Promise<string[]>;  // from `sessions/` prefix, unique dir names

  export async function deleteSessionRecord(s: StorageClient, id: string): Promise<void>;  // deletePrefix(`sessions/${id}`)
  ```

- [ ] **Step 1: Write the failing test** (uses `createFsStorage` in a temp dir)

```ts
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createFsStorage } from './storage/fsStorage.js';
import {
  createSessionRecord, loadSessionRecord, writeStatus, writeLog,
  listSessionIds, deleteSessionRecord, hashOwnerToken,
} from './persistence.js';

let root: string; let s: ReturnType<typeof createFsStorage>;
beforeEach(async () => { root = await mkdtemp(join(tmpdir(), 'gantt-persist-')); s = createFsStorage(root); });
afterEach(() => rm(root, { recursive: true, force: true }));

const baseline = { schedule: { t: 1 }, envConfig: { e: 2 }, currentView: 'worker' as const };

describe('persistence', () => {
  it('creates then loads a full record with status=close and empty log', async () => {
    const id = await createSessionRecord(s, { name: 'Plan A', baseline, ownerTokenHash: hashOwnerToken('secret') });
    const rec = await loadSessionRecord(s, id);
    expect(rec?.meta.name).toBe('Plan A');
    expect(rec?.meta.ownerTokenHash).toBe(hashOwnerToken('secret'));
    expect(rec?.status.status).toBe('close');
    expect(rec?.baseline).toEqual(baseline);
    expect(rec?.log).toEqual([]);
  });

  it('loadSessionRecord returns null for unknown id', async () => {
    expect(await loadSessionRecord(s, 'nope')).toBeNull();
  });

  it('writeStatus merges and bumps lastActivityAt', async () => {
    const id = await createSessionRecord(s, { name: 'x', baseline, ownerTokenHash: 'h' });
    const before = (await loadSessionRecord(s, id))!.status.lastActivityAt;
    await new Promise((r) => setTimeout(r, 2));
    const next = await writeStatus(s, id, { status: 'open', relayInstance: 'r1', relayUrl: 'http://r' });
    expect(next.status).toBe('open');
    expect(next.relayInstance).toBe('r1');
    expect(next.lastActivityAt).toBeGreaterThan(before);
  });

  it('writeLog persists and reloads', async () => {
    const id = await createSessionRecord(s, { name: 'x', baseline, ownerTokenHash: 'h' });
    await writeLog(s, id, [{ seq: 0, type: 'SET_SCHEDULE', payload: { a: 1 } }]);
    expect((await loadSessionRecord(s, id))!.log).toEqual([{ seq: 0, type: 'SET_SCHEDULE', payload: { a: 1 } }]);
  });

  it('listSessionIds returns created ids; deleteSessionRecord removes them', async () => {
    const a = await createSessionRecord(s, { name: 'a', baseline, ownerTokenHash: 'h' });
    const b = await createSessionRecord(s, { name: 'b', baseline, ownerTokenHash: 'h' });
    expect((await listSessionIds(s)).sort()).toEqual([a, b].sort());
    await deleteSessionRecord(s, a);
    expect(await listSessionIds(s)).toEqual([b]);
  });
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd GanttChartEditor/server && npx vitest run src/collab/persistence.test.ts` → FAIL.

- [ ] **Step 3: Implement `persistence.ts`**

Key builders as above. `hashOwnerToken` = `createHash('sha256').update(token).digest('hex')` from `node:crypto`. `createSessionRecord`: `id = randomUUID()`, `putJson` all four keys, `status.json` = `{ status: 'close', relayInstance: null, relayUrl: null, lastActivityAt: Date.now() }`, `meta.json` = `{ id, name, createdAt: Date.now(), ownerTokenHash }`. `loadSessionRecord`: `getJson` the four keys; return `null` if `meta` is null; default `log` to `[]` if that key is null. `writeStatus`: load current (or a `close` default), merge `patch`, set `lastActivityAt: patch.lastActivityAt ?? Date.now()`, `putJson`, return merged. `listSessionIds`: `listPrefix('sessions/')` → take the 2nd path segment → `[...new Set(...)]`.

- [ ] **Step 4: Run to verify it passes** → PASS.

- [ ] **Step 5: Commit**

```bash
git add GanttChartEditor/server/src/collab/persistence.ts GanttChartEditor/server/src/collab/persistence.test.ts
git commit -m "feat(server): add session-record persistence over StorageClient"
```

---

## Task 5: `sessionStore` — storage-backed activate / flush / evict / lock state

**Files:**
- Modify: `GanttChartEditor/server/src/collab/sessionStore.ts`
- Modify: `GanttChartEditor/server/src/collab/sessionStore.test.ts`

**Interfaces:**
- Consumes: `persistence.ts` (Task 4), `StorageClient` (Task 2), `types.ts` (Task 1).
- Produces — the store becomes an injected object rather than module-level functions so a test can pass a fake storage:
  ```ts
  export interface SessionStoreDeps { storage: StorageClient; }
  export function createSessionStore(deps: SessionStoreDeps): SessionStore;

  export interface SessionStore {
    // in-memory, synchronous once activated:
    getSession(id: string): { name: string; baseline: SessionBaseline; actions: LoggedAction[]; participants: SessionParticipant[]; status: SessionStatus } | null;
    appendAction(id: string, type: string, payload: unknown): LoggedAction | null;   // marks dirty
    addParticipant(id: string, pid: string, name: string, role: SessionRole): SessionParticipant[] | null;
    removeParticipant(id: string, pid: string): SessionParticipant[] | null;
    participantCount(id: string): number;
    setLocked(id: string, locked: boolean): SessionStatus | null;                    // 'lock' | 'open'
    isLoaded(id: string): boolean;
    // storage-touching, async:
    activateFromStorage(id: string): Promise<boolean>;   // loads meta+baseline+log into memory if not present; false if no record
    flush(id: string): Promise<void>;                     // write log.json + status.lastActivityAt if dirty
    flushAll(): Promise<void>;
    evict(id: string): Promise<void>;                     // flush, then drop from memory
    getLive(id: string): { active: boolean; participantCount: number; status: SessionStatus };
    ownerTokenHash(id: string): string | null;
  }
  ```
- Note: `createSession` / `getSessionName` module functions are **removed from here** — session creation now lives in ACA1 (`persistence.createSessionRecord`). Local mode (Task 12) creates via `persistence` too.

- [ ] **Step 1: Rewrite the test file around `createSessionStore`**

Replace `sessionStore.test.ts` contents. Use a real `createFsStorage` temp dir, and seed a record with `persistence.createSessionRecord` before exercising the store. Cases:

```ts
// pseudocode outline — write these as real vitest `it(...)` blocks
// setup: root=tmp, storage=createFsStorage(root), store=createSessionStore({storage})
// seed:  id = await createSessionRecord(storage, { name:'S', baseline, ownerTokenHash: hashOwnerToken('own') })

it('activateFromStorage loads baseline+log into memory');           // getSession(id) non-null after, null before
it('activateFromStorage returns false for an unknown id');
it('appendAction assigns increasing seq and marks dirty');
it('flush writes log.json so a fresh store re-activates with the actions');
it('addParticipant / removeParticipant / participantCount track presence');
it('setLocked(true) → getSession().status === "lock"; setLocked(false) → "open"');
it('evict flushes then unloads (isLoaded false, getSession null)');
it('getLive reports active=false + status from storage when not loaded');   // reads status.json
it('ownerTokenHash returns the seeded hash');
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd GanttChartEditor/server && npx vitest run src/collab/sessionStore.test.ts` → FAIL (no `createSessionStore`).

- [ ] **Step 3: Implement**

In-memory `Map<string, InMemSession>` where `InMemSession = { meta, baseline, actions, participants: Map, nextSeq, status: 'open'|'lock', dirty: boolean, lastActivityAt }`. `activateFromStorage`: if already in map → `true`; else `loadSessionRecord`; if null → `false`; else populate map, set `status` to `'open'` (or `'lock'` if the stored status was `'lock'`), `nextSeq = (last log seq)+1`. `appendAction`: push, `dirty=true`, bump `lastActivityAt`. `setLocked`: flip `status`, `dirty=true`. `flush`: if `dirty` → `writeLog(storage,id,actions)` + `writeStatus(storage,id,{ status, lastActivityAt })`, `dirty=false`. `evict`: `await flush(id)` then `map.delete(id)`. `getLive`: if loaded → `{active:true, participantCount, status}`; else read `status.json` via `storage.getJson(statusKey(id))` → `{active:false, participantCount:0, status: rec?.status ?? 'close'}`. Keep `sweepIdleSessions` but operate on the in-memory map only (unchanged semantics) — the ACA1 sweep handles storage.

- [ ] **Step 4: Fix compile fallout**

`routes/collab.ts` and `collabSocket.ts` import the old module functions — they will not compile yet. That is expected; Tasks 6 and 12 rewire them. For now, add `// @ts-expect-error rewired in Task 6` is NOT allowed — instead, keep `collabSocket.ts` compiling by having it accept a `SessionStore` param (do the minimal signature change here, full behaviour in Task 6): `export function createCollabSocketServer(httpServer: HttpServer, store: SessionStore, config: AppConfig): Server` and update `index.ts`'s local path to build a store. Keep local behaviour working.

- [ ] **Step 5: Run the whole suite**

Run: `cd GanttChartEditor/server && npx vitest run` → PASS (sessionStore + persistence + fsStorage + types + config; collabSocket test may need a one-line store injection — update it minimally to pass a `createSessionStore({ storage: createFsStorage(tmp) })`).

- [ ] **Step 6: Commit**

```bash
git add GanttChartEditor/server/src/collab/sessionStore.ts GanttChartEditor/server/src/collab/sessionStore.test.ts GanttChartEditor/server/src/collab/collabSocket.ts GanttChartEditor/server/src/index.ts
git commit -m "feat(server): storage-backed session store with activate/flush/evict/lock"
```

---

## Task 6: `collabSocket` — ownerToken, lock/unlock events, drop-on-lock, flush+evict on last leave

**Files:**
- Modify: `GanttChartEditor/server/src/collab/collabSocket.ts`
- Modify: `GanttChartEditor/server/src/collab/collabSocket.test.ts`

**Interfaces:**
- Consumes: `SessionStore` (Task 5), `AppConfig` (Task 3), `persistence.hashOwnerToken` (Task 4).
- Produces: unchanged export name `createCollabSocketServer(httpServer, store, config)`. New socket contract per Design reference (`join` gains `ownerToken`; new `lock`/`unlock` in, `session-status` out; `sync-init` gains `status`).

- [ ] **Step 1: Add failing tests to `collabSocket.test.ts`**

Using an in-process httpServer + `socket.io-client` (mirror the existing test style). Seed a session record, then:

```ts
it('join lazily activates a session that is only in storage');           // no prior activate call
it('sync-init payload includes status:"open"');
it('an owner (matching ownerToken) emitting "lock" flips status and broadcasts session-status to the room');
it('a non-owner emitting "lock" is ignored (status stays open)');
it('while locked, an edit-role action from anyone is NOT broadcast');
it('after unlock, actions broadcast again');
it('when the last participant disconnects, the log is flushed to storage and status.json becomes "close"');
```

- [ ] **Step 2: Run to verify the new tests fail**

Run: `cd GanttChartEditor/server && npx vitest run src/collab/collabSocket.test.ts` → new cases FAIL.

- [ ] **Step 3: Implement**

- On `connection`: `let isOwner = false;`
- `join`: `const ok = store.isLoaded(sessionId) || await store.activateFromStorage(sessionId);` if `!ok` → `socket.emit('sync-init', { ok: false })`. Else set `joinedSessionId/joinedRole`, `socket.join(sessionId)`, compute `isOwner = !!ownerToken && store.ownerTokenHash(sessionId) === hashOwnerToken(ownerToken)`. `const s = store.getSession(sessionId)!;` emit `sync-init` with `{ ok:true, name:s.name, baseline:s.baseline, actions:s.actions, participants, status: s.status }`. Broadcast `presence`.
- `action`: `if (!joinedSessionId || joinedRole !== 'edit') return;` **`if (store.getSession(joinedSessionId)?.status === 'lock') return;`** then `store.appendAction(...)` + `socket.to(id).emit('action', …)`.
- `lock` / `unlock`: `if (!joinedSessionId || !isOwner) return;` `const status = store.setLocked(joinedSessionId, event === 'lock');` `await store.flush(joinedSessionId);` `io.to(joinedSessionId).emit('session-status', { status });`
- `handleLeave`: remove participant, broadcast `presence`; **`if (store.participantCount(joinedSessionId) === 0) { await store.evict(joinedSessionId); await store.writeStatusClose(...) }`** — `evict` already flushes; then `writeStatus(storage,id,{status:'close', relayInstance:null, relayUrl:null})`. Expose a `store` method `markClosed(id)` that does the `writeStatus` so the socket file does not import `persistence` directly. Add `markClosed(id: string): Promise<void>` to the `SessionStore` interface (update Task 5 impl + interface accordingly — small addition).
- Idle sweep `setInterval` stays but now also calls `store.flushAll()` each tick.

- [ ] **Step 4: Run to verify pass**

Run: `cd GanttChartEditor/server && npx vitest run src/collab/collabSocket.test.ts` → PASS.

- [ ] **Step 5: Full suite + typecheck**

Run: `cd GanttChartEditor/server && npx vitest run && npx tsc --noEmit` → PASS / no errors.

- [ ] **Step 6: Commit**

```bash
git add GanttChartEditor/server/src/collab/collabSocket.ts GanttChartEditor/server/src/collab/collabSocket.test.ts GanttChartEditor/server/src/collab/sessionStore.ts
git commit -m "feat(server): socket lock/unlock, drop-on-lock, flush+close on last leave"
```

---

## Task 7: `internalAuth` middleware + `routes/internal.ts`

**Files:**
- Create: `GanttChartEditor/server/src/internalAuth.ts`
- Create: `GanttChartEditor/server/src/routes/internal.ts`
- Create: `GanttChartEditor/server/src/routes/internal.test.ts`

**Interfaces:**
- Consumes: `SessionStore` (Tasks 5–6), `AppConfig` (Task 3).
- Produces:
  ```ts
  export function internalAuth(expectedKey: string): RequestHandler;  // 401 unless x-internal-key === expectedKey
  export function createInternalRouter(store: SessionStore, config: AppConfig): Router;
  ```
  Routes: `POST /internal/sessions/:id/activate`, `GET /internal/sessions/:id/live`, `POST /internal/sessions/:id/evict`.

- [ ] **Step 1: Write the failing test** (supertest against a bare express app)

```ts
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import express from 'express';
import request from 'supertest';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createFsStorage } from '../collab/storage/fsStorage.js';
import { createSessionStore } from '../collab/sessionStore.js';
import { createSessionRecord, hashOwnerToken } from '../collab/persistence.js';
import { createInternalRouter, internalAuth } from './internal.js';

let root: string; let app: express.Express; let id: string;
const config = { internalKey: 'topsecret', instanceId: 'replica-1', publicRelayUrl: 'http://relay:4010' } as any;

beforeEach(async () => {
  root = await mkdtemp(join(tmpdir(), 'gantt-internal-'));
  const storage = createFsStorage(root);
  id = await createSessionRecord(storage, { name: 'S', baseline: { schedule: {}, envConfig: {}, currentView: 'worker' }, ownerTokenHash: hashOwnerToken('o') });
  const store = createSessionStore({ storage });
  app = express();
  app.use(express.json());
  app.use(internalAuth(config.internalKey));
  app.use(createInternalRouter(store, config));
});
afterEach(() => rm(root, { recursive: true, force: true }));

describe('internal routes', () => {
  it('401 without the internal key', async () => {
    await request(app).get(`/internal/sessions/${id}/live`).expect(401);
  });
  it('activate loads the session and writes status open + relay info', async () => {
    const res = await request(app).post(`/internal/sessions/${id}/activate`).set('x-internal-key', 'topsecret').expect(200);
    expect(res.body).toMatchObject({ ok: true, relayUrl: 'http://relay:4010', status: 'open' });
  });
  it('live reports active after activate', async () => {
    await request(app).post(`/internal/sessions/${id}/activate`).set('x-internal-key', 'topsecret');
    const res = await request(app).get(`/internal/sessions/${id}/live`).set('x-internal-key', 'topsecret').expect(200);
    expect(res.body).toMatchObject({ ok: true, active: true, participantCount: 0, status: 'open' });
  });
  it('activate on unknown id → 404', async () => {
    await request(app).post(`/internal/sessions/does-not-exist/activate`).set('x-internal-key', 'topsecret').expect(404);
  });
  it('evict sets status close', async () => {
    await request(app).post(`/internal/sessions/${id}/activate`).set('x-internal-key', 'topsecret');
    await request(app).post(`/internal/sessions/${id}/evict`).set('x-internal-key', 'topsecret').expect(200);
    const res = await request(app).get(`/internal/sessions/${id}/live`).set('x-internal-key', 'topsecret').expect(200);
    expect(res.body.status).toBe('close');
  });
});
```

Add `supertest` + `@types/supertest` to `server` devDependencies (`npm i -D supertest @types/supertest` inside `GanttChartEditor/server`).

- [ ] **Step 2: Run to verify it fails** → FAIL (modules missing).

- [ ] **Step 3: Implement `internalAuth.ts`**

```ts
import type { RequestHandler } from 'express';
export function internalAuth(expectedKey: string): RequestHandler {
  return (req, res, next) => {
    if (req.get('x-internal-key') !== expectedKey) { res.status(401).json({ ok: false, error: 'unauthorized' }); return; }
    next();
  };
}
```

- [ ] **Step 4: Implement `routes/internal.ts`**

`activate`: `const ok = await store.activateFromStorage(id);` if `!ok` → 404. `await store.markActivated(id, { relayInstance: config.instanceId, relayUrl: config.publicRelayUrl });` — add `markActivated` to `SessionStore`: writes `status.json` `{ status: currentStatus === 'lock' ? 'lock' : 'open', relayInstance, relayUrl }`. Respond `{ ok:true, relayUrl: config.publicRelayUrl, status }`. `live`: `res.json({ ok:true, ...store.getLive(id) })`. `evict`: `await store.evict(id); await store.markClosed(id); res.json({ ok:true })`.

- [ ] **Step 5: Run to verify pass** → PASS.

- [ ] **Step 6: Commit**

```bash
git add GanttChartEditor/server/src/internalAuth.ts GanttChartEditor/server/src/routes/internal.ts GanttChartEditor/server/src/routes/internal.test.ts GanttChartEditor/server/package.json GanttChartEditor/server/package-lock.json GanttChartEditor/server/src/collab/sessionStore.ts
git commit -m "feat(server): internal activate/live/evict routes with shared-key auth"
```

---

## Task 8: `aca1/yamlIntake.ts` — parse + validate uploaded YAML

**Files:**
- Create: `GanttChartEditor/server/src/aca1/yamlIntake.ts`
- Create: `GanttChartEditor/server/src/aca1/yamlIntake.test.ts`

**Interfaces:**
- Produces:
  ```ts
  export interface IntakeInput { name: string; scheduleYaml?: string; envConfigYaml?: string;
    schedule?: unknown; envConfig?: unknown; currentView?: unknown; }
  export interface IntakeResult { name: string; baseline: SessionBaseline; }
  export class IntakeError extends Error {}           // thrown on invalid input → caller maps to 400
  export function intake(input: IntakeInput): IntakeResult;
  ```
  Rules: `name` required, trimmed, 1–120 chars. Exactly one of (`scheduleYaml`+`envConfigYaml`) or (`schedule`+`envConfig`) must be present. YAML is parsed with `js-yaml` `load`; a parse error → `IntakeError`. `schedule` and `envConfig` must each be non-null objects. `currentView` defaults to `'worker'`; if present must be `'worker'|'device'`.

- [ ] **Step 1: Write the failing test**

```ts
import { describe, it, expect } from 'vitest';
import { intake, IntakeError } from './yamlIntake.js';

describe('intake', () => {
  it('accepts pre-parsed JSON payload (desktop path)', () => {
    const r = intake({ name: 'Plan', schedule: { a: 1 }, envConfig: { b: 2 }, currentView: 'device' });
    expect(r).toEqual({ name: 'Plan', baseline: { schedule: { a: 1 }, envConfig: { b: 2 }, currentView: 'device' } });
  });
  it('parses YAML strings (browser path) and defaults currentView to worker', () => {
    const r = intake({ name: 'Plan', scheduleYaml: 'a: 1\n', envConfigYaml: 'b: 2\n' });
    expect(r.baseline).toEqual({ schedule: { a: 1 }, envConfig: { b: 2 }, currentView: 'worker' });
  });
  it('rejects a missing name', () => { expect(() => intake({ name: '  ', schedule: {}, envConfig: {} })).toThrow(IntakeError); });
  it('rejects when neither YAML nor JSON pair is supplied', () => { expect(() => intake({ name: 'x' })).toThrow(IntakeError); });
  it('rejects malformed YAML', () => { expect(() => intake({ name: 'x', scheduleYaml: ':::', envConfigYaml: 'b: 2' })).toThrow(IntakeError); });
  it('rejects a non-object schedule', () => { expect(() => intake({ name: 'x', schedule: 5, envConfig: {} })).toThrow(IntakeError); });
  it('rejects an invalid currentView', () => { expect(() => intake({ name: 'x', schedule: {}, envConfig: {}, currentView: 'nope' })).toThrow(IntakeError); });
});
```

- [ ] **Step 2: Run to verify it fails** → FAIL.

- [ ] **Step 3: Implement** using `import yaml from 'js-yaml';` (already a dependency of the client but NOT of `server` — add it: `npm i js-yaml @types/js-yaml` inside `GanttChartEditor/server`). Plain hand-rolled validation is fine; `zod` optional.

- [ ] **Step 4: Run to verify pass** → PASS.

- [ ] **Step 5: Commit**

```bash
git add GanttChartEditor/server/src/aca1/yamlIntake.ts GanttChartEditor/server/src/aca1/yamlIntake.test.ts GanttChartEditor/server/package.json GanttChartEditor/server/package-lock.json
git commit -m "feat(server): yaml/json intake for session creation"
```

---

## Task 9: `aca1/aca2Client.ts` — typed fetch wrapper for ACA2 internal API

**Files:**
- Create: `GanttChartEditor/server/src/aca1/aca2Client.ts`
- Create: `GanttChartEditor/server/src/aca1/aca2Client.test.ts`

**Interfaces:**
- Consumes: `AppConfig` (`aca2Url`, `internalKey`).
- Produces:
  ```ts
  export interface Aca2Client {
    activate(id: string): Promise<{ relayUrl: string; status: SessionStatus } | { notFound: true }>;
    live(id: string): Promise<{ active: boolean; participantCount: number; status: SessionStatus } | { unreachable: true }>;
    evict(id: string): Promise<void>;  // swallows network errors
  }
  export function createAca2Client(config: Pick<AppConfig, 'aca2Url' | 'internalKey'>, fetchImpl?: typeof fetch): Aca2Client;
  ```
  `live` returns `{ unreachable: true }` on any network error (ACA2 asleep) — callers treat that as "not active".

- [ ] **Step 1: Write the failing test** with an injected fake `fetch`:

```ts
import { describe, it, expect } from 'vitest';
import { createAca2Client } from './aca2Client.js';

const cfg = { aca2Url: 'http://aca2:4010', internalKey: 'k' };
const ok = (body: unknown) => ({ ok: true, status: 200, json: async () => body } as Response);

describe('aca2Client', () => {
  it('activate returns relayUrl+status', async () => {
    const c = createAca2Client(cfg, (async () => ok({ ok: true, relayUrl: 'http://r', status: 'open' })) as any);
    expect(await c.activate('s1')).toEqual({ relayUrl: 'http://r', status: 'open' });
  });
  it('activate maps 404 to notFound', async () => {
    const c = createAca2Client(cfg, (async () => ({ ok: false, status: 404, json: async () => ({ ok: false }) })) as any);
    expect(await c.activate('s1')).toEqual({ notFound: true });
  });
  it('live maps a thrown fetch to unreachable', async () => {
    const c = createAca2Client(cfg, (async () => { throw new Error('ECONNREFUSED'); }) as any);
    expect(await c.live('s1')).toEqual({ unreachable: true });
  });
  it('sends the internal key header', async () => {
    let seen: any;
    const c = createAca2Client(cfg, (async (_u: string, init: any) => { seen = init; return ok({ ok: true, active: true, participantCount: 0, status: 'open' }); }) as any);
    await c.live('s1');
    expect(seen.headers['x-internal-key']).toBe('k');
  });
});
```

- [ ] **Step 2: Run to verify it fails** → FAIL.

- [ ] **Step 3: Implement** with `globalThis.fetch` default (Node 20 has it). URL join: `${aca2Url}/internal/sessions/${encodeURIComponent(id)}/…`.

- [ ] **Step 4: Run to verify pass** → PASS.

- [ ] **Step 5: Commit**

```bash
git add GanttChartEditor/server/src/aca1/aca2Client.ts GanttChartEditor/server/src/aca1/aca2Client.test.ts
git commit -m "feat(server): aca1→aca2 internal client with unreachable handling"
```

---

## Task 10: `aca1/sessionApi.ts` + `aca1/app.ts` — the session API

**Files:**
- Create: `GanttChartEditor/server/src/aca1/sessionApi.ts`
- Create: `GanttChartEditor/server/src/aca1/app.ts`
- Create: `GanttChartEditor/server/src/aca1/app.test.ts`

**Interfaces:**
- Consumes: `intake` (Task 8), `Aca2Client` (Task 9), `persistence` (Task 4), `StorageClient` (Task 2), `AppConfig` (Task 3).
- Produces:
  ```ts
  export function createSessionApiRouter(deps: {
    storage: StorageClient; aca2: Aca2Client; config: AppConfig;
  }): Router;
  export function createAca1App(deps: { storage: StorageClient; aca2: Aca2Client; config: AppConfig }): express.Express;
  ```
  Endpoints exactly as the ACA1 table in the Design reference. `GET /api/sessions` builds each `SessionSummary` from `meta.json` + `status.json`; `participantCount` from `aca2.live(id)` (→ `null` on `{unreachable:true}` or when `status==='close'`). `POST /api/sessions/:id/open`: read `status.json`; if `close` (or `live` says not active) → `aca2.activate` (404 → 404); else use stored `relayUrl`; respond `{ ok, sessionId, relayUrl, status }`. `DELETE`: compare `hashOwnerToken(req.get('x-owner-token'))` to `meta.ownerTokenHash` → 403 on mismatch → `aca2.evict` → `deleteSessionRecord`.

- [ ] **Step 1: Write the failing test** (supertest; fake `Aca2Client` object; real fs storage temp dir)

```ts
// cases:
it('POST /api/sessions (json) creates a session and returns sessionId + ownerToken');
it('POST /api/sessions (multipart) parses two yaml files');            // use .attach('schedule', Buffer.from('a: 1'), 'Schedule.yaml')
it('POST /api/sessions with a bad body → 400');
it('GET /api/sessions lists created sessions with status "close" and participantCount null');
it('GET /api/sessions/:id → 404 for unknown');
it('POST /api/sessions/:id/open on a close session calls aca2.activate and returns its relayUrl + status open');
it('POST /api/sessions/:id/open on an open session returns the stored relayUrl without calling activate');
it('DELETE /api/sessions/:id without the owner token → 403');
it('DELETE /api/sessions/:id with the owner token removes the record and calls aca2.evict');
it('GET /api/health → { ok: true, role: "aca1" }');
```

Fake `Aca2Client`: `{ activate: vi.fn(async () => ({ relayUrl: 'http://relay:4010', status: 'open' })), live: vi.fn(async () => ({ unreachable: true })), evict: vi.fn(async () => {}) }`.

- [ ] **Step 2: Run to verify it fails** → FAIL.

- [ ] **Step 3: Implement `sessionApi.ts`** — one `Router`. Use `multer({ storage: multer.memoryStorage(), limits: { fileSize: 5 * 1024 * 1024, files: 2 } }).fields([{ name: 'schedule', maxCount: 1 }, { name: 'envConfig', maxCount: 1 }])` on `POST /api/sessions`, then branch on `req.is('multipart/form-data')`. Add `multer` + `@types/multer` to `server` devDependencies/deps.

- [ ] **Step 4: Implement `app.ts`** — `express()`, `cors()` (allow `config.webOrigin` or reflect when null), `express.json({ limit: '10mb' })`, mount the router, `GET /api/health`.

- [ ] **Step 5: Run to verify pass** → PASS.

- [ ] **Step 6: Commit**

```bash
git add GanttChartEditor/server/src/aca1/ GanttChartEditor/server/package.json GanttChartEditor/server/package-lock.json
git commit -m "feat(server): aca1 session API (create/list/open/delete)"
```

---

## Task 11: `aca1/sweep.ts` — absolute-TTL + orphan-open backstop

**Files:**
- Create: `GanttChartEditor/server/src/aca1/sweep.ts`
- Create: `GanttChartEditor/server/src/aca1/sweep.test.ts`

**Interfaces:**
- Consumes: `persistence` (Task 4), `Aca2Client` (Task 9), `AppConfig`.
- Produces:
  ```ts
  export async function runSweepOnce(deps: {
    storage: StorageClient; aca2: Aca2Client; config: AppConfig; now?: number;
  }): Promise<{ deleted: string[]; closed: string[] }>;
  export function startSweep(deps: {...}): () => void;   // setInterval wrapper, returns stop()
  ```
  Rules: for each session — if `status.json.status === 'close'` and `now - lastActivityAt > config.absoluteSessionMaxMs` → `deleteSessionRecord` (push to `deleted`). If `status === 'open' | 'lock'` and `aca2.live(id)` is `{unreachable:true}` or `active:false` **and** `now - lastActivityAt > config.idleSessionTimeoutMs` → `writeStatus(id, { status: 'close', relayInstance: null, relayUrl: null })` (push to `closed`).

- [ ] **Step 1: Write the failing test** — seed sessions with hand-written `status.json` via `writeStatus`, fake `aca2.live`, call `runSweepOnce({ now: <far future> })`, assert `deleted` / `closed`.

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement.**

- [ ] **Step 4: Run → PASS.**

- [ ] **Step 5: Commit**

```bash
git add GanttChartEditor/server/src/aca1/sweep.ts GanttChartEditor/server/src/aca1/sweep.test.ts
git commit -m "feat(server): aca1 sweep for stale-close cleanup and orphaned-open backstop"
```

---

## Task 12: `index.ts` — role switch (local / aca1 / aca2)

**Files:**
- Modify: `GanttChartEditor/server/src/index.ts`
- Modify: `GanttChartEditor/server/src/routes/collab.ts` (only mount in `local`)
- Create: `GanttChartEditor/server/src/index.smoke.test.ts`

**Interfaces:**
- Consumes: everything above.
- Produces: three boot paths. `local` = today's behaviour (Express + all routes + `createCollabSocketServer` on one port, `SERVE_STATIC_DIR` static hosting, `save-files`/`network-info` present) — but the session store is now `createSessionStore({ storage: makeStorage(config) })` and session creation in `routes/collab.ts` switches to `persistence.createSessionRecord`. `aca1` = `createAca1App(...).listen(port)` + `startSweep(...)`. `aca2` = Express with `internalAuth` + `createInternalRouter` + `GET /api/health` `{role:'aca2', instance}` + `createCollabSocketServer(httpServer, store, config)`; **no** `save-files` / `network-info` / static hosting / `collabRouter`.

- [ ] **Step 1: Write the failing smoke test**

```ts
import { describe, it, expect } from 'vitest';
import { loadConfig } from './config.js';

// Boot each role on an ephemeral port via a helper exported from index.ts:
//   export async function startServer(config: AppConfig): Promise<{ port: number; close: () => Promise<void> }>
import { startServer } from './index.js';

describe('role switch', () => {
  it('aca1 serves /api/health with role aca1 and NOT /internal/*', async () => {
    const srv = await startServer(loadConfig({ ROLE: 'aca1', PORT: '0', INTERNAL_KEY: 'k', ACA2_URL: 'http://x', MOCK_BLOB_DIR: '/tmp/gantt-roleswitch-a' }));
    const h = await fetch(`http://localhost:${srv.port}/api/health`).then((r) => r.json());
    expect(h).toMatchObject({ ok: true, role: 'aca1' });
    const i = await fetch(`http://localhost:${srv.port}/internal/sessions/x/live`);
    expect(i.status).toBe(404);
    await srv.close();
  });

  it('aca2 serves /internal/* (401 without key) and /api/health role aca2', async () => {
    const srv = await startServer(loadConfig({ ROLE: 'aca2', PORT: '0', INTERNAL_KEY: 'k', MOCK_BLOB_DIR: '/tmp/gantt-roleswitch-b' }));
    expect((await fetch(`http://localhost:${srv.port}/internal/sessions/x/live`)).status).toBe(401);
    expect(await fetch(`http://localhost:${srv.port}/api/health`).then((r) => r.json())).toMatchObject({ role: 'aca2' });
    await srv.close();
  });
});
```

- [ ] **Step 2: Run → FAIL** (no `startServer` export).

- [ ] **Step 3: Refactor `index.ts`** to export `startServer(config)` returning `{ port, close }` (resolve `port` from `httpServer.address()`), and a bottom `if (import.meta.url === ...) startServer(loadConfig())`. Branch on `config.role`.

- [ ] **Step 4: Run → PASS**, then full suite: `cd GanttChartEditor/server && npx vitest run && npx tsc --noEmit`.

- [ ] **Step 5: Manual local-mode check**

Run: `cd GanttChartEditor && npm run dev:all`, open the app, start a session, drag a bar in a second browser tab — still works exactly as before.

- [ ] **Step 6: Commit**

```bash
git add GanttChartEditor/server/src/index.ts GanttChartEditor/server/src/routes/collab.ts GanttChartEditor/server/src/index.smoke.test.ts
git commit -m "feat(server): ROLE-based boot (local | aca1 | aca2)"
```

---

## Task 13: Docker image + Compose + `dev:mock` + Postman + full integration test

**Files:**
- Create: `GanttChartEditor/server/Dockerfile`, `GanttChartEditor/server/.dockerignore`
- Create: `GanttChartEditor/server/docker-compose.mock.yml`
- Create: `GanttChartEditor/server/postman/OnlineCollabAzure.postman_collection.json`
- Create: `GanttChartEditor/server/src/__integration__/localFlow.test.ts`
- Modify: `GanttChartEditor/package.json` (add `dev:mock`)
- Create: `GanttChartEditor/server/MOCK_AZURE.md`

**Interfaces:**
- Consumes: `startServer` (Task 12), `createFsStorage`, `persistence`, `socket.io-client`.

- [ ] **Step 1: Write the failing integration test**

`localFlow.test.ts` — boots `startServer` for `aca1` (port 0) and `aca2` (port 0) sharing one temp `MOCK_BLOB_DIR`, wires `aca1`'s `ACA2_URL` to the `aca2` port and `PUBLIC_RELAY_URL` to the same. Then:

```ts
// 1. POST /api/sessions (json) → { sessionId, ownerToken }
// 2. GET  /api/sessions → one entry, status 'close'
// 3. POST /api/sessions/:id/open → { relayUrl, status: 'open' }
// 4. socket A connects to relayUrl, join {role:'edit', ownerToken}; expect sync-init ok, status 'open'
// 5. socket B connects, join {role:'edit'}; expect sync-init with baseline
// 6. A emits action {type:'SET_SCHEDULE', payload:{v:1}} → B receives 'action'
// 7. A emits 'lock' → both receive session-status {status:'lock'}; GET /api/sessions/:id shows status 'lock'
// 8. B emits action → A does NOT receive it (200ms window)
// 9. A emits 'unlock' → B action now propagates again
// 10. A and B disconnect → poll GET /api/sessions/:id until status 'close' (<=2s)
// 11. read <blob>/sessions/<id>/log.json from disk → contains the SET_SCHEDULE action
// 12. POST /api/sessions/:id/open again → status 'open'; new socket join → sync-init actions include SET_SCHEDULE (replayed from storage)
// 13. DELETE /api/sessions/:id with x-owner-token → 200; GET /api/sessions → empty
```

- [ ] **Step 2: Run → FAIL** (behaviour gaps surface here).

- [ ] **Step 3: Fix any gaps** found by the integration test in the Task 5–12 files until green. Do not weaken the test.

- [ ] **Step 4: Run → PASS**

Run: `cd GanttChartEditor/server && npx vitest run src/__integration__/localFlow.test.ts`

- [ ] **Step 5: Write the `Dockerfile`**

```dockerfile
FROM node:20-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY tsconfig.json ./
COPY src ./src
RUN npm run build

FROM node:20-alpine
WORKDIR /app
ENV NODE_ENV=production
COPY package.json package-lock.json ./
RUN npm ci --omit=dev
COPY --from=build /app/dist ./dist
EXPOSE 4000
CMD ["node", "dist/index.js"]
```

`.dockerignore`: `node_modules`, `dist`, `*.test.ts`, `postman`, `mock-blob`, `MOCK_AZURE.md`.

- [ ] **Step 6: Write `docker-compose.mock.yml`**

```yaml
services:
  aca1:
    build: .
    environment:
      ROLE: aca1
      PORT: "4000"
      STORAGE: fs
      MOCK_BLOB_DIR: /data
      INTERNAL_KEY: mock-internal-key
      ACA2_URL: http://aca2:4010
      WEB_ORIGIN: http://localhost:5173
    ports: ["4000:4000"]
    volumes: ["./mock-blob:/data"]
  aca2:
    build: .
    environment:
      ROLE: aca2
      PORT: "4010"
      STORAGE: fs
      MOCK_BLOB_DIR: /data
      INTERNAL_KEY: mock-internal-key
      PUBLIC_RELAY_URL: http://localhost:4010
      WEB_ORIGIN: http://localhost:5173
    ports: ["4010:4010"]
    volumes: ["./mock-blob:/data"]
```

- [ ] **Step 7: Verify the Docker path**

Run: `cd GanttChartEditor/server && docker compose -f docker-compose.mock.yml up --build -d`
Then: `curl -s localhost:4000/api/health` → `{"ok":true,"role":"aca1"}`; `curl -s -X POST localhost:4010/internal/sessions/x/live` → 401.
Then: `docker compose -f docker-compose.mock.yml down`.

- [ ] **Step 8: Add `dev:mock` to `GanttChartEditor/package.json`**

```json
"dev:mock": "concurrently -n aca1,aca2,web \"cross-env ROLE=aca1 PORT=4000 STORAGE=fs MOCK_BLOB_DIR=./mock-blob INTERNAL_KEY=mock-internal-key ACA2_URL=http://localhost:4010 WEB_ORIGIN=http://localhost:5173 npm --prefix server run dev\" \"cross-env ROLE=aca2 PORT=4010 STORAGE=fs MOCK_BLOB_DIR=./mock-blob INTERNAL_KEY=mock-internal-key PUBLIC_RELAY_URL=http://localhost:4010 WEB_ORIGIN=http://localhost:5173 npm --prefix server run dev\" \"vite\""
```

Add `cross-env` to root devDependencies (Windows-safe env vars). `mock-blob/` → add to `GanttChartEditor/.gitignore`.

- [ ] **Step 9: Postman collection**

`postman/OnlineCollabAzure.postman_collection.json` with a `baseUrl` variable (`http://localhost:4000`) and requests: `Health`, `Create session (JSON)`, `Create session (2 YAML)` (form-data, file bodies from `../Test_data/Schedule.yaml` + `EnvConfig.yaml`), `List sessions`, `Get session`, `Open session` (saves `relayUrl` to a collection var via a test script), `Delete session` (uses `{{ownerToken}}` captured by the create request's test script). Include a short `MOCK_AZURE.md` explaining: `npm run dev:mock`, or `docker compose -f server/docker-compose.mock.yml up`, then import the Postman collection; lock/unlock are Socket.IO events (not in Postman) — covered by `localFlow.test.ts`.

- [ ] **Step 10: Full server suite + typecheck**

Run: `cd GanttChartEditor/server && npx vitest run && npx tsc --noEmit` → all green.

- [ ] **Step 11: Commit**

```bash
git add GanttChartEditor/server/Dockerfile GanttChartEditor/server/.dockerignore GanttChartEditor/server/docker-compose.mock.yml GanttChartEditor/server/postman/ GanttChartEditor/server/src/__integration__/ GanttChartEditor/server/MOCK_AZURE.md GanttChartEditor/package.json GanttChartEditor/package-lock.json GanttChartEditor/.gitignore
git commit -m "feat(server): mock-azure harness — docker compose, dev:mock, postman, full local integration test"
```

---

## Self-review (against this plan's own scope — the local backend milestone)

- **Storage abstraction:** Task 2 (`StorageClient` + `fsStorage`), Task 4 (records). Blob impl is deferred to Phase 4 (Appendix C) — the interface makes it a drop-in. ✅
- **ACA2 persistence — load baseline + history / flush on idle & last leave / evict / scale-to-zero:** Tasks 5, 6 (+ integration checks 10–12 in Task 13). "Scale to zero" locally = process idle after evict; the real ACA `minReplicas:0` is Phase 4. ✅
- **Lock/unlock on ACA2, owner must join first, `session-status` broadcast, drop actions while locked:** Task 6 + integration steps 7–9. ✅
- **`status.json` `open`/`lock`/`close` state machine:** Tasks 4–7, 11; transitions exercised end-to-end in Task 13. ✅
- **ACA1 list/create(2-YAML + JSON)/open/delete:** Tasks 8–10. ✅
- **One image, `ROLE` switch:** Task 12. Local mode preserved — Task 12 Step 5 + Step 3 keep `save-files`/`network-info`/static hosting only in `local`. ✅
- **Dockerfile + Compose + Postman + integration test + `dev:mock`:** Task 13. ✅
- **Internal-key guard on `/internal/*`:** Task 7 (`internalAuth`). Full rate-limiting/size caps are Phase 3 (Appendix B) — a 5 MB `multer` cap is set in Task 10 as a floor. ✅
- **Placeholder scan:** no `TBD`/`TODO`/"handle edge cases" — each step has concrete code or a concrete command. The one soft spot is `fsStorage.listPrefix` (Task 2 Step 4 note) — deliberately flagged with the exact test cases that pin the behaviour. ✅
- **Type consistency:** `SessionStore` grows three methods across tasks — `markClosed` (Task 6), `markActivated` (Task 7), and the core set (Task 5). When implementing Task 5, add all three signatures to the interface up front from this list so later tasks only fill bodies:
  `activateFromStorage, flush, flushAll, evict, getLive, ownerTokenHash, getSession, appendAction, addParticipant, removeParticipant, participantCount, setLocked, isLoaded, markClosed, markActivated`. ✅

---

## Appendix A — Follow-on plan: Web client (Phase 2)

Own plan file: `GanttChartEditor_OnlineCollabAzure_ACA_ClientPlan<date>.md`. Task outline:

1. `collabService.ts`: add `listSessions()`, `createSessionFromYaml(name, File, File)`, `createSessionFromState(name, baseline)`, `openSession(id)` → `{ relayUrl, status }`; `joinCollabRoom` takes an explicit `relayUrl` (drop `getSocketOrigin`/`parseSessionOrigin`/`fetchCollabLink`/`network-info`). New env `VITE_ACA1_URL`.
2. `SessionListPage` component: table of `SessionSummary` (name, status badge open/lock/close, participant count), "Open" per row, "Create session" opener. Poll `listSessions()` every ~5 s while mounted.
3. Create-session form: `name` + two `<input type="file" accept=".yaml,.yml">`; on submit → `createSessionFromYaml` → store `ownerToken` in memory (and `sessionStorage` keyed by id) → auto-open.
4. Rework `SessionDialog.tsx`: keep "start from current schedule" (desktop) → `createSessionFromState`; remove the LAN link rows and the "Join" link-paste tab; the join path is now the list.
5. `AppContext.tsx`: `startCollabSession`/`joinCollabSession` call the new open flow; hold `ownerToken`; add `lockSession()` / `unlockSession()` that emit the socket events; handle inbound `session-status` → set a `state.session.status` and flip read-only.
6. Locked UI: reuse the existing viewer/read-only gating (branch `live-collab-edit` work) driven by `status === 'lock' || role === 'view'`; add a banner.
7. Cold-start: `openSession` retries on `502/503`/timeout with backoff and a "waking the session…" spinner.
8. Tests: Jest+RTL for `SessionListPage` and create flow (mock `fetch`); extend Cypress viewer-gating specs with a locked-session case.

## Appendix B — Follow-on plan: Security & limits (Phase 3)

1. `express-rate-limit` on `POST /api/sessions` — 10/hour/IP (`aca1/app.ts`).
2. Config-driven caps: `MAX_CONCURRENT_SESSIONS` (default 100) checked in `POST /api/sessions`; `MAX_PARTICIPANTS` (default 25) checked in `collabSocket` `join`; `multer` `fileSize` from config (default 5 MB).
3. `maxHttpBufferSize` on the Socket.IO server ← config (default 1 MB) to match Web-PubSub-compatible sizing.
4. `internalAuth` already in place (Task 7) — add a startup assertion that `INTERNAL_KEY` is ≥ 16 chars in non-local roles.
5. Owner token: already hashed at rest (Task 4). Add constant-time compare in `DELETE` and in the socket `lock`/`unlock` owner check (`crypto.timingSafeEqual`).
6. Secrets: document the env set; in Azure, `INTERNAL_KEY` and `BLOB_CONNECTION_STRING` become Key Vault secret refs (Appendix C).
7. CORS: in `aca1`/`aca2` cloud roles, `WEB_ORIGIN` is required (no reflect-any); add a test.

## Appendix C — Follow-on plan: Azure deploy (Phase 4, needs Azure access)

1. `blobStorage.ts`: implement `StorageClient` on `@azure/storage-blob` (`BlockBlobClient` per key; `listPrefix` via `containerClient.listBlobsByHierarchy`/flat with prefix; `deletePrefix` = list+delete). Add `@azure/storage-blob` to `server` deps. Contract test: run the **same** `fsStorage.test.ts` suite against an Azurite container.
2. `infra/deploy.sh` (`az` CLI): resource group, ACA environment, ACR (`az acr build -r <acr> -t gantt-collab:$(git rev-parse --short HEAD) ./server`), Storage account + `sessions` container, Key Vault + `internal-key` secret.
3. `az containerapp create` ACA1 — image, `--min-replicas 1 --max-replicas 2`, env (`ROLE=aca1`, `STORAGE=blob`, `BLOB_*`, `INTERNAL_KEY=secretref:internal-key`, `ACA2_URL=<aca2 internal FQDN>`, `WEB_ORIGIN`), external ingress `:4000`.
4. `az containerapp create` ACA2 — same image, `--min-replicas 0 --max-replicas 5`, `--ingress external --transport auto`, `az containerapp ingress sticky-sessions set --affinity sticky`, env (`ROLE=aca2`, `PUBLIC_RELAY_URL=https://<aca2 FQDN>`, `STORAGE=blob`, …). Internal-only ingress is *not* usable (browsers hit ACA2 directly) — protect `/internal/*` with the key and, optionally, an IP allow-list for the ACA1 subnet.
5. Static Web Apps: `az staticwebapp create`, deploy the Vite build, set `VITE_ACA1_URL` at build time; custom domain + managed cert on ACA1 and the SWA.
6. GitHub Actions: build+push image, `az containerapp update` both apps, deploy SWA — on push to `main`.
7. Smoke: `curl` health on both, run a trimmed `localFlow` against the deployed URLs (or note it is blocked by the office VPN and hand to someone with access).

## Appendix D — Follow-on: Docs (Phase 5)

- Update `GanttChartEditor_OnlineCollabAzure_Presentation20260904*.md` — replace the single-relay diagrams with the ACA1/ACA2 split, add the `open`/`lock`/`close` state diagram, drop the share-link section, describe the session-list entry point.
- New `GanttChartEditor_OnlineCollabAzure_Runbook<date>.md` — env var reference, how to force-close or delete a stuck session (Postman `DELETE`), cost levers (`minReplicas`, sweep TTLs), when to add the Socket.IO Redis adapter (multi-replica ACA2).

---

## Execution handoff

Plan complete. Two execution options:

**1. Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks.
**2. Inline Execution** — execute tasks here in batches with checkpoints.
