# GanttChartEditor Live Collaborative Editing — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let multiple people edit the same GanttChartEditor schedule live, through the existing custom Gantt UI, by wrapping reducer actions and relaying them through a small session server — with solo use completely unaffected.

**Architecture:** A new session/action-log module on the existing `server/src/index.ts` (Express + Socket.IO, port 3010) replaces the old snapshot-only "Live View" broadcast. The client wraps its single `dispatch` function once, centrally, in `AppContext.tsx`: syncable (schedule-mutating) actions are sent to the server after being applied locally; incoming remote actions are applied directly through the raw `useReducer` dispatch so they never echo back. Joining works both via a link opened in a plain browser tab and via pasting that same link into the app. Electron's shutdown wiring is changed so a host closing their window doesn't kill the server process the session depends on.

**Tech Stack:** React 19 + TypeScript (client), Express + Socket.IO (server, already a dependency), Electron 32 (desktop shell), Jest + ts-jest (client tests, existing), Vitest (new — server has no test runner yet).

**Reference documents:**
- Design spec: `documents/GanttChartEditor/GanttChartEditor_LiveCollabEdit_Design20260826.md`
- Background: `documents/GanttChartEditor/GanttChartEditor_RealTime_Collaboration_FullReference.md`

## Global Constraints

- No login/accounts — a typed display name only, for presence/attribution.
- No persistence beyond server process memory — a killed server process loses anything not already `Save`d locally by some participant.
- No field-level locking or CRDT — last-applied-wins, ordered by arrival at the server.
- No outsourced/cloud server for v1 — the collab server stays self-hosted on whichever machine runs it (the host's own PC by default).
- Solo use (no session) must work exactly as it does today — a session is opt-in and additive, never required.
- Existing read-only "Live View" feature (`ShareViewButton`, `ShareViewDialog`, `viewBroadcastService.ts`, `useHostBroadcast`, `useViewerSync`, `server/src/collab/viewBroadcast.ts`) is retired and replaced by the unified session flow — do not keep both running in parallel.
- Collaboration UI lives inside the ファイル/編集/表示/ヘルプ menu bar (`MenuBar.tsx`) as a new "共同編集" menu — not a standalone corner button.
- All new user-facing text follows the existing app's Japanese-first convention in `src/config/uiText.ts` (see existing keys for tone/register).

---

### Task 1: Server — session store (pure, in-memory)

**Files:**
- Create: `GanttChartEditor/server/src/collab/sessionStore.ts`
- Create: `GanttChartEditor/server/src/collab/sessionStore.test.ts`
- Modify: `GanttChartEditor/server/package.json` (add `vitest` devDependency + `test` script)

**Interfaces:**
- Produces: `SessionBaseline { schedule: unknown; envConfig: unknown; currentView: 'worker' | 'device' }`, `LoggedAction { seq: number; type: string; payload: unknown }`, `SessionParticipant { id: string; name: string; role: 'edit' | 'view' }`, and functions `createSession(baseline): string`, `getSession(id): { baseline, actions, participants } | null`, `appendAction(sessionId, type, payload): LoggedAction | null`, `addParticipant(sessionId, participantId, name, role): SessionParticipant[] | null`, `removeParticipant(sessionId, participantId): SessionParticipant[] | null`, `sweepIdleSessions(maxIdleMs, now?): number`, `_resetForTests(): void`.

- [ ] **Step 1: Add vitest to the server package**

```bash
cd GanttChartEditor/server && npm install --save-dev vitest
```

Edit `GanttChartEditor/server/package.json` scripts block to add:
```json
"test": "vitest run",
```
(keep the existing `dev`, `build`, `start` scripts as-is).

- [ ] **Step 2: Write the failing tests**

Create `GanttChartEditor/server/src/collab/sessionStore.test.ts`:

```ts
import { describe, it, expect, beforeEach } from 'vitest';
import {
  createSession, getSession, appendAction, addParticipant, removeParticipant,
  sweepIdleSessions, _resetForTests,
} from './sessionStore.js';

const BASELINE = { schedule: { foo: 'bar' }, envConfig: { baz: 1 }, currentView: 'worker' as const };

beforeEach(() => _resetForTests());

describe('createSession / getSession', () => {
  it('creates a session with the given baseline and no actions or participants', () => {
    const id = createSession(BASELINE);
    const session = getSession(id);
    expect(session).not.toBeNull();
    expect(session?.baseline).toEqual(BASELINE);
    expect(session?.actions).toEqual([]);
    expect(session?.participants).toEqual([]);
  });

  it('returns null for an unknown session id', () => {
    expect(getSession('does-not-exist')).toBeNull();
  });
});

describe('appendAction', () => {
  it('assigns increasing sequence numbers and stores the action', () => {
    const id = createSession(BASELINE);
    const a1 = appendAction(id, 'SET_SCHEDULE', { hello: 'world' });
    const a2 = appendAction(id, 'UPDATE_PLAN_RANGE', { startDate: '2026-01-01', endDate: '2026-01-31' });
    expect(a1).toEqual({ seq: 0, type: 'SET_SCHEDULE', payload: { hello: 'world' } });
    expect(a2).toEqual({ seq: 1, type: 'UPDATE_PLAN_RANGE', payload: { startDate: '2026-01-01', endDate: '2026-01-31' } });
    expect(getSession(id)?.actions).toEqual([a1, a2]);
  });

  it('returns null for an unknown session id', () => {
    expect(appendAction('does-not-exist', 'SET_SCHEDULE', {})).toBeNull();
  });
});

describe('addParticipant / removeParticipant', () => {
  it('adds a participant and returns the full list', () => {
    const id = createSession(BASELINE);
    const list = addParticipant(id, 'p1', 'Alice', 'edit');
    expect(list).toEqual([{ id: 'p1', name: 'Alice', role: 'edit' }]);
  });

  it('removes a participant and returns the remaining list', () => {
    const id = createSession(BASELINE);
    addParticipant(id, 'p1', 'Alice', 'edit');
    addParticipant(id, 'p2', 'Bob', 'view');
    const list = removeParticipant(id, 'p1');
    expect(list).toEqual([{ id: 'p2', name: 'Bob', role: 'view' }]);
  });

  it('returns null for an unknown session id', () => {
    expect(addParticipant('does-not-exist', 'p1', 'Alice', 'edit')).toBeNull();
    expect(removeParticipant('does-not-exist', 'p1')).toBeNull();
  });
});

describe('sweepIdleSessions', () => {
  it('removes sessions with zero participants past the idle threshold', () => {
    const id = createSession(BASELINE);
    const removed = sweepIdleSessions(1000, Date.now() + 2000);
    expect(removed).toBe(1);
    expect(getSession(id)).toBeNull();
  });

  it('keeps sessions that still have participants', () => {
    const id = createSession(BASELINE);
    addParticipant(id, 'p1', 'Alice', 'edit');
    const removed = sweepIdleSessions(1000, Date.now() + 2000);
    expect(removed).toBe(0);
    expect(getSession(id)).not.toBeNull();
  });

  it('keeps sessions inside the idle threshold', () => {
    const id = createSession(BASELINE);
    const removed = sweepIdleSessions(60_000, Date.now() + 1000);
    expect(removed).toBe(0);
    expect(getSession(id)).not.toBeNull();
  });
});
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd GanttChartEditor/server && npm test`
Expected: FAIL — `sessionStore.ts` doesn't exist yet.

- [ ] **Step 4: Implement `sessionStore.ts`**

Create `GanttChartEditor/server/src/collab/sessionStore.ts`:

```ts
import { randomUUID } from 'node:crypto';

// Pure, in-memory session/action-log store — no I/O, no Socket.IO here, so
// this can be unit-tested directly. Deliberately dumb: it stores and orders
// actions, it never interprets them — the reducer (shared, already tested,
// on every client) is the single source of truth for what an action does.

export interface SessionBaseline {
  schedule: unknown;
  envConfig: unknown;
  currentView: 'worker' | 'device';
}

export interface LoggedAction {
  seq: number;
  type: string;
  payload: unknown;
}

export interface SessionParticipant {
  id: string;
  name: string;
  role: 'edit' | 'view';
}

interface CollabSession {
  id: string;
  baseline: SessionBaseline;
  actions: LoggedAction[];
  participants: Map<string, SessionParticipant>;
  nextSeq: number;
  lastActivityAt: number;
}

const sessions = new Map<string, CollabSession>();

export function createSession(baseline: SessionBaseline): string {
  const id = randomUUID();
  sessions.set(id, {
    id,
    baseline,
    actions: [],
    participants: new Map(),
    nextSeq: 0,
    lastActivityAt: Date.now(),
  });
  return id;
}

export function getSession(
  id: string,
): { baseline: SessionBaseline; actions: LoggedAction[]; participants: SessionParticipant[] } | null {
  const session = sessions.get(id);
  if (!session) return null;
  return { baseline: session.baseline, actions: session.actions, participants: [...session.participants.values()] };
}

export function appendAction(sessionId: string, type: string, payload: unknown): LoggedAction | null {
  const session = sessions.get(sessionId);
  if (!session) return null;
  const action: LoggedAction = { seq: session.nextSeq, type, payload };
  session.nextSeq += 1;
  session.actions.push(action);
  session.lastActivityAt = Date.now();
  return action;
}

export function addParticipant(
  sessionId: string, participantId: string, name: string, role: 'edit' | 'view',
): SessionParticipant[] | null {
  const session = sessions.get(sessionId);
  if (!session) return null;
  session.participants.set(participantId, { id: participantId, name, role });
  session.lastActivityAt = Date.now();
  return [...session.participants.values()];
}

export function removeParticipant(sessionId: string, participantId: string): SessionParticipant[] | null {
  const session = sessions.get(sessionId);
  if (!session) return null;
  session.participants.delete(participantId);
  session.lastActivityAt = Date.now();
  return [...session.participants.values()];
}

export function sweepIdleSessions(maxIdleMs: number, now = Date.now()): number {
  let removed = 0;
  for (const [id, session] of sessions) {
    if (session.participants.size === 0 && now - session.lastActivityAt > maxIdleMs) {
      sessions.delete(id);
      removed += 1;
    }
  }
  return removed;
}

/** Test-only: clears all sessions so tests don't leak state into each other. */
export function _resetForTests(): void {
  sessions.clear();
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd GanttChartEditor/server && npm test`
Expected: PASS (all `sessionStore.test.ts` cases green).

- [ ] **Step 6: Commit**

```bash
git add GanttChartEditor/server/package.json GanttChartEditor/server/package-lock.json GanttChartEditor/server/src/collab/sessionStore.ts GanttChartEditor/server/src/collab/sessionStore.test.ts
git commit -m "feat(server): add in-memory collab session store"
```

---

### Task 2: Server — Socket.IO session relay + REST create endpoint

**Files:**
- Create: `GanttChartEditor/server/src/collab/collabSocket.ts`
- Create: `GanttChartEditor/server/src/collab/collabSocket.test.ts`
- Create: `GanttChartEditor/server/src/routes/collab.ts`
- Modify: `GanttChartEditor/server/src/index.ts`
- Delete: `GanttChartEditor/server/src/collab/viewBroadcast.ts`

**Interfaces:**
- Consumes: `sessionStore.ts` from Task 1 (`createSession`, `getSession`, `appendAction`, `addParticipant`, `removeParticipant`, `sweepIdleSessions`).
- Produces: `createCollabSocketServer(httpServer: HttpServer): Server` (mounted at Socket.IO path `/collab/socket.io`); `collabRouter` (Express `Router`, exposes `POST /collab/sessions`); events used by the client in Task 5: server emits `sync-init`, `action`, `presence`; server listens for `join`, `action`, `leave`.

- [ ] **Step 1: Write the failing integration test**

Create `GanttChartEditor/server/src/collab/collabSocket.test.ts`:

```ts
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { createServer, Server as HttpServer } from 'node:http';
import { AddressInfo } from 'node:net';
import { io as ioClient, Socket as ClientSocket } from 'socket.io-client';
import { createCollabSocketServer } from './collabSocket.js';
import { createSession, _resetForTests } from './sessionStore.js';

const BASELINE = { schedule: { assignments: [] }, envConfig: { workers: [] }, currentView: 'worker' as const };

let httpServer: HttpServer;
let port: number;

beforeEach(async () => {
  _resetForTests();
  httpServer = createServer();
  createCollabSocketServer(httpServer);
  await new Promise<void>(resolve => httpServer.listen(0, resolve));
  port = (httpServer.address() as AddressInfo).port;
});

afterEach(async () => {
  await new Promise<void>(resolve => httpServer.close(() => resolve()));
});

function connect(): ClientSocket {
  return ioClient(`http://localhost:${port}`, { path: '/collab/socket.io', transports: ['websocket'] });
}

describe('join', () => {
  it('replies with the baseline, empty action log, and participant list for a fresh session', async () => {
    const sessionId = createSession(BASELINE);
    const client = connect();
    const syncInit = await new Promise<any>(resolve => {
      client.on('connect', () => client.emit('join', { sessionId, name: 'Alice', role: 'edit' }));
      client.on('sync-init', resolve);
    });
    expect(syncInit).toEqual({ ok: true, baseline: BASELINE, actions: [], participants: [{ id: expect.any(String), name: 'Alice', role: 'edit' }] });
    client.disconnect();
  });

  it('replies with ok:false for an unknown session id', async () => {
    const client = connect();
    const syncInit = await new Promise<any>(resolve => {
      client.on('connect', () => client.emit('join', { sessionId: 'nope', name: 'Alice', role: 'edit' }));
      client.on('sync-init', resolve);
    });
    expect(syncInit).toEqual({ ok: false });
    client.disconnect();
  });
});

describe('action relay', () => {
  it('broadcasts an edit-role action to other participants but not back to the sender', async () => {
    const sessionId = createSession(BASELINE);
    const alice = connect();
    const bob = connect();
    await new Promise<void>(resolve => {
      alice.on('connect', () => alice.emit('join', { sessionId, name: 'Alice', role: 'edit' }));
      alice.on('sync-init', () => resolve());
    });
    await new Promise<void>(resolve => {
      bob.on('connect', () => bob.emit('join', { sessionId, name: 'Bob', role: 'edit' }));
      bob.on('sync-init', () => resolve());
    });

    const bobReceived = new Promise<any>(resolve => bob.on('action', resolve));
    let aliceReceivedOwnAction = false;
    alice.on('action', () => { aliceReceivedOwnAction = true; });

    alice.emit('action', { type: 'UPDATE_PLAN_RANGE', payload: { startDate: '2026-02-01', endDate: '2026-02-28' } });

    expect(await bobReceived).toEqual({ type: 'UPDATE_PLAN_RANGE', payload: { startDate: '2026-02-01', endDate: '2026-02-28' } });
    expect(aliceReceivedOwnAction).toBe(false);
    alice.disconnect();
    bob.disconnect();
  });

  it('ignores actions from view-role participants', async () => {
    const sessionId = createSession(BASELINE);
    const alice = connect();
    const viewer = connect();
    await new Promise<void>(resolve => {
      alice.on('connect', () => alice.emit('join', { sessionId, name: 'Alice', role: 'edit' }));
      alice.on('sync-init', () => resolve());
    });
    await new Promise<void>(resolve => {
      viewer.on('connect', () => viewer.emit('join', { sessionId, name: 'Viewer', role: 'view' }));
      viewer.on('sync-init', () => resolve());
    });

    let aliceReceivedAction = false;
    alice.on('action', () => { aliceReceivedAction = true; });
    viewer.emit('action', { type: 'UPDATE_PLAN_RANGE', payload: { startDate: '2026-03-01', endDate: '2026-03-31' } });

    await new Promise(resolve => setTimeout(resolve, 200));
    expect(aliceReceivedAction).toBe(false);
    alice.disconnect();
    viewer.disconnect();
  });
});

describe('presence', () => {
  it('notifies remaining participants when someone disconnects', async () => {
    const sessionId = createSession(BASELINE);
    const alice = connect();
    const bob = connect();
    await new Promise<void>(resolve => {
      alice.on('connect', () => alice.emit('join', { sessionId, name: 'Alice', role: 'edit' }));
      alice.on('sync-init', () => resolve());
    });
    const aliceSawPresenceDrop = new Promise<any>(resolve => alice.on('presence', resolve));
    await new Promise<void>(resolve => {
      bob.on('connect', () => bob.emit('join', { sessionId, name: 'Bob', role: 'edit' }));
      bob.on('sync-init', () => resolve());
    });
    bob.disconnect();
    expect(await aliceSawPresenceDrop).toEqual([{ id: expect.any(String), name: 'Alice', role: 'edit' }]);
    alice.disconnect();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd GanttChartEditor/server && npm test`
Expected: FAIL — `collabSocket.ts` doesn't exist yet.

- [ ] **Step 3: Implement `collabSocket.ts`**

Create `GanttChartEditor/server/src/collab/collabSocket.ts`:

```ts
import type { Server as HttpServer } from 'node:http';
import { Server, Socket } from 'socket.io';
import { randomUUID } from 'node:crypto';
import * as store from './sessionStore.js';

interface JoinPayload {
  sessionId: string;
  name: string;
  role: 'edit' | 'view';
}

interface ActionPayload {
  type: string;
  payload: unknown;
}

const IDLE_SWEEP_INTERVAL_MS = 5 * 60 * 1000;
const IDLE_SESSION_TIMEOUT_MS = 30 * 60 * 1000;

export function createCollabSocketServer(httpServer: HttpServer): Server {
  const io = new Server(httpServer, {
    path: '/collab/socket.io',
    cors: { origin: true },
    maxHttpBufferSize: 20 * 1024 * 1024, // schedules can be a few MB of JSON
  });

  io.on('connection', (socket: Socket) => {
    const participantId = randomUUID();
    let joinedSessionId: string | null = null;
    let joinedRole: 'edit' | 'view' = 'view';

    socket.on('join', ({ sessionId, name, role }: JoinPayload) => {
      const session = store.getSession(sessionId);
      if (!session) {
        socket.emit('sync-init', { ok: false });
        return;
      }
      joinedSessionId = sessionId;
      joinedRole = role;
      void socket.join(sessionId);
      const participants = store.addParticipant(sessionId, participantId, name, role) ?? [];
      socket.emit('sync-init', { ok: true, baseline: session.baseline, actions: session.actions, participants });
      socket.to(sessionId).emit('presence', participants);
    });

    socket.on('action', ({ type, payload }: ActionPayload) => {
      if (!joinedSessionId || joinedRole !== 'edit') return;
      const logged = store.appendAction(joinedSessionId, type, payload);
      if (!logged) return;
      socket.to(joinedSessionId).emit('action', { type: logged.type, payload: logged.payload });
    });

    const handleLeave = () => {
      if (!joinedSessionId) return;
      const participants = store.removeParticipant(joinedSessionId, participantId) ?? [];
      socket.to(joinedSessionId).emit('presence', participants);
      joinedSessionId = null;
    };

    socket.on('leave', handleLeave);
    socket.on('disconnect', handleLeave);
  });

  setInterval(() => store.sweepIdleSessions(IDLE_SESSION_TIMEOUT_MS), IDLE_SWEEP_INTERVAL_MS).unref();

  return io;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd GanttChartEditor/server && npm test`
Expected: PASS.

- [ ] **Step 5: Add the REST create-session route**

Create `GanttChartEditor/server/src/routes/collab.ts`:

```ts
import { Router } from 'express';
import * as store from '../collab/sessionStore.js';

export const collabRouter = Router();

collabRouter.post('/collab/sessions', (req, res) => {
  const { schedule, envConfig, currentView } = req.body as {
    schedule?: unknown;
    envConfig?: unknown;
    currentView?: 'worker' | 'device';
  };
  if (!schedule || !envConfig || (currentView !== 'worker' && currentView !== 'device')) {
    res.status(400).json({ ok: false, error: 'schedule, envConfig, currentView are required' });
    return;
  }
  const sessionId = store.createSession({ schedule, envConfig, currentView });
  res.json({ ok: true, sessionId });
});
```

- [ ] **Step 6: Wire both into `index.ts`, remove the old viewBroadcast module**

Modify `GanttChartEditor/server/src/index.ts`:

```ts
import express from 'express';
import cors from 'cors';
import path from 'node:path';
import { createServer } from 'node:http';
import { writeFile } from 'node:fs/promises';
import { constraintsRouter } from './routes/constraints.js';
import { handoffRouter } from './routes/handoff.js';
import { networkInfoRouter } from './routes/networkInfo.js';
import { collabRouter } from './routes/collab.js';
import { createCollabSocketServer } from './collab/collabSocket.js';

const app = express();
const PORT = Number(process.env.PORT ?? 3010);

app.use(cors({ origin: ['http://localhost:5173', 'http://localhost:5174'] }));
app.use(express.json({ limit: '10mb' }));

app.use('/api', constraintsRouter);
app.use('/api', handoffRouter);
app.use('/api', networkInfoRouter);
app.use('/api', collabRouter);
```

(Everything from `app.get('/api/health', ...)` down to the end of the file is unchanged, **except** replace the final block:)

```ts
// Plain app.listen() can't also host Socket.IO on the same port, so the
// collab relay wraps app in its own http.Server first.
const httpServer = createServer(app);
createCollabSocketServer(httpServer);

httpServer.listen(PORT, () => {
  console.log(`[server] running on http://localhost:${PORT}`);
});
```

- [ ] **Step 7: Delete the superseded module**

```bash
rm GanttChartEditor/server/src/collab/viewBroadcast.ts
```

- [ ] **Step 8: Run the full server test suite and build**

Run: `cd GanttChartEditor/server && npm test && npm run build`
Expected: PASS, and `tsc` reports no errors (confirms nothing else still imports `viewBroadcast.ts`).

- [ ] **Step 9: Commit**

```bash
git add GanttChartEditor/server/src/collab/collabSocket.ts GanttChartEditor/server/src/collab/collabSocket.test.ts GanttChartEditor/server/src/routes/collab.ts GanttChartEditor/server/src/index.ts
git rm GanttChartEditor/server/src/collab/viewBroadcast.ts
git commit -m "feat(server): replace snapshot-only Live View with a session/action-log relay"
```

---

### Task 3: Client types — session state and new action types

**Files:**
- Modify: `GanttChartEditor/src/types/appState.ts`

**Interfaces:**
- Produces: `SessionRole = 'edit' | 'view'`, `SessionConnectionStatus = 'disconnected' | 'connecting' | 'connected'`, `SessionParticipant { id: string; name: string; role: SessionRole }`, `SessionBaseline { schedule: ScheduleData; envConfig: EnvConfig; currentView: ViewMode }`, `SessionState { id: string; role: SessionRole; connectionStatus: SessionConnectionStatus; participants: SessionParticipant[] }`. Adds `session: SessionState | null` and `isSessionDialogOpen: boolean` to `AppState`. Adds action types `SET_SESSION`, `SET_SESSION_BASELINE`, `SET_SESSION_CONNECTION_STATUS`, `SET_SESSION_PARTICIPANTS`, `OPEN_SESSION_DIALOG`, `CLOSE_SESSION_DIALOG`. Removes `isSharingLiveView`, `liveViewShareLink`, `isShareViewDialogOpen`, `viewConnectionStatus` from `AppState`, and removes action types `SET_SHARING_LIVE_VIEW`, `SET_LIVE_VIEW_SHARE_LINK`, `OPEN_SHARE_VIEW_DIALOG`, `CLOSE_SHARE_VIEW_DIALOG`, `SET_VIEW_CONNECTION_STATUS`, `SET_VIEW_STATE`.

This task is type-only (no runtime behavior yet); it's verified together with Task 4's reducer tests, since a type-only change has nothing to independently test. Do Task 3 and Task 4 as one commit.

- [ ] **Step 1: Add the new session types and state fields**

In `GanttChartEditor/src/types/appState.ts`, add after the `WorkerDateCellFilter` interface (before `export interface AppState`):

```ts
export type SessionRole = 'edit' | 'view';
export type SessionConnectionStatus = 'disconnected' | 'connecting' | 'connected';

export interface SessionParticipant {
  id: string;
  name: string;
  role: SessionRole;
}

export interface SessionBaseline {
  schedule: ScheduleData;
  envConfig: EnvConfig;
  currentView: ViewMode;
}

export interface SessionState {
  id: string;
  role: SessionRole;
  connectionStatus: SessionConnectionStatus;
  participants: SessionParticipant[];
}
```

Replace this block in `AppState`:

```ts
  // Live View sharing (fast read-only broadcast, see services/viewBroadcastService.ts).
  // Not a collaboration session — one-way, host-to-viewers, no editing on the viewer side.
  isSharingLiveView: boolean;
  liveViewShareLink: string | null;
  isShareViewDialogOpen: boolean;
  viewConnectionStatus: 'disconnected' | 'connecting' | 'connected';
```

with:

```ts
  // Live collaboration session (see services/collabService.ts). null when not
  // in a session — solo editing/viewing is unaffected either way.
  session: SessionState | null;
  isSessionDialogOpen: boolean;
```

- [ ] **Step 2: Replace the Live View action types**

Replace this block at the end of `ActionType`:

```ts
  // Live View sharing
  | { type: 'SET_SHARING_LIVE_VIEW'; payload: boolean }
  | { type: 'SET_LIVE_VIEW_SHARE_LINK'; payload: string | null }
  | { type: 'OPEN_SHARE_VIEW_DIALOG' }
  | { type: 'CLOSE_SHARE_VIEW_DIALOG' }
  | { type: 'SET_VIEW_CONNECTION_STATUS'; payload: 'disconnected' | 'connecting' | 'connected' }
  | { type: 'SET_VIEW_STATE'; payload: { schedule: ScheduleData; envConfig: EnvConfig; currentView: ViewMode } };
```

with:

```ts
  // Live collaboration session
  | { type: 'SET_SESSION'; payload: SessionState | null }
  | { type: 'SET_SESSION_BASELINE'; payload: SessionBaseline }
  | { type: 'SET_SESSION_CONNECTION_STATUS'; payload: SessionConnectionStatus }
  | { type: 'SET_SESSION_PARTICIPANTS'; payload: SessionParticipant[] }
  | { type: 'OPEN_SESSION_DIALOG' }
  | { type: 'CLOSE_SESSION_DIALOG' };
```

- [ ] **Step 3: Continue to Task 4** (reducer changes) before compiling/testing — `appState.ts` alone will not type-check against `reducer.ts` until Task 4 lands.

---

### Task 4: Reducer — session actions

**Files:**
- Modify: `GanttChartEditor/src/context/reducer.ts`
- Modify: `GanttChartEditor/src/context/AppContext.tsx` (initial state)
- Modify: `GanttChartEditor/src/__tests__/context/reducer.test.ts`

**Interfaces:**
- Consumes: types from Task 3 (`SessionState`, `SessionBaseline`, `SessionParticipant`, `SessionConnectionStatus`).
- Produces: reducer handling for `SET_SESSION`, `SET_SESSION_BASELINE`, `SET_SESSION_CONNECTION_STATUS`, `SET_SESSION_PARTICIPANTS`, `OPEN_SESSION_DIALOG`, `CLOSE_SESSION_DIALOG`.

- [ ] **Step 1: Write the failing reducer tests**

In `GanttChartEditor/src/__tests__/context/reducer.test.ts`, add `session: null,` to `BASE_STATE` (after `isSendToSchedulerDialogOpen: false,`), then append at the end of the file:

```ts
// ── Live collaboration session ─────────────────────────────────────────────

describe('SET_SESSION', () => {
  it('sets the session', () => {
    const session = { id: 's1', role: 'edit' as const, connectionStatus: 'connecting' as const, participants: [] };
    const next = reducer(BASE_STATE, { type: 'SET_SESSION', payload: session });
    expect(next.session).toEqual(session);
  });

  it('clears the session', () => {
    const state = { ...BASE_STATE, session: { id: 's1', role: 'edit' as const, connectionStatus: 'connected' as const, participants: [] } };
    const next = reducer(state, { type: 'SET_SESSION', payload: null });
    expect(next.session).toBeNull();
  });
});

describe('SET_SESSION_BASELINE', () => {
  it('replaces schedule/envConfig/currentView, resets undo/redo and selection', () => {
    const state = {
      ...BASE_STATE,
      undoStack: [EMPTY_SCHEDULE],
      redoStack: [EMPTY_SCHEDULE],
      selectedAssignmentIndex: 0,
      selectedUnavailableInfo: { workerId: 'w001', startDate: '2025-09-01', endDate: '2025-09-02' },
    };
    const newSchedule = { ...EMPTY_SCHEDULE, planRange: { startDate: '2026-01-01', endDate: '2026-01-31' } };
    const next = reducer(state, { type: 'SET_SESSION_BASELINE', payload: { schedule: newSchedule, envConfig: EMPTY_ENV, currentView: 'device' } });
    expect(next.schedule).toBe(newSchedule);
    expect(next.currentView).toBe('device');
    expect(next.undoStack).toEqual([]);
    expect(next.redoStack).toEqual([]);
    expect(next.selectedAssignmentIndex).toBeNull();
    expect(next.selectedUnavailableInfo).toBeNull();
  });
});

describe('SET_SESSION_CONNECTION_STATUS', () => {
  it('updates connectionStatus on an existing session', () => {
    const state = { ...BASE_STATE, session: { id: 's1', role: 'edit' as const, connectionStatus: 'connecting' as const, participants: [] } };
    const next = reducer(state, { type: 'SET_SESSION_CONNECTION_STATUS', payload: 'connected' });
    expect(next.session?.connectionStatus).toBe('connected');
  });

  it('is a no-op when there is no session', () => {
    const next = reducer(BASE_STATE, { type: 'SET_SESSION_CONNECTION_STATUS', payload: 'connected' });
    expect(next.session).toBeNull();
  });
});

describe('SET_SESSION_PARTICIPANTS', () => {
  it('updates participants on an existing session', () => {
    const state = { ...BASE_STATE, session: { id: 's1', role: 'edit' as const, connectionStatus: 'connected' as const, participants: [] } };
    const participants = [{ id: 'p1', name: 'Alice', role: 'edit' as const }];
    const next = reducer(state, { type: 'SET_SESSION_PARTICIPANTS', payload: participants });
    expect(next.session?.participants).toEqual(participants);
  });
});

describe('OPEN_SESSION_DIALOG / CLOSE_SESSION_DIALOG', () => {
  it('opens and closes the session dialog', () => {
    const opened = reducer(BASE_STATE, { type: 'OPEN_SESSION_DIALOG' });
    expect(opened.isSessionDialogOpen).toBe(true);
    const closed = reducer(opened, { type: 'CLOSE_SESSION_DIALOG' });
    expect(closed.isSessionDialogOpen).toBe(false);
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd GanttChartEditor && npm run test:jest -- reducer.test`
Expected: FAIL — new action types not handled (falls through to `default`).

- [ ] **Step 3: Implement the reducer cases**

In `GanttChartEditor/src/context/reducer.ts`, replace this block (the old Live View cases through the switch's end):

```ts
    case 'SET_SHARING_LIVE_VIEW':
      return { ...state, isSharingLiveView: action.payload };

    case 'SET_LIVE_VIEW_SHARE_LINK':
      return { ...state, liveViewShareLink: action.payload };

    case 'OPEN_SHARE_VIEW_DIALOG':
      return { ...state, isShareViewDialogOpen: true };

    case 'CLOSE_SHARE_VIEW_DIALOG':
      return { ...state, isShareViewDialogOpen: false };

    case 'SET_VIEW_CONNECTION_STATUS':
      return { ...state, viewConnectionStatus: action.payload };

    case 'SET_VIEW_STATE':
      // Viewer-side only: mirrors whatever the host currently has. No undo/redo,
      // no saved-ref tracking — this is a read-only reflection, not an edit.
      return {
        ...state,
        schedule: action.payload.schedule,
        envConfig: action.payload.envConfig,
        currentView: action.payload.currentView,
      };

    default:
      return state;
  }
}
```

with:

```ts
    case 'SET_SESSION':
      return { ...state, session: action.payload };

    case 'SET_SESSION_BASELINE':
      // Applied when this client just joined a session (or is catching up):
      // replaces the working data with the session's current state. Undo/redo
      // and selection reset because they'd otherwise reference data from
      // before this client had any relationship to the session.
      return {
        ...state,
        schedule: action.payload.schedule,
        envConfig: action.payload.envConfig,
        currentView: action.payload.currentView,
        undoStack: [],
        redoStack: [],
        selectedAssignmentIndex: null,
        selectedUnavailableInfo: null,
        savedScheduleRef: action.payload.schedule,
        savedEnvConfigRef: action.payload.envConfig,
      };

    case 'SET_SESSION_CONNECTION_STATUS':
      return state.session ? { ...state, session: { ...state.session, connectionStatus: action.payload } } : state;

    case 'SET_SESSION_PARTICIPANTS':
      return state.session ? { ...state, session: { ...state.session, participants: action.payload } } : state;

    case 'OPEN_SESSION_DIALOG':
      return { ...state, isSessionDialogOpen: true };

    case 'CLOSE_SESSION_DIALOG':
      return { ...state, isSessionDialogOpen: false };

    default:
      return state;
  }
}
```

- [ ] **Step 4: Update `AppContext.tsx`'s initial state**

In `GanttChartEditor/src/context/AppContext.tsx`, replace:

```ts
  isSharingLiveView: false,
  liveViewShareLink: null,
  isShareViewDialogOpen: false,
  viewConnectionStatus: 'disconnected',
```

with:

```ts
  session: null,
  isSessionDialogOpen: false,
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd GanttChartEditor && npm run test:jest -- reducer.test`
Expected: PASS.

- [ ] **Step 6: Type-check the whole client**

Run: `cd GanttChartEditor && npx tsc -b`
Expected: no errors from `appState.ts`/`reducer.ts`/`AppContext.tsx` (other files still referencing the removed live-view fields are fixed in later tasks — if `tsc -b` fails elsewhere at this point, that's expected and resolved by Task 7/9/10; don't fix those files here).

- [ ] **Step 7: Commit**

```bash
git add GanttChartEditor/src/types/appState.ts GanttChartEditor/src/context/reducer.ts GanttChartEditor/src/context/AppContext.tsx GanttChartEditor/src/__tests__/context/reducer.test.ts
git commit -m "feat: add live-session state and reducer cases, remove old Live View fields"
```

---

### Task 5: Client — `collabService.ts` (transport layer)

**Files:**
- Create: `GanttChartEditor/src/services/collabService.ts`
- Delete: `GanttChartEditor/src/services/viewBroadcastService.ts`

**Interfaces:**
- Consumes: `SessionBaseline`, `SessionParticipant`, `SessionRole`, `SessionConnectionStatus` from `../types/appState`.
- Produces: `createCollabSession(baseline: SessionBaseline): Promise<string>`, `joinCollabRoom(sessionId: string, name: string, role: SessionRole, isCreator: boolean, onSyncInit: (baseline: SessionBaseline, actions: LoggedAction[]) => void, onAction: (action: { type: string; payload: unknown }) => void, onPresence: (participants: SessionParticipant[]) => void, onStatusChange: (status: SessionConnectionStatus) => void): () => void`, `sendCollabAction(type: string, payload: unknown): void`, `fetchCollabLink(sessionId: string, role: SessionRole): Promise<string>`, `parseSessionId(input: string): string`, `LoggedAction { seq: number; type: string; payload: unknown }`.

This is a thin I/O wrapper around `socket.io-client` and `fetch`, directly mirroring the existing (untested) `viewBroadcastService.ts` pattern — the codebase doesn't unit-test that kind of thin transport shim (there's no mock-socket test harness in this project), so this task is verified by the server integration test from Task 2 (same wire protocol) plus the end-to-end manual check in Task 12. Do not invent a fragile socket-mocking unit test here.

- [ ] **Step 1: Implement `collabService.ts`**

Create `GanttChartEditor/src/services/collabService.ts`:

```ts
import { io, Socket } from 'socket.io-client';
import type { SessionBaseline, SessionParticipant, SessionRole, SessionConnectionStatus } from '../types/appState';

export interface LoggedAction {
  seq: number;
  type: string;
  payload: unknown;
}

// Same-machine relay: the API/socket server always runs on port 3010 on the
// same machine as the frontend, whether reached as localhost (host) or a LAN
// IP (a joiner who opened the shared link) — window.location.hostname already
// reflects whichever address was actually used to load the page.
function getSocketOrigin(): string {
  return `${window.location.protocol}//${window.location.hostname}:3010`;
}

let socket: Socket | null = null;

function ensureSocket(): Socket {
  if (!socket) {
    socket = io(getSocketOrigin(), {
      path: '/collab/socket.io',
      transports: ['websocket', 'polling'],
    });
  }
  return socket;
}

export async function createCollabSession(baseline: SessionBaseline): Promise<string> {
  const res = await fetch(`${getSocketOrigin()}/api/collab/sessions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(baseline),
  });
  const data = (await res.json().catch(() => ({ ok: false }))) as { ok: boolean; sessionId?: string; error?: string };
  if (!res.ok || !data.ok || !data.sessionId) throw new Error(data.error ?? 'セッションの作成に失敗しました');
  return data.sessionId;
}

export function joinCollabRoom(
  sessionId: string,
  name: string,
  role: SessionRole,
  isCreator: boolean,
  onSyncInit: (baseline: SessionBaseline, actions: LoggedAction[]) => void,
  onAction: (action: { type: string; payload: unknown }) => void,
  onPresence: (participants: SessionParticipant[]) => void,
  onStatusChange: (status: SessionConnectionStatus) => void,
): () => void {
  const s = ensureSocket();
  onStatusChange('connecting');

  const handleConnect = () => s.emit('join', { sessionId, name, role });
  const handleSyncInit = (payload: { ok: boolean; baseline?: SessionBaseline; actions?: LoggedAction[]; participants?: SessionParticipant[] }) => {
    if (!payload.ok || !payload.baseline) {
      onStatusChange('disconnected');
      return;
    }
    // The creator's local state already IS the baseline (it was just POSTed
    // from there) — re-applying it would needlessly wipe their own undo
    // history. Everyone else (a real joiner) replays baseline + log to catch up.
    if (!isCreator) {
      onSyncInit(payload.baseline, payload.actions ?? []);
    }
    onPresence(payload.participants ?? []);
    onStatusChange('connected');
  };
  const handleAction = (payload: { type: string; payload: unknown }) => onAction(payload);
  const handlePresence = (participants: SessionParticipant[]) => onPresence(participants);
  const handleDisconnect = () => onStatusChange('disconnected');

  s.on('connect', handleConnect);
  s.on('sync-init', handleSyncInit);
  s.on('action', handleAction);
  s.on('presence', handlePresence);
  s.on('disconnect', handleDisconnect);

  if (s.connected) handleConnect();

  return () => {
    s.off('connect', handleConnect);
    s.off('sync-init', handleSyncInit);
    s.off('action', handleAction);
    s.off('presence', handlePresence);
    s.off('disconnect', handleDisconnect);
    s.emit('leave');
    s.disconnect();
    socket = null;
  };
}

export function sendCollabAction(type: string, payload: unknown): void {
  socket?.emit('action', { type, payload });
}

// Builds a link a participant on another PC (or the same PC, another tab) can
// open directly, e.g. http://192.168.1.23:5173/?session=<id>&role=edit —
// reuses whatever port/path this client is currently on.
export async function fetchCollabLink(sessionId: string, role: SessionRole): Promise<string> {
  const res = await fetch(`${getSocketOrigin()}/api/network-info`);
  const data = (await res.json()) as { ok: boolean; addresses: string[] };
  const lanIp = data.addresses[0] ?? window.location.hostname;
  return `${window.location.protocol}//${lanIp}:${window.location.port}${window.location.pathname}?session=${sessionId}&role=${role}`;
}

// Accepts either a bare session id or a full link (as produced by
// fetchCollabLink above) pasted into the "Join Session" dialog.
export function parseSessionId(input: string): string {
  const trimmed = input.trim();
  try {
    const url = new URL(trimmed);
    return url.searchParams.get('session') ?? trimmed;
  } catch {
    return trimmed;
  }
}
```

- [ ] **Step 2: Delete the superseded service**

```bash
rm GanttChartEditor/src/services/viewBroadcastService.ts
```

- [ ] **Step 3: Commit**

```bash
git add GanttChartEditor/src/services/collabService.ts
git rm GanttChartEditor/src/services/viewBroadcastService.ts
git commit -m "feat(client): add collabService, remove superseded viewBroadcastService"
```

(`npx tsc -b` will still show errors for `useHostBroadcast.ts`/`useViewerSync.ts`/`ShareViewButton.tsx`/`ShareViewDialog.tsx` importing the deleted service — expected until Task 7.)

---

### Task 6: AppContext — wrap dispatch, apply remote actions, expose session commands

**Files:**
- Modify: `GanttChartEditor/src/context/AppContext.tsx`
- Create: `GanttChartEditor/src/__tests__/context/AppContext.test.tsx`
- Modify: `GanttChartEditor/src/config/uiText.ts` (one new error string)

**Interfaces:**
- Consumes: `collabService.ts` from Task 5.
- Produces: `useAppContext()` now also returns `startCollabSession(name: string): Promise<{ sessionId: string; link: string }>`, `joinCollabSession(idOrLink: string, name: string, role: SessionRole): Promise<void>`, `leaveCollabSession(): void`.

- [ ] **Step 1: Add the UI error string**

In `GanttChartEditor/src/config/uiText.ts`, near `savePathUnknownMessage` (line 285), add:

```ts
  collabNoScheduleError: 'スケジュールを読み込んでから開始してください',
```

- [ ] **Step 2: Write the failing test**

Create `GanttChartEditor/src/__tests__/context/AppContext.test.tsx`:

```tsx
import { render, screen, act, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { AppProvider, useAppContext } from '../../context/AppContext';
import * as collabService from '../../services/collabService';
import { ScheduleData } from '../../types/schedule';
import { EnvConfig } from '../../types/envConfig';

jest.mock('../../services/collabService');
const mockedCollab = collabService as jest.Mocked<typeof collabService>;

const SCHEDULE: ScheduleData = {
  planRange: { startDate: '2026-01-01', endDate: '2026-01-31' },
  workflowTaskList: [],
  assignmentList: [],
};
const ENV_CONFIG: EnvConfig = {
  workflowList: [], fabList: [], regionList: [], customerCompanyList: [], workerCompanyList: [], workerList: [], transiteDayMap: [],
};

function TestConsumer() {
  const { state, dispatch, startCollabSession, joinCollabSession, leaveCollabSession } = useAppContext();
  return (
    <div>
      <div data-testid="schedule-start">{state.schedule?.planRange.startDate ?? 'none'}</div>
      <div data-testid="session-role">{state.session?.role ?? 'none'}</div>
      <button onClick={() => dispatch({ type: 'LOAD_FILES', payload: { schedule: SCHEDULE, envConfig: ENV_CONFIG, envPath: 'e.yaml', schedulePath: 's.yaml' } })}>load</button>
      <button onClick={() => dispatch({ type: 'UPDATE_PLAN_RANGE', payload: { startDate: '2026-02-01', endDate: '2026-02-28' } })}>edit</button>
      <button onClick={() => dispatch({ type: 'UNDO' })}>undo</button>
      <button onClick={() => dispatch({ type: 'TOGGLE_FLIGHT_STINTS' })}>toggle-local</button>
      <button onClick={() => void startCollabSession('Alice')}>start</button>
      <button onClick={() => void joinCollabSession('abc', 'Bob', 'edit')}>join</button>
      <button onClick={() => leaveCollabSession()}>leave</button>
    </div>
  );
}

function renderApp() {
  return render(<AppProvider><TestConsumer /></AppProvider>);
}

beforeEach(() => {
  jest.clearAllMocks();
  mockedCollab.parseSessionId.mockImplementation((s: string) => s);
});

it('forwards a syncable action to the server while in an edit session, but not a local-only one', async () => {
  mockedCollab.createCollabSession.mockResolvedValue('s1');
  mockedCollab.fetchCollabLink.mockResolvedValue('http://host/?session=s1&role=edit');
  mockedCollab.joinCollabRoom.mockReturnValue(() => {});

  renderApp();
  await userEvent.click(screen.getByText('load'));
  await act(async () => { await userEvent.click(screen.getByText('start')); });

  await userEvent.click(screen.getByText('edit'));
  expect(mockedCollab.sendCollabAction).toHaveBeenCalledTimes(1);
  expect(mockedCollab.sendCollabAction).toHaveBeenCalledWith('UPDATE_PLAN_RANGE', { startDate: '2026-02-01', endDate: '2026-02-28' });

  await userEvent.click(screen.getByText('toggle-local'));
  expect(mockedCollab.sendCollabAction).toHaveBeenCalledTimes(1); // still 1 — the local-only action wasn't forwarded
});

it('applies a remote action via the raw dispatch without forwarding it back to the server', async () => {
  let capturedOnAction: ((action: { type: string; payload: unknown }) => void) | null = null;
  mockedCollab.joinCollabRoom.mockImplementation((_id, _name, _role, _isCreator, onSyncInit, onAction) => {
    onSyncInit({ schedule: SCHEDULE, envConfig: ENV_CONFIG, currentView: 'worker' }, []);
    capturedOnAction = onAction;
    return () => {};
  });

  renderApp();
  await act(async () => { await userEvent.click(screen.getByText('join')); });
  await waitFor(() => expect(screen.getByTestId('schedule-start')).toHaveTextContent('2026-01-01'));

  act(() => capturedOnAction!({ type: 'UPDATE_PLAN_RANGE', payload: { startDate: '2026-03-01', endDate: '2026-03-31' } }));

  await waitFor(() => expect(screen.getByTestId('schedule-start')).toHaveTextContent('2026-03-01'));
  expect(mockedCollab.sendCollabAction).not.toHaveBeenCalled();
});

it('skips re-applying the baseline for the session creator', async () => {
  mockedCollab.createCollabSession.mockResolvedValue('s1');
  mockedCollab.fetchCollabLink.mockResolvedValue('http://host/?session=s1&role=edit');
  mockedCollab.joinCollabRoom.mockImplementation((_id, _name, _role, isCreator, onSyncInit) => {
    if (!isCreator) {
      onSyncInit({ schedule: { ...SCHEDULE, planRange: { startDate: '1999-01-01', endDate: '1999-01-02' } }, envConfig: ENV_CONFIG, currentView: 'worker' }, []);
    }
    return () => {};
  });

  renderApp();
  await userEvent.click(screen.getByText('load'));
  await act(async () => { await userEvent.click(screen.getByText('start')); });

  expect(screen.getByTestId('schedule-start')).toHaveTextContent('2026-01-01');
  expect(screen.getByTestId('session-role')).toHaveTextContent('edit');
});

it('forwards undo as the resulting SET_SCHEDULE snapshot, not the bare UNDO token', async () => {
  mockedCollab.joinCollabRoom.mockImplementation((_id, _name, _role, _isCreator, onSyncInit) => {
    onSyncInit({ schedule: SCHEDULE, envConfig: ENV_CONFIG, currentView: 'worker' }, []);
    return () => {};
  });

  renderApp();
  await act(async () => { await userEvent.click(screen.getByText('join')); });
  await waitFor(() => expect(screen.getByTestId('schedule-start')).toHaveTextContent('2026-01-01'));

  await userEvent.click(screen.getByText('edit'));
  mockedCollab.sendCollabAction.mockClear();

  await userEvent.click(screen.getByText('undo'));

  expect(mockedCollab.sendCollabAction).toHaveBeenCalledWith('SET_SCHEDULE', SCHEDULE);
  expect(mockedCollab.sendCollabAction).not.toHaveBeenCalledWith('UNDO', undefined);
});
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd GanttChartEditor && npm run test:jest -- AppContext.test`
Expected: FAIL — `startCollabSession`/`joinCollabSession`/`leaveCollabSession` don't exist on the context yet.

- [ ] **Step 4: Implement the wrapping in `AppContext.tsx`**

Replace the full contents of `GanttChartEditor/src/context/AppContext.tsx` with:

```tsx
import { createContext, useContext, useReducer, useCallback, useEffect, useRef, Dispatch, ReactNode } from 'react';
import {
  AppState, ActionType, SessionRole,
  DEFAULT_WORKER_VIEW_FILTER, DEFAULT_MODULE_VIEW_FILTER, DEFAULT_WORKER_COLUMN_FILTER,
} from '../types/appState';
import { reducer } from './reducer';
import {
  createCollabSession, joinCollabRoom, sendCollabAction, fetchCollabLink, parseSessionId,
} from '../services/collabService';
import { UI } from '../config/uiText';

const initialState: AppState = {
  envConfig: null,
  schedule: null,
  currentView: 'worker',
  violations: [],
  undoStack: [],
  redoStack: [],
  selectedAssignmentIndex: null,
  selectedUnavailableInfo: null,
  expandedDeviceIds: new Set(),
  workerViewFilter: { ...DEFAULT_WORKER_VIEW_FILTER },
  moduleViewFilter: { ...DEFAULT_MODULE_VIEW_FILTER },
  workerColumnFilter: { ...DEFAULT_WORKER_COLUMN_FILTER },
  workerDateCellFilter: { date: '', tasks: [] },
  currentEnvPath: null,
  currentSchedulePath: null,
  savedScheduleRef: null,
  savedEnvConfigRef: null,
  errorMessage: null,
  isTaskAddDialogOpen: false,
  isFileOpenDialogOpen: false,
  isNewScheduleDialogOpen: false,
  isSendToSchedulerDialogOpen: false,
  isConstraintDialogOpen: false,
  isConstraintChecking: false,
  backendViolations: [],
  constraintCheckedAt: null,
  showFlightStints: false,
  scrollToSelectedAssignment: false,
  session: null,
  isSessionDialogOpen: false,
};

// Reducer actions that mutate schedule/envConfig content and must reach every
// participant. Everything else (selection, filters, dialogs, which tab
// you're on, your own constraint-check run) is local UI state, per user.
const SYNCABLE_ACTION_TYPES = new Set<ActionType['type']>([
  'SET_SCHEDULE', 'UPDATE_PLAN_RANGE', 'ADD_ASSIGNMENT', 'UPDATE_ASSIGNMENT', 'DELETE_ASSIGNMENT',
  'UPDATE_PHASE_TASK', 'UPDATE_OPERATION_TASK', 'BULK_UPDATE_FLEXIBILITY', 'ADD_WORKFLOW_TASKS',
  'MERGE_DATA', 'DELETE_UNAVAILABLE_DATE', 'DELETE_UNAVAILABLE_RANGE', 'MOVE_UNAVAILABLE_DATE',
  'ADD_UNAVAILABLE_DATES', 'RESIZE_UNAVAILABLE_RANGE', 'UPDATE_OPERATION_TASK_COLOR',
  'UPDATE_WORKFLOW_TASK_COLOR', 'UPDATE_WORKER_DEFINITION', 'UPDATE_WORKER_DESC_FIELD',
]);

interface ContextType {
  state: AppState;
  dispatch: Dispatch<ActionType>;
  startCollabSession: (name: string) => Promise<{ sessionId: string; link: string }>;
  joinCollabSession: (idOrLink: string, name: string, role: SessionRole) => Promise<void>;
  leaveCollabSession: () => void;
}

const AppContext = createContext<ContextType | undefined>(undefined);

export function AppProvider({ children }: { children: ReactNode }) {
  const [state, rawDispatch] = useReducer(reducer, initialState);
  const stateRef = useRef(state);
  stateRef.current = state;
  const disconnectRef = useRef<(() => void) | null>(null);

  // Outgoing: apply locally as normal, and if we're an editor in an active
  // session, also forward data-mutating actions to the server. UNDO/REDO are
  // client-local snapshot-stack operations (see reducer.ts) — a late joiner's
  // stack only has what happened since they joined, so we forward the
  // resulting schedule instead of the bare token, and everyone converges on
  // the same content regardless of their own undo history.
  const dispatch: Dispatch<ActionType> = useCallback((action: ActionType) => {
    if (action.type === 'UNDO' || action.type === 'REDO') {
      const before = stateRef.current;
      rawDispatch(action);
      if (before.session?.role === 'edit') {
        const resulting = action.type === 'UNDO'
          ? before.undoStack[before.undoStack.length - 1]
          : before.redoStack[before.redoStack.length - 1];
        if (resulting) sendCollabAction('SET_SCHEDULE', resulting);
      }
      return;
    }
    rawDispatch(action);
    if (stateRef.current.session?.role === 'edit' && SYNCABLE_ACTION_TYPES.has(action.type)) {
      sendCollabAction(action.type, (action as { payload?: unknown }).payload);
    }
  }, []);

  const joinInternal = useCallback((sessionId: string, name: string, role: SessionRole, isCreator: boolean) => {
    disconnectRef.current?.();
    disconnectRef.current = joinCollabRoom(
      sessionId, name, role, isCreator,
      (baseline, actions) => {
        // Incoming: applied via the raw dispatch, never the wrapped one —
        // otherwise a remote action would be immediately re-forwarded back
        // to the server and echo forever.
        rawDispatch({ type: 'SET_SESSION_BASELINE', payload: baseline });
        for (const a of actions) rawDispatch({ type: a.type, payload: a.payload } as ActionType);
      },
      (action) => rawDispatch({ type: action.type, payload: action.payload } as ActionType),
      (participants) => rawDispatch({ type: 'SET_SESSION_PARTICIPANTS', payload: participants }),
      (status) => rawDispatch({ type: 'SET_SESSION_CONNECTION_STATUS', payload: status }),
    );
  }, []);

  const startCollabSession = useCallback(async (name: string) => {
    const { schedule, envConfig, currentView } = stateRef.current;
    if (!schedule || !envConfig) throw new Error(UI.collabNoScheduleError);
    const sessionId = await createCollabSession({ schedule, envConfig, currentView });
    const link = await fetchCollabLink(sessionId, 'edit');
    rawDispatch({ type: 'SET_SESSION', payload: { id: sessionId, role: 'edit', connectionStatus: 'connecting', participants: [] } });
    joinInternal(sessionId, name, 'edit', true);
    return { sessionId, link };
  }, [joinInternal]);

  const joinCollabSession = useCallback(async (idOrLink: string, name: string, role: SessionRole) => {
    const sessionId = parseSessionId(idOrLink);
    rawDispatch({ type: 'SET_SESSION', payload: { id: sessionId, role, connectionStatus: 'connecting', participants: [] } });
    joinInternal(sessionId, name, role, false);
  }, [joinInternal]);

  const leaveCollabSession = useCallback(() => {
    disconnectRef.current?.();
    disconnectRef.current = null;
    rawDispatch({ type: 'SET_SESSION', payload: null });
  }, []);

  useEffect(() => () => disconnectRef.current?.(), []);

  return (
    <AppContext.Provider value={{ state, dispatch, startCollabSession, joinCollabSession, leaveCollabSession }}>
      {children}
    </AppContext.Provider>
  );
}

export function useAppContext(): ContextType {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error('useAppContext must be used within AppProvider');
  return ctx;
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd GanttChartEditor && npm run test:jest -- AppContext.test`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add GanttChartEditor/src/context/AppContext.tsx GanttChartEditor/src/__tests__/context/AppContext.test.tsx GanttChartEditor/src/config/uiText.ts
git commit -m "feat(client): wrap dispatch to sync/apply collab session actions"
```

---

### Task 7: Delete the superseded Live View hooks and button

**Files:**
- Delete: `GanttChartEditor/src/hooks/useHostBroadcast.ts`
- Delete: `GanttChartEditor/src/hooks/useViewerSync.ts`
- Delete: `GanttChartEditor/src/components/Toolbar/ShareViewButton.tsx`
- Delete: `GanttChartEditor/src/components/Dialogs/ShareViewDialog.tsx`

These are fully superseded: `useHostBroadcast`/`useViewerSync` by `AppContext.tsx`'s new wiring (Task 6), `ShareViewButton`/`ShareViewDialog` by the Collaboration menu and `SessionDialog` (Tasks 8-9). They're still referenced from `MenuBar.tsx` and `App.tsx` until those are updated in Tasks 8-10, so this task's deletions land in the same commit as Task 8 (the plan lists them separately for clarity, but don't run `tsc -b` in between — it will show errors until Task 10 is also done).

- [ ] **Step 1: Delete the files**

```bash
rm GanttChartEditor/src/hooks/useHostBroadcast.ts
rm GanttChartEditor/src/hooks/useViewerSync.ts
rm GanttChartEditor/src/components/Toolbar/ShareViewButton.tsx
rm GanttChartEditor/src/components/Dialogs/ShareViewDialog.tsx
```

(No commit yet — continue directly to Task 8, then commit both together as instructed there.)

---

### Task 8: `SessionDialog` — start/join/leave UI

**Files:**
- Create: `GanttChartEditor/src/components/Dialogs/SessionDialog.tsx`
- Modify: `GanttChartEditor/src/config/uiText.ts`

**Interfaces:**
- Consumes: `useAppContext()` (`state.session`, `state.isSessionDialogOpen`, `dispatch`, `startCollabSession`, `joinCollabSession`, `leaveCollabSession`) and `fetchCollabLink` from `../../services/collabService`.

- [ ] **Step 1: Add UI text**

In `GanttChartEditor/src/config/uiText.ts`, replace the block at lines 348-355:

```ts
  shareLiveViewBtn: 'ライブビューを共有',
  stopSharingBtn: '共有を停止',
  shareViewDialogTitle: 'ライブビューの共有',
  shareViewDialogDesc: 'このリンクを開くと、他の人が現在の内容を閲覧専用で見ることができます（編集は不可）。ホストの操作に応じて自動で更新されます。',
  shareViewLinkLoading: 'リンクを生成中…',
  copyLinkBtn: 'コピー',
  copyLinkCopied: 'コピーしました',
  shareViewClose: '閉じる',
```

with:

```ts
  collabMenu: '共同編集',
  startSessionItem: 'セッションを開始',
  joinSessionItem: 'セッションに参加',
  leaveSessionItem: 'セッションを終了',
  sessionParticipantsLabel: (n: number) => `🟢 ${n}人が参加中`,
  sessionDialogStartTitle: 'セッションを開始',
  sessionDialogStartDesc: '現在のスケジュールを元にセッションを開始します。他の人は編集用または閲覧用のリンクから参加できます。',
  sessionDialogJoinTitle: 'セッションに参加',
  sessionDialogJoinDesc: '受け取ったリンク、またはセッションIDを貼り付けてください。',
  sessionNamePlaceholder: '表示名を入力',
  sessionStartBtn: '開始する',
  sessionJoinLinkPlaceholder: 'リンクまたはセッションID',
  sessionJoinRoleEdit: '編集',
  sessionJoinRoleView: '閲覧のみ',
  sessionJoinBtn: '参加する',
  sessionActiveTitle: 'セッション情報',
  sessionEditLinkLabel: '編集用リンク',
  sessionViewLinkLabel: '閲覧用リンク',
  sessionLeaveBtn: 'セッションを終了',
  sessionCloseBtn: '閉じる',
  copyLinkBtn: 'コピー',
  copyLinkCopied: 'コピーしました',
```

(`copyLinkBtn`/`copyLinkCopied` are kept — reused by the new dialog.)

- [ ] **Step 2: Implement `SessionDialog.tsx`**

Create `GanttChartEditor/src/components/Dialogs/SessionDialog.tsx`:

```tsx
import { useState } from 'react';
import { useAppContext } from '../../context/AppContext';
import { fetchCollabLink } from '../../services/collabService';
import { SessionRole } from '../../types/appState';
import { UI } from '../../config/uiText';

const overlayStyle: React.CSSProperties = {
  position: 'fixed', inset: 0, backgroundColor: 'rgba(0,0,0,0.4)',
  display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000,
};
const boxStyle: React.CSSProperties = {
  backgroundColor: '#fff', borderRadius: 6, padding: 24, maxWidth: 480, width: '90%',
  boxShadow: '0 4px 16px rgba(0,0,0,0.3)', fontFamily: 'MS Gothic, monospace',
};
const inputStyle: React.CSSProperties = {
  width: '100%', padding: '6px 8px', fontSize: 12, border: '1px solid #ccc', borderRadius: 4, boxSizing: 'border-box',
};
const primaryBtnStyle: React.CSSProperties = {
  padding: '6px 16px', backgroundColor: '#1976d2', color: '#fff', border: 'none', borderRadius: 4, cursor: 'pointer', fontSize: 13,
};
const dangerBtnStyle: React.CSSProperties = {
  padding: '6px 16px', backgroundColor: '#c62828', color: '#fff', border: 'none', borderRadius: 4, cursor: 'pointer', fontSize: 13,
};
const neutralBtnStyle: React.CSSProperties = {
  padding: '6px 16px', backgroundColor: '#78909c', color: '#fff', border: 'none', borderRadius: 4, cursor: 'pointer', fontSize: 13,
};

function LinkRow({ label, link }: { label: string; link: string }) {
  const [copied, setCopied] = useState(false);
  const handleCopy = async () => {
    await navigator.clipboard.writeText(link);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };
  return (
    <div style={{ marginBottom: 12 }}>
      <div style={{ fontSize: 11, color: '#666', marginBottom: 4 }}>{label}</div>
      <div style={{ display: 'flex', gap: 8 }}>
        <input readOnly value={link} onFocus={e => e.currentTarget.select()} style={{ ...inputStyle, flex: 1, backgroundColor: '#f8f9fa' }} />
        <button onClick={() => void handleCopy()} style={{ ...primaryBtnStyle, backgroundColor: copied ? '#388e3c' : '#1976d2', whiteSpace: 'nowrap' }}>
          {copied ? UI.copyLinkCopied : UI.copyLinkBtn}
        </button>
      </div>
    </div>
  );
}

function ActiveSessionPanel({ onClose }: { onClose: () => void }) {
  const { state, leaveCollabSession } = useAppContext();
  const [editLink, setEditLink] = useState<string | null>(null);
  const [viewLink, setViewLink] = useState<string | null>(null);
  const session = state.session;
  if (!session) return null;

  const loadLinks = () => {
    if (editLink || viewLink) return;
    void fetchCollabLink(session.id, 'edit').then(setEditLink);
    void fetchCollabLink(session.id, 'view').then(setViewLink);
  };
  loadLinks();

  return (
    <div>
      <div style={{ fontSize: 15, fontWeight: 'bold', color: '#1a2e3f', marginBottom: 8 }}>{UI.sessionActiveTitle}</div>
      {editLink && <LinkRow label={UI.sessionEditLinkLabel} link={editLink} />}
      {viewLink && <LinkRow label={UI.sessionViewLinkLabel} link={viewLink} />}
      <div style={{ fontSize: 11, color: '#666', marginBottom: 16 }}>
        {UI.sessionParticipantsLabel(session.participants.length)}
      </div>
      <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
        <button onClick={() => { leaveCollabSession(); onClose(); }} style={dangerBtnStyle}>{UI.sessionLeaveBtn}</button>
        <button onClick={onClose} style={neutralBtnStyle}>{UI.sessionCloseBtn}</button>
      </div>
    </div>
  );
}

function StartOrJoinPanel({ onClose }: { onClose: () => void }) {
  const { startCollabSession, joinCollabSession } = useAppContext();
  const [tab, setTab] = useState<'start' | 'join'>('start');
  const [name, setName] = useState('');
  const [joinInput, setJoinInput] = useState('');
  const [joinRole, setJoinRole] = useState<SessionRole>('edit');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleStart = async () => {
    if (!name.trim()) return;
    setBusy(true);
    setError(null);
    try {
      await startCollabSession(name.trim());
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  const handleJoin = async () => {
    if (!name.trim() || !joinInput.trim()) return;
    setBusy(true);
    setError(null);
    try {
      await joinCollabSession(joinInput.trim(), name.trim(), joinRole);
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div>
      <div style={{ display: 'flex', gap: 8, marginBottom: 16 }}>
        <button onClick={() => setTab('start')} style={{ ...primaryBtnStyle, backgroundColor: tab === 'start' ? '#1976d2' : '#b0bec5' }}>{UI.startSessionItem}</button>
        <button onClick={() => setTab('join')} style={{ ...primaryBtnStyle, backgroundColor: tab === 'join' ? '#1976d2' : '#b0bec5' }}>{UI.joinSessionItem}</button>
      </div>

      <div style={{ fontSize: 15, fontWeight: 'bold', color: '#1a2e3f', marginBottom: 8 }}>
        {tab === 'start' ? UI.sessionDialogStartTitle : UI.sessionDialogJoinTitle}
      </div>
      <div style={{ fontSize: 12, color: '#666', marginBottom: 16 }}>
        {tab === 'start' ? UI.sessionDialogStartDesc : UI.sessionDialogJoinDesc}
      </div>

      <input placeholder={UI.sessionNamePlaceholder} value={name} onChange={e => setName(e.target.value)} style={{ ...inputStyle, marginBottom: 12 }} />

      {tab === 'join' && (
        <>
          <input placeholder={UI.sessionJoinLinkPlaceholder} value={joinInput} onChange={e => setJoinInput(e.target.value)} style={{ ...inputStyle, marginBottom: 12 }} />
          <div style={{ display: 'flex', gap: 16, marginBottom: 16, fontSize: 12 }}>
            <label><input type="radio" checked={joinRole === 'edit'} onChange={() => setJoinRole('edit')} /> {UI.sessionJoinRoleEdit}</label>
            <label><input type="radio" checked={joinRole === 'view'} onChange={() => setJoinRole('view')} /> {UI.sessionJoinRoleView}</label>
          </div>
        </>
      )}

      {error && <div style={{ color: '#c62828', fontSize: 12, marginBottom: 12 }}>{error}</div>}

      <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
        <button
          onClick={() => void (tab === 'start' ? handleStart() : handleJoin())}
          disabled={busy || !name.trim() || (tab === 'join' && !joinInput.trim())}
          style={primaryBtnStyle}
        >
          {tab === 'start' ? UI.sessionStartBtn : UI.sessionJoinBtn}
        </button>
        <button onClick={onClose} style={neutralBtnStyle}>{UI.sessionCloseBtn}</button>
      </div>
    </div>
  );
}

export function SessionDialog() {
  const { state, dispatch } = useAppContext();
  if (!state.isSessionDialogOpen) return null;
  const handleClose = () => dispatch({ type: 'CLOSE_SESSION_DIALOG' });

  return (
    <div style={overlayStyle}>
      <div style={boxStyle}>
        {state.session ? <ActiveSessionPanel onClose={handleClose} /> : <StartOrJoinPanel onClose={handleClose} />}
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Commit (together with Task 7's deletions)**

```bash
git add GanttChartEditor/src/components/Dialogs/SessionDialog.tsx GanttChartEditor/src/config/uiText.ts
git rm GanttChartEditor/src/hooks/useHostBroadcast.ts GanttChartEditor/src/hooks/useViewerSync.ts GanttChartEditor/src/components/Toolbar/ShareViewButton.tsx GanttChartEditor/src/components/Dialogs/ShareViewDialog.tsx
git commit -m "feat(client): add SessionDialog, remove superseded Live View hooks/UI"
```

(`tsc -b` will still fail on `MenuBar.tsx`/`App.tsx` referring to the removed pieces until Task 9-10 — expected, continue.)

---

### Task 9: MenuBar — Collaboration menu, remove `ShareViewButton`

**Files:**
- Modify: `GanttChartEditor/src/components/Toolbar/MenuBar.tsx`

**Interfaces:**
- Consumes: `dispatch`, `state.session` from `useAppContext()`.

- [ ] **Step 1: Remove the `ShareViewButton` import and usage**

In `GanttChartEditor/src/components/Toolbar/MenuBar.tsx`, remove:

```ts
import { ShareViewButton } from './ShareViewButton';
```

and remove the block:

```tsx
        <div style={{ marginLeft: 'auto', marginRight: 8, display: 'flex' }}>
          <ShareViewButton />
        </div>
```

- [ ] **Step 2: Add the Collaboration menu**

Replace:

```ts
    { id: 'edit', label: UI.editMenu, items: [] },
    { id: 'view', label: UI.viewMenu, items: [] },
    { id: 'help', label: UI.helpMenu, items: [] },
  ];
```

with:

```ts
    { id: 'edit', label: UI.editMenu, items: [] },
    { id: 'view', label: UI.viewMenu, items: [] },
    {
      id: 'collab',
      label: UI.collabMenu,
      items: state.session
        ? [{ label: UI.leaveSessionItem, action: () => { leaveCollabSession(); setOpenMenu(null); } }]
        : [
            { label: UI.startSessionItem, action: () => { dispatch({ type: 'OPEN_SESSION_DIALOG' }); setOpenMenu(null); }, disabled: !canSave },
            { label: UI.joinSessionItem, action: () => { dispatch({ type: 'OPEN_SESSION_DIALOG' }); setOpenMenu(null); } },
          ],
    },
    { id: 'help', label: UI.helpMenu, items: [] },
  ];
```

(`canSave` already exists in this component — reused here as the "a schedule is loaded" check, same gate as Save/Save As.)

Update the destructure at the top of `MenuBar()`:

```ts
  const { state, dispatch } = useAppContext();
```

to:

```ts
  const { state, dispatch, leaveCollabSession } = useAppContext();
```

- [ ] **Step 3: Show session presence next to the menus**

Replace:

```tsx
        {saveStatus && (
          <span style={{ color: '#a8d4f5', fontSize: 11, marginLeft: 16, fontFamily: 'Meiryo, sans-serif' }}>
            {saveStatus}
          </span>
        )}
```

with:

```tsx
        {saveStatus && (
          <span style={{ color: '#a8d4f5', fontSize: 11, marginLeft: 16, fontFamily: 'Meiryo, sans-serif' }}>
            {saveStatus}
          </span>
        )}

        {state.session && (
          <span style={{ color: '#a8d4f5', fontSize: 11, marginLeft: 16, fontFamily: 'Meiryo, sans-serif' }}>
            {UI.sessionParticipantsLabel(state.session.participants.length)}
          </span>
        )}
```

- [ ] **Step 4: Render `SessionDialog` from `App.tsx`, not `MenuBar.tsx`**

`SessionDialog` is a full-screen overlay like `ShareViewDialog` was — it belongs at the `AppContent` level in `App.tsx` (Task 10), not inside `MenuBar`. No change needed here beyond what Steps 1-3 already did.

- [ ] **Step 5: Commit**

```bash
git add GanttChartEditor/src/components/Toolbar/MenuBar.tsx
git commit -m "feat(client): move collaboration entry point into the menu bar"
```

(Still expect `tsc -b` errors in `App.tsx` until Task 10.)

---

### Task 10: App.tsx — browser join flow (`?session=<id>&role=edit|view`)

**Files:**
- Modify: `GanttChartEditor/src/App.tsx`
- Modify: `GanttChartEditor/src/pages/ViewPage.tsx`
- Modify: `GanttChartEditor/src/config/uiText.ts`

**Interfaces:**
- Consumes: `joinCollabSession` from `useAppContext()`.

- [ ] **Step 1: Add UI text for the join-name prompt**

In `GanttChartEditor/src/config/uiText.ts`, add near the collab keys added in Task 8:

```ts
  joinSessionPromptTitle: 'このセッションに参加',
  joinSessionNamePlaceholder: '表示名を入力',
  joinSessionSubmitBtn: '参加する',
  joinSessionErrorFallback: 'セッションへの参加に失敗しました',
```

- [ ] **Step 2: Simplify `ViewPage.tsx`** (it no longer drives its own connection — `AppContext`'s join wiring already populates `state.schedule`/`state.envConfig`/`state.currentView`)

In `GanttChartEditor/src/pages/ViewPage.tsx`, remove:

```ts
import { useViewerSync } from '../hooks/useViewerSync';
```

and remove the line `useViewerSync();` from inside `ViewPage()`. Replace:

```ts
  const { schedule, currentView, viewConnectionStatus } = state;
```

with:

```ts
  const { schedule, currentView, session } = state;
  const viewConnectionStatus = session?.connectionStatus ?? 'disconnected';
```

Everything else in `ViewPage.tsx` (the status bar rendering, `statusLabel`/`statusColor` lookups, the `pointer-events: none` wrapper) stays exactly as-is.

- [ ] **Step 3: Rewrite `App.tsx`**

Replace the full contents of `GanttChartEditor/src/App.tsx` with:

```tsx
import { useEffect, useState } from 'react';
import { AppProvider, useAppContext } from './context/AppContext';
import { GanttPage } from './pages/GanttPage';
import { ViewPage } from './pages/ViewPage';
import { ErrorDialog } from './components/Dialogs/ErrorDialog';
import { TaskAddDialog } from './components/Dialogs/TaskAddDialog';
import { NewScheduleDialog } from './components/Dialogs/NewScheduleDialog';
import { ConstraintResultDialog } from './components/Dialogs/ConstraintResultDialog';
import { SendToSchedulerDialog } from './components/Dialogs/SendToSchedulerDialog';
import { SessionDialog } from './components/Dialogs/SessionDialog';
import { useKeyboardShortcuts } from './hooks/useKeyboardShortcuts';
import { useConstraintCheck } from './hooks/useConstraintCheck';
import { useIncomingGanttTransfer } from './hooks/useIncomingGanttTransfer';
import { UI } from './config/uiText';
import { SessionRole } from './types/appState';

// A link generated by "Start Session"/"Join Session" lands here
// (?session=<id>&role=edit|view) — opened by anyone on the network, in a
// plain browser, no login/build required.
function getSessionParamsFromUrl(): { sessionId: string; role: SessionRole } | null {
  const params = new URLSearchParams(window.location.search);
  const sessionId = params.get('session');
  if (!sessionId) return null;
  return { sessionId, role: params.get('role') === 'view' ? 'view' : 'edit' };
}

function AppContent() {
  useKeyboardShortcuts();
  useConstraintCheck();
  useIncomingGanttTransfer();
  return (
    <>
      <GanttPage />
      <TaskAddDialog />
      <NewScheduleDialog />
      <ConstraintResultDialog />
      <SendToSchedulerDialog />
      <SessionDialog />
      <ErrorDialog />
    </>
  );
}

function SessionJoinGate({ sessionId, role }: { sessionId: string; role: SessionRole }) {
  const { joinCollabSession } = useAppContext();
  const [name, setName] = useState('');
  const [joining, setJoining] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!joining) return;
    let cancelled = false;
    joinCollabSession(sessionId, name, role).catch(err => {
      if (!cancelled) setError(err instanceof Error ? err.message : UI.joinSessionErrorFallback);
    });
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [joining]);

  if (error) {
    return (
      <div style={{ height: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', fontFamily: 'MS Gothic, monospace', color: '#c62828' }}>
        {error}
      </div>
    );
  }

  if (!joining) {
    return (
      <div style={{ height: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', fontFamily: 'MS Gothic, monospace' }}>
        <form
          onSubmit={e => { e.preventDefault(); if (name.trim()) setJoining(true); }}
          style={{ display: 'flex', flexDirection: 'column', gap: 12, width: 280 }}
        >
          <div style={{ fontSize: 14, fontWeight: 'bold' }}>{UI.joinSessionPromptTitle}</div>
          <input
            autoFocus
            value={name}
            onChange={e => setName(e.target.value)}
            placeholder={UI.joinSessionNamePlaceholder}
            style={{ padding: '6px 8px', fontSize: 13, border: '1px solid #ccc', borderRadius: 4 }}
          />
          <button
            type="submit"
            disabled={!name.trim()}
            style={{ padding: '6px 12px', backgroundColor: '#1976d2', color: '#fff', border: 'none', borderRadius: 4, cursor: name.trim() ? 'pointer' : 'default' }}
          >
            {UI.joinSessionSubmitBtn}
          </button>
        </form>
      </div>
    );
  }

  return role === 'view' ? <ViewPage /> : <AppContent />;
}

export default function App() {
  const sessionParams = getSessionParamsFromUrl();
  return (
    <AppProvider>
      {sessionParams
        ? <SessionJoinGate sessionId={sessionParams.sessionId} role={sessionParams.role} />
        : <AppContent />}
    </AppProvider>
  );
}
```

- [ ] **Step 4: Type-check and run the full client test suite**

Run: `cd GanttChartEditor && npx tsc -b && npm run test:jest`
Expected: no type errors, all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add GanttChartEditor/src/App.tsx GanttChartEditor/src/pages/ViewPage.tsx GanttChartEditor/src/config/uiText.ts
git commit -m "feat(client): add browser/app join flow for collab sessions"
```

---

### Task 11: Guard "Open File" while in a session

**Files:**
- Modify: `GanttChartEditor/src/components/Toolbar/MenuBar.tsx`
- Modify: `GanttChartEditor/src/hooks/useKeyboardShortcuts.ts`

Loading a different file mid-session would silently replace shared data out from under other participants, so "Open File" is disabled while a session is active (leaving the session re-enables it).

- [ ] **Step 1: Disable the menu item**

In `GanttChartEditor/src/components/Toolbar/MenuBar.tsx`, the `file` menu's `items` array currently has:

```ts
        { label: UI.open, shortcut: 'Ctrl+O', action: openFileDialog },
```

Change to:

```ts
        { label: UI.open, shortcut: 'Ctrl+O', action: openFileDialog, disabled: !!state.session },
```

- [ ] **Step 2: Guard the keyboard shortcut**

In `GanttChartEditor/src/hooks/useKeyboardShortcuts.ts`, change:

```ts
          case 'o':
            e.preventDefault();
            dispatch({ type: 'OPEN_FILE_DIALOG' });
            break;
```

to:

```ts
          case 'o':
            e.preventDefault();
            if (!state.session) dispatch({ type: 'OPEN_FILE_DIALOG' });
            break;
```

and add `state.session` to the effect's dependency array:

```ts
  }, [state.schedule, state.envConfig, state.selectedAssignmentIndex, state.currentSchedulePath, state.currentEnvPath, state.session, dispatch]);
```

- [ ] **Step 3: Manual check**

Run the app (`npm run dev:all` in one terminal, `npm run electron:dev` or open `http://localhost:5173` in a browser), start a session, confirm File > Open is greyed out and Ctrl+O does nothing; leave the session, confirm it's usable again.

- [ ] **Step 4: Commit**

```bash
git add GanttChartEditor/src/components/Toolbar/MenuBar.tsx GanttChartEditor/src/hooks/useKeyboardShortcuts.ts
git commit -m "fix(client): disable Open File while a collab session is active"
```

---

### Task 12: Electron — detach the embedded server, tray icon, host-leaves-without-ending-session

**Files:**
- Modify: `GanttChartEditor/electron/main.cts`
- Modify: `GanttChartEditor/electron/preload.cts`
- Modify: `GanttChartEditor/src/types/electron.d.ts`
- Modify: `GanttChartEditor/src/context/AppContext.tsx`

**Interfaces:**
- Produces: `window.electronAPI.setCollabSessionActive(active: boolean): void`.

- [ ] **Step 1: Add the IPC method to the preload's `ElectronAPI`**

In `GanttChartEditor/electron/preload.cts`, add to the `ElectronAPI` interface:

```ts
  /** Tells the main process whether a collab session is active, so it can
   *  keep the embedded server running (hidden to tray) instead of quitting
   *  when the window closes. */
  setCollabSessionActive: (active: boolean) => void;
```

and to the `api` object:

```ts
  setCollabSessionActive: (active: boolean) => ipcRenderer.send('collab:session-active-changed', active),
```

- [ ] **Step 2: Mirror the same interface in `src/types/electron.d.ts`**

In `GanttChartEditor/src/types/electron.d.ts`, add to `ElectronAPI`:

```ts
  setCollabSessionActive: (active: boolean) => void;
```

- [ ] **Step 3: Add tray + close-intercept logic to `main.cts`**

In `GanttChartEditor/electron/main.cts`, change the import line:

```ts
import { app, BrowserWindow, dialog, ipcMain } from 'electron';
```

to:

```ts
import { app, BrowserWindow, dialog, ipcMain, Tray, Menu, nativeImage } from 'electron';
```

Add, near the top-level `let mainWindow: BrowserWindow | null = null;` (line 37):

```ts
let tray: Tray | null = null;
let sessionActive = false;

// 1x1 transparent PNG — a placeholder tray icon good enough to make the
// tray entry appear and be clickable; swap for a real icon asset later.
const TRAY_ICON_DATA_URL =
  'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=';

function ensureTray(): void {
  if (tray) return;
  tray = new Tray(nativeImage.createFromDataURL(TRAY_ICON_DATA_URL));
  tray.setToolTip('GanttChartEditor — 共同編集セッション実行中');
  tray.setContextMenu(Menu.buildFromTemplate([
    {
      label: 'ウィンドウを開く',
      click: () => {
        if (mainWindow) { mainWindow.show(); mainWindow.focus(); } else { void createWindow(); }
      },
    },
    {
      label: '終了（セッションも終了します）',
      click: () => { sessionActive = false; mainWindow?.destroy(); app.quit(); },
    },
  ]));
}

function destroyTray(): void {
  tray?.destroy();
  tray = null;
}

ipcMain.on('collab:session-active-changed', (_evt, active: boolean) => {
  sessionActive = active;
  if (active) ensureTray(); else destroyTray();
});
```

In `createWindow()`, right after `mainWindow = new BrowserWindow({...})` (after line 137's closing `});`), add:

```ts
  mainWindow.on('close', (event) => {
    if (sessionActive) {
      event.preventDefault();
      mainWindow?.hide();
    }
  });
```

Change `startEmbeddedServer()`'s `spawn` call to detach the child process so it can outlive a window close:

```ts
  serverProcess = spawn(process.execPath, [serverEntry], {
    env: {
      ...process.env,
      ELECTRON_RUN_AS_NODE: '1',
      SERVE_STATIC_DIR: staticDir,
      PORT: String(SERVER_PORT),
      DESKTOP_MODE: '1',
    },
    stdio: 'inherit',
    detached: true,
  });
  serverProcess.unref();
```

(The rest of `startEmbeddedServer`/`stopEmbeddedServer` is unchanged — `stopEmbeddedServer` still explicitly kills it on a real quit, e.g. from the tray's "終了" item or a normal quit while no session is active.)

- [ ] **Step 4: Notify main process from the renderer whenever the session state changes**

In `GanttChartEditor/src/context/AppContext.tsx`, add an effect inside `AppProvider` (after the existing cleanup `useEffect`):

```ts
  useEffect(() => {
    window.electronAPI?.setCollabSessionActive(state.session !== null);
  }, [state.session]);
```

- [ ] **Step 5: Recompile the Electron main process and manually verify**

Run: `cd GanttChartEditor && npm run electron:compile`
Expected: compiles with no errors.

Manual check (packaged build, since `startEmbeddedServer` is a no-op in dev — see its own early-return comment):
1. `npm run electron:build`, install and launch the packaged app.
2. Load a schedule, start a session, join from a second window/tab.
3. Close the host's window (✕). Expected: window hides, a tray icon appears; the second participant keeps syncing.
4. Right-click the tray icon → "ウィンドウを開く" — the window reappears with the session still active.
5. Right-click the tray icon → "終了（セッションも終了します）" — the app fully quits and the other participant's connection drops.

- [ ] **Step 6: Commit**

```bash
git add GanttChartEditor/electron/main.cts GanttChartEditor/electron/preload.cts GanttChartEditor/src/types/electron.d.ts GanttChartEditor/src/context/AppContext.tsx
git commit -m "feat(electron): keep the collab server running when the host closes their window"
```

---

### Task 13: End-to-end manual verification pass

**Files:** none (verification only).

- [ ] **Step 1: Single-PC multi-tab check**

Run `npm run dev:all` in `GanttChartEditor`. Open `http://localhost:5173` in one tab, load a test schedule (`Test_data/EnvConfig.yaml` + `Test_data/Schedule.yaml`), start a session via 共同編集 > セッションを開始. Copy the edit link, open it in a second tab, join with a different name.
Expected: editing an assignment in either tab appears live in the other; the participant count shown next to the menu bar in both tabs reads `🟢 2人が参加中`.

- [ ] **Step 2: View-only join**

Copy the view-only link from the session panel, open it in a third tab.
Expected: read-only mirror (`pointer-events: none`), no menu bar, shows the same live content; edits made in the two edit tabs still appear here.

- [ ] **Step 3: Undo/redo convergence**

From tab 1, make two edits, then press Ctrl+Z once.
Expected: tab 2's content updates to match tab 1's post-undo state; tab 2's own undo count (visible in its status bar) is unaffected by tab 1's undo.

- [ ] **Step 4: Late joiner catch-up**

With tabs 1 and 2 already having made several edits, open a fourth tab and join as edit.
Expected: the fourth tab lands on the current, edited state — not the original baseline — and can immediately make further edits that reach the other tabs.

- [ ] **Step 5: Solo mode unaffected**

Close all session tabs, open the app fresh with no `?session=` param.
Expected: opens exactly as before this feature — no session, no collaboration UI beyond the (idle) 共同編集 menu, full local editing.

- [ ] **Step 6: Leave and rejoin**

From one tab, use 共同編集 > セッションを終了; confirm the tab reverts to normal solo editing with its current data intact, while the remaining tab(s) keep syncing with each other.

No commit for this task — record any issues found and fix them by revisiting the relevant earlier task.

---

## Self-Review Notes

- **Spec coverage:** session/action-log server (Tasks 1-2), centralized dispatch wrapping + undo/redo special-casing (Task 6), both join paths (browser link via Task 10, in-app paste via Task 8), menu-bar placement replacing the standalone button (Task 9), solo-mode-unaffected (verified in Task 13 Step 5, and structurally true throughout since nothing new activates without an explicit session), process-lifecycle fix so host-leaves doesn't end the session (Task 12) — all covered.
- **Placeholder scan:** the only intentionally minimal piece is the 1x1 tray icon in Task 12, called out explicitly as a placeholder-quality-but-functional asset, not a missing implementation — `new Tray()` works correctly with it today.
- **Type consistency:** `SessionRole`/`SessionConnectionStatus`/`SessionParticipant`/`SessionBaseline`/`SessionState` are defined once in `types/appState.ts` (Task 3) and imported everywhere else (`collabService.ts`, `AppContext.tsx`, `SessionDialog.tsx`, `App.tsx`) rather than redeclared. `LoggedAction` is defined once in `collabService.ts` (client) and separately in `sessionStore.ts` (server) since they're different processes with no shared package — both have the same `{ seq, type, payload }` shape by convention, matched by the Task 2 integration test.
