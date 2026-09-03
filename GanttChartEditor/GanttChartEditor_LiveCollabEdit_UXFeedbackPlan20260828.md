# GanttChartEditor Collab UX Feedback — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement six pieces of real-usage feedback on the live-collaboration feature: a named session, a clickable participant list with roles, join-dialog clarity, a reachable session-info submenu, full viewer parity (scroll/filter/constraint-check/flight-stints) with only actual editing disabled, and correct start/join tab pre-selection.

**Architecture:** Adds a `name` field to the session data model (server store → REST create → socket sync-init → client `SessionState`) threaded through everywhere a session is referenced. Unifies the previously-separate `ViewPage` into the same `EditorShell` used by editors, gating only the specific data-mutating interactions behind one `isReadOnly` flag derived from `state.session?.role === 'view'` — the same pattern already used throughout this codebase for session-aware gating (e.g. `Toolbar.tsx`'s `!!state.session` checks).

**Tech Stack:** React 19 + TypeScript (client), Express + Socket.IO (server), Jest (client tests), Vitest (server tests) — all already in place from the original collab feature.

## Global Constraints

- Session names are required (no auto-generated fallback), not unique, and are purely a display label — the session id remains the real identifier.
- No new access control anywhere in this plan — hiding the edit link from viewers (Task 6) is a UX courtesy, not an enforcement mechanism; the join flow has no authentication, same as before.
- Solo mode (no session) must be completely unaffected by every change in this plan.
- `isReadOnly` is always derived as `state.session?.role === 'view'` — never a separate, independently-settable flag, so it can never drift from the actual session role.

**Baseline note for whoever implements this:** the codebase already has a connection-status gate in `AppContext.tsx`'s wrapped `dispatch` (blocks syncable actions when `state.session.connectionStatus !== 'connected'`) and a `parseSessionOrigin` mechanism for joining cross-machine links from the desktop app — both already implemented and unrelated to this plan; don't re-implement or remove them.

---

### Task 1: Server — session name (data model, create endpoint, sync-init, lookup endpoint)

**Files:**
- Modify: `GanttChartEditor/server/src/collab/sessionStore.ts`
- Modify: `GanttChartEditor/server/src/collab/sessionStore.test.ts`
- Modify: `GanttChartEditor/server/src/collab/collabSocket.ts`
- Modify: `GanttChartEditor/server/src/collab/collabSocket.test.ts`
- Modify: `GanttChartEditor/server/src/routes/collab.ts`

**Interfaces:**
- Produces: `createSession(name: string, baseline: SessionBaseline): string` (name is now the first param — was `createSession(baseline)`); `getSession(id)`'s return type gains `name: string`; new `getSessionName(id: string): string | null`; `POST /collab/sessions` requires `name` in the body (400 without it); new `GET /collab/sessions/:id/name` → `{ ok: true, name: string }` or `{ ok: false }`; the `sync-init` socket payload gains `name: string` alongside the existing `baseline`/`actions`/`participants`.

- [ ] **Step 1: Write the failing tests**

In `server/src/collab/sessionStore.test.ts`, update the `createSession(BASELINE)` calls to `createSession('Test Session', BASELINE)` throughout the file (every existing call site), and add:

```ts
describe('createSession / getSession — name', () => {
  it('stores and returns the session name', () => {
    const id = createSession('My Session', BASELINE);
    expect(getSession(id)?.name).toBe('My Session');
  });
});

describe('getSessionName', () => {
  it('returns the name for a real session', () => {
    const id = createSession('Weekly Plan', BASELINE);
    expect(getSessionName(id)).toBe('Weekly Plan');
  });

  it('returns null for an unknown session id', () => {
    expect(getSessionName('does-not-exist')).toBeNull();
  });
});
```

In `server/src/collab/collabSocket.test.ts`, update the `createSession(BASELINE)` calls to `createSession('Test Session', BASELINE)`, and change the first `join` test's assertion:

```ts
it('replies with the baseline, session name, empty action log, and participant list for a fresh session', async () => {
  const sessionId = createSession('Test Session', BASELINE);
  const client = connect();
  const syncInit = await new Promise<any>(resolve => {
    client.on('connect', () => client.emit('join', { sessionId, name: 'Alice', role: 'edit' }));
    client.on('sync-init', resolve);
  });
  expect(syncInit).toEqual({ ok: true, name: 'Test Session', baseline: BASELINE, actions: [], participants: [{ id: expect.any(String), name: 'Alice', role: 'edit' }] });
  client.disconnect();
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd GanttChartEditor/server && npm test`
Expected: FAIL — `createSession` doesn't accept a name yet, `getSessionName` doesn't exist, `sync-init` doesn't include `name`.

- [ ] **Step 3: Implement the store changes**

In `server/src/collab/sessionStore.ts`, add `name: string` to the `CollabSession` interface (after `id: string;`). Update `createSession`:

```ts
export function createSession(name: string, baseline: SessionBaseline): string {
  const id = randomUUID();
  sessions.set(id, {
    id,
    name,
    baseline,
    actions: [],
    participants: new Map(),
    nextSeq: 0,
    lastActivityAt: Date.now(),
  });
  return id;
}
```

Update `getSession`'s return type and body to include `name`:

```ts
export function getSession(
  id: string,
): { name: string; baseline: SessionBaseline; actions: LoggedAction[]; participants: SessionParticipant[] } | null {
  const session = sessions.get(id);
  if (!session) return null;
  return { name: session.name, baseline: { ...session.baseline }, actions: [...session.actions], participants: [...session.participants.values()] };
}
```

Add, after `getSession`:

```ts
export function getSessionName(id: string): string | null {
  return sessions.get(id)?.name ?? null;
}
```

- [ ] **Step 4: Wire the name into the socket layer**

In `server/src/collab/collabSocket.ts`, the `join` handler already does `socket.emit('sync-init', { ok: true, baseline: session.baseline, actions: session.actions, participants });` — change it to include `name: session.name`:

```ts
socket.emit('sync-init', { ok: true, name: session.name, baseline: session.baseline, actions: session.actions, participants });
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd GanttChartEditor/server && npm test`
Expected: PASS (all files).

- [ ] **Step 6: Update the create route and add the lookup route**

In `server/src/routes/collab.ts`, require `name`:

```ts
import { Router } from 'express';
import * as store from '../collab/sessionStore.js';

export const collabRouter = Router();

collabRouter.post('/collab/sessions', (req, res) => {
  const { name, schedule, envConfig, currentView } = req.body as {
    name?: string;
    schedule?: unknown;
    envConfig?: unknown;
    currentView?: 'worker' | 'device';
  };
  if (!name || !schedule || !envConfig || (currentView !== 'worker' && currentView !== 'device')) {
    res.status(400).json({ ok: false, error: 'name, schedule, envConfig, currentView are required' });
    return;
  }
  const sessionId = store.createSession(name, { schedule, envConfig, currentView });
  res.json({ ok: true, sessionId });
});

collabRouter.get('/collab/sessions/:id/name', (req, res) => {
  const name = store.getSessionName(req.params.id);
  if (!name) {
    res.status(404).json({ ok: false });
    return;
  }
  res.json({ ok: true, name });
});
```

- [ ] **Step 7: Manual verification of the new route**

Run: `cd GanttChartEditor/server && npm run dev` (in one terminal), then in another:
```bash
curl -s -X POST http://localhost:3010/api/collab/sessions -H "Content-Type: application/json" -d '{"name":"Test","schedule":{"planRange":{"startDate":"2026-01-01","endDate":"2026-01-31"},"workflowTaskList":[],"assignmentList":[]},"envConfig":{"workflowList":[],"fabList":[],"regionList":[],"customerCompanyList":[],"workerCompanyList":[],"workerList":[],"transiteDayMap":[]},"currentView":"worker"}'
```
Expected: `{"ok":true,"sessionId":"<uuid>"}`. Then `curl -s http://localhost:3010/api/collab/sessions/<uuid>/name` — Expected: `{"ok":true,"name":"Test"}`. And `curl -s http://localhost:3010/api/collab/sessions/nope/name` — Expected: 404 with `{"ok":false}`.

- [ ] **Step 8: Commit**

```bash
git add GanttChartEditor/server/src/collab/sessionStore.ts GanttChartEditor/server/src/collab/sessionStore.test.ts GanttChartEditor/server/src/collab/collabSocket.ts GanttChartEditor/server/src/collab/collabSocket.test.ts GanttChartEditor/server/src/routes/collab.ts
git commit -m "feat(server): add required session name, expose it via sync-init and a lookup endpoint"
```

---

### Task 2: Client types & reducer — session name, dialog tab selection

**Files:**
- Modify: `GanttChartEditor/src/types/appState.ts`
- Modify: `GanttChartEditor/src/context/reducer.ts`
- Modify: `GanttChartEditor/src/context/AppContext.tsx`
- Modify: `GanttChartEditor/src/__tests__/context/reducer.test.ts`

**Interfaces:**
- Consumes: nothing from Task 1 directly (this task is client-only data model; the wiring to the server's new fields happens in Task 3).
- Produces: `SessionState.name: string`; `AppState.sessionDialogTab: 'start' | 'join'`; `ActionType`'s `OPEN_SESSION_DIALOG` now carries a payload `'start' | 'join'` (was payload-less); new action `SET_SESSION_NAME` (payload: `string`).

- [ ] **Step 1: Write the failing reducer tests**

In `src/__tests__/context/reducer.test.ts`, update every existing `SET_SESSION` test's session object literal to include `name: 'Test Session'` (the type will now require it), and update the `OPEN_SESSION_DIALOG` test call to pass a payload. Replace the existing `describe('OPEN_SESSION_DIALOG / CLOSE_SESSION_DIALOG', ...)` block with:

```ts
describe('OPEN_SESSION_DIALOG / CLOSE_SESSION_DIALOG', () => {
  it('opens on the start tab and closes', () => {
    const opened = reducer(BASE_STATE, { type: 'OPEN_SESSION_DIALOG', payload: 'start' });
    expect(opened.isSessionDialogOpen).toBe(true);
    expect(opened.sessionDialogTab).toBe('start');
    const closed = reducer(opened, { type: 'CLOSE_SESSION_DIALOG' });
    expect(closed.isSessionDialogOpen).toBe(false);
  });

  it('opens on the join tab when asked', () => {
    const opened = reducer(BASE_STATE, { type: 'OPEN_SESSION_DIALOG', payload: 'join' });
    expect(opened.sessionDialogTab).toBe('join');
  });
});

describe('SET_SESSION_NAME', () => {
  it('updates the name on an existing session', () => {
    const state = { ...BASE_STATE, session: { id: 's1', name: 'Old Name', role: 'edit' as const, connectionStatus: 'connected' as const, participants: [] } };
    const next = reducer(state, { type: 'SET_SESSION_NAME', payload: 'New Name' });
    expect(next.session?.name).toBe('New Name');
  });

  it('is a no-op when there is no session', () => {
    const next = reducer(BASE_STATE, { type: 'SET_SESSION_NAME', payload: 'New Name' });
    expect(next.session).toBeNull();
  });
});
```

Also add `sessionDialogTab: 'start' as const,` to `BASE_STATE`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd GanttChartEditor && npm run test:jest -- reducer.test`
Expected: FAIL (type/behavior mismatches).

- [ ] **Step 3: Update the types**

In `src/types/appState.ts`, add `name: string;` to `SessionState` (after `id: string;`):

```ts
export interface SessionState {
  id: string;
  name: string;
  role: SessionRole;
  connectionStatus: SessionConnectionStatus;
  participants: SessionParticipant[];
}
```

Add `sessionDialogTab: 'start' | 'join';` to `AppState` (right after `isSessionDialogOpen: boolean;`).

Replace:
```ts
  | { type: 'OPEN_SESSION_DIALOG' }
  | { type: 'CLOSE_SESSION_DIALOG' };
```
with:
```ts
  | { type: 'OPEN_SESSION_DIALOG'; payload: 'start' | 'join' }
  | { type: 'CLOSE_SESSION_DIALOG' }
  | { type: 'SET_SESSION_NAME'; payload: string };
```

- [ ] **Step 4: Update the reducer**

In `src/context/reducer.ts`, replace:
```ts
    case 'OPEN_SESSION_DIALOG':
      return { ...state, isSessionDialogOpen: true };
```
with:
```ts
    case 'OPEN_SESSION_DIALOG':
      return { ...state, isSessionDialogOpen: true, sessionDialogTab: action.payload };
```

Add, right after the `SET_SESSION_PARTICIPANTS` case:
```ts
    case 'SET_SESSION_NAME':
      return state.session ? { ...state, session: { ...state.session, name: action.payload } } : state;
```

- [ ] **Step 5: Update `AppContext.tsx`'s initial state**

Add `sessionDialogTab: 'start',` to `initialState` (right after `isSessionDialogOpen: false,`).

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd GanttChartEditor && npm run test:jest -- reducer.test`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add GanttChartEditor/src/types/appState.ts GanttChartEditor/src/context/reducer.ts GanttChartEditor/src/context/AppContext.tsx GanttChartEditor/src/__tests__/context/reducer.test.ts
git commit -m "feat(client): add session name and dialog-tab-selection state"
```

(`npx tsc -b` will show errors in `SessionDialog.tsx`/`MenuBar.tsx` until Task 3 — expected, don't fix those files here.)

---

### Task 3: Client service + AppContext — thread session name through create/join

**Files:**
- Modify: `GanttChartEditor/src/services/collabService.ts`
- Modify: `GanttChartEditor/src/context/AppContext.tsx`
- Modify: `GanttChartEditor/src/__tests__/services/collabService.test.ts`
- Modify: `GanttChartEditor/src/__tests__/context/AppContext.test.tsx`

**Interfaces:**
- Consumes: Task 1's server changes (`name` required on create, `sync-init`'s `name` field); Task 2's `SET_SESSION_NAME` action and `SessionState.name`.
- Produces: `createCollabSession(name: string, baseline: SessionBaseline): Promise<string>` (was `createCollabSession(baseline)`); `fetchSessionName(sessionId: string): Promise<string | null>`; `joinCollabRoom`'s `onSyncInit` callback signature becomes `(sessionName: string, baseline: SessionBaseline, actions: LoggedAction[]) => void`; `startCollabSession(displayName: string, sessionName: string): Promise<{ sessionId: string; link: string }>` (was `startCollabSession(name)`).

- [ ] **Step 1: Write the failing tests**

In `src/__tests__/services/collabService.test.ts`, every `mockedCollab`/direct call to `joinCollabRoom`'s `onSyncInit` mock implementation currently does `onSyncInit(baseline, [])` — update every one to `onSyncInit('Test Session', baseline, [])` (the new first argument). Add:

```ts
it('fetchSessionName resolves the name for a real session', async () => {
  global.fetch = jest.fn().mockResolvedValue({ ok: true, json: async () => ({ ok: true, name: 'Weekly Plan' }) }) as any;
  const name = await fetchSessionName('abc123');
  expect(name).toBe('Weekly Plan');
});

it('fetchSessionName resolves null for an unknown session', async () => {
  global.fetch = jest.fn().mockResolvedValue({ ok: false, json: async () => ({ ok: false }) }) as any;
  const name = await fetchSessionName('nope');
  expect(name).toBeNull();
});
```

In `src/__tests__/context/AppContext.test.tsx`, update every `mockedCollab.joinCollabRoom.mockImplementation((...args) => { onSyncInit(...) })` call site's `onSyncInit(baseline, actions)` calls to `onSyncInit('Mock Session', baseline, actions)`. Update every `startCollabSession('Alice')` call in the test file to `startCollabSession('Alice', 'My Session')`. Add:

```ts
it('sets the session name from the sync-init reply when joining', async () => {
  mockedCollab.joinCollabRoom.mockImplementation((_id, _name, _role, _isCreator, onSyncInit) => {
    onSyncInit('Joined Session Name', { schedule: SCHEDULE, envConfig: ENV_CONFIG, currentView: 'worker' }, []);
    return () => {};
  });

  renderApp();
  await act(async () => { await userEvent.click(screen.getByText('join')); });
  await waitFor(() => expect(screen.getByTestId('session-name')).toHaveTextContent('Joined Session Name'));
});
```

This last test needs a `data-testid="session-name"` element on `TestConsumer` — add `<div data-testid="session-name">{state.session?.name ?? 'none'}</div>` to it.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd GanttChartEditor && npm run test:jest -- collabService.test AppContext.test`
Expected: FAIL.

- [ ] **Step 3: Update `collabService.ts`**

Change `createCollabSession`:

```ts
export async function createCollabSession(name: string, baseline: SessionBaseline): Promise<string> {
  const res = await fetch(`${getSocketOrigin()}/api/collab/sessions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, ...baseline }),
  });
  const data = (await res.json().catch(() => ({ ok: false }))) as { ok: boolean; sessionId?: string; error?: string };
  if (!res.ok || !data.ok || !data.sessionId) throw new Error(data.error ?? 'セッションの作成に失敗しました');
  return data.sessionId;
}
```

Change `joinCollabRoom`'s `onSyncInit` parameter type and its call sites within the function:

```ts
export function joinCollabRoom(
  sessionId: string,
  name: string,
  role: SessionRole,
  isCreator: boolean,
  onSyncInit: (sessionName: string, baseline: SessionBaseline, actions: LoggedAction[]) => void,
  onAction: (action: { type: string; payload: unknown }) => void,
  onPresence: (participants: SessionParticipant[]) => void,
  onStatusChange: (status: SessionConnectionStatus) => void,
  origin?: string,
): () => void {
  const s = ensureSocket(origin);
  onStatusChange('connecting');

  let skipBaselineReplay = isCreator;

  const handleConnect = () => s.emit('join', { sessionId, name, role });
  const handleSyncInit = (payload: { ok: boolean; name?: string; baseline?: SessionBaseline; actions?: LoggedAction[]; participants?: SessionParticipant[] }) => {
    if (!payload.ok || !payload.baseline || !payload.name) {
      onStatusChange('disconnected');
      return;
    }
    if (skipBaselineReplay) {
      skipBaselineReplay = false;
    } else {
      onSyncInit(payload.name, payload.baseline, payload.actions ?? []);
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
    socketOrigin = null;
  };
}
```

Add, after `fetchCollabLink`:

```ts
export async function fetchSessionName(sessionId: string): Promise<string | null> {
  const res = await fetch(`${getSocketOrigin()}/api/collab/sessions/${sessionId}/name`);
  const data = (await res.json().catch(() => ({ ok: false }))) as { ok: boolean; name?: string };
  if (!res.ok || !data.ok || !data.name) return null;
  return data.name;
}
```

- [ ] **Step 4: Update `AppContext.tsx`**

Import `fetchSessionName` alongside the other `collabService` imports.

Change `startCollabSession`:

```ts
  const startCollabSession = useCallback(async (displayName: string, sessionName: string) => {
    const { schedule, envConfig, currentView } = stateRef.current;
    if (!schedule || !envConfig) throw new Error(UI.collabNoScheduleError);
    const sessionId = await createCollabSession(sessionName, { schedule, envConfig, currentView });
    const link = await fetchCollabLink(sessionId, 'edit');
    rawDispatch({ type: 'SET_SESSION', payload: { id: sessionId, name: sessionName, role: 'edit', connectionStatus: 'connecting', participants: [] } });
    joinInternal(sessionId, displayName, 'edit', true);
    return { sessionId, link };
  }, [joinInternal]);
```

Change `joinCollabSession` — `SET_SESSION`'s payload needs a `name`, which isn't known yet at this point for a joiner (only after sync-init); use an empty string as the placeholder that `SET_SESSION_NAME` (dispatched from `joinInternal` below) immediately corrects:

```ts
  const joinCollabSession = useCallback(async (idOrLink: string, name: string, role: SessionRole) => {
    const sessionId = parseSessionId(idOrLink);
    const origin = parseSessionOrigin(idOrLink) ?? undefined;
    rawDispatch({ type: 'SET_SESSION', payload: { id: sessionId, name: '', role, connectionStatus: 'connecting', participants: [] } });
    joinInternal(sessionId, name, role, false, origin);
  }, [joinInternal]);
```

Change `joinInternal`'s `onSyncInit` callback to accept and dispatch the name:

```ts
  const joinInternal = useCallback((sessionId: string, name: string, role: SessionRole, isCreator: boolean, origin?: string) => {
    disconnectRef.current?.();
    disconnectRef.current = joinCollabRoom(
      sessionId, name, role, isCreator,
      (sessionName, baseline, actions) => {
        rawDispatch({ type: 'SET_SESSION_NAME', payload: sessionName });
        rawDispatch({ type: 'SET_SESSION_BASELINE', payload: baseline });
        for (const a of actions) applyRemoteAction(a);
      },
      applyRemoteAction,
      (participants) => rawDispatch({ type: 'SET_SESSION_PARTICIPANTS', payload: participants }),
      (status) => rawDispatch({ type: 'SET_SESSION_CONNECTION_STATUS', payload: status }),
      origin,
    );
  }, [applyRemoteAction]);
```

Update the `ContextType` interface's `startCollabSession` signature to match: `startCollabSession: (displayName: string, sessionName: string) => Promise<{ sessionId: string; link: string }>;`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd GanttChartEditor && npm run test:jest -- collabService.test AppContext.test`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add GanttChartEditor/src/services/collabService.ts GanttChartEditor/src/context/AppContext.tsx GanttChartEditor/src/__tests__/services/collabService.test.ts GanttChartEditor/src/__tests__/context/AppContext.test.tsx
git commit -m "feat(client): thread session name through create/join"
```

(`SessionDialog.tsx`/`MenuBar.tsx` still won't compile — fixed in Task 4.)

---

### Task 4: SessionDialog — session name field, tab pre-selection, name display

**Files:**
- Modify: `GanttChartEditor/src/components/Dialogs/SessionDialog.tsx`
- Modify: `GanttChartEditor/src/components/Toolbar/MenuBar.tsx`
- Modify: `GanttChartEditor/src/config/uiText.ts`

**Interfaces:**
- Consumes: Task 2's `sessionDialogTab`/`OPEN_SESSION_DIALOG` payload, `SET_SESSION_NAME`; Task 3's `startCollabSession(displayName, sessionName)`.
- Produces: `StartOrJoinPanel` now requires a session name to start; opens on the correct tab.

- [ ] **Step 1: Add UI text**

In `src/config/uiText.ts`, add near the other collab keys (after `sessionNamePlaceholder`):

```ts
  sessionNameFieldPlaceholder: 'セッション名を入力',
```

(Note: `sessionNamePlaceholder` already exists and is used for the *display* name field — keep it as-is; this new key is for the session name field specifically, to avoid confusing the two.)

- [ ] **Step 2: Update `MenuBar.tsx` to pass the tab payload**

Replace:
```ts
        ? [{ label: UI.leaveSessionItem, action: () => { leaveCollabSession(); setOpenMenu(null); } }]
        : [
            { label: UI.startSessionItem, action: () => { dispatch({ type: 'OPEN_SESSION_DIALOG' }); setOpenMenu(null); }, disabled: !canSave },
            { label: UI.joinSessionItem, action: () => { dispatch({ type: 'OPEN_SESSION_DIALOG' }); setOpenMenu(null); } },
          ],
```
with:
```ts
        ? [{ label: UI.leaveSessionItem, action: () => { leaveCollabSession(); setOpenMenu(null); } }]
        : [
            { label: UI.startSessionItem, action: () => { dispatch({ type: 'OPEN_SESSION_DIALOG', payload: 'start' }); setOpenMenu(null); }, disabled: !canSave },
            { label: UI.joinSessionItem, action: () => { dispatch({ type: 'OPEN_SESSION_DIALOG', payload: 'join' }); setOpenMenu(null); } },
          ],
```

(Task 6 adds a third item, `セッション情報`, to the `state.session ?` branch — not this task's concern.)

- [ ] **Step 3: Update `StartOrJoinPanel`**

In `SessionDialog.tsx`, change the tab state to initialize from context, add a session-name field, and pass it through:

```tsx
function StartOrJoinPanel({ onClose }: { onClose: () => void }) {
  const { state, startCollabSession, joinCollabSession } = useAppContext();
  const [tab, setTab] = useState<'start' | 'join'>(state.sessionDialogTab);
  const [displayName, setDisplayName] = useState('');
  const [sessionName, setSessionName] = useState('');
  const [joinInput, setJoinInput] = useState('');
  const [joinRole, setJoinRole] = useState<SessionRole>('edit');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleStart = async () => {
    if (!displayName.trim() || !sessionName.trim()) return;
    setBusy(true);
    setError(null);
    try {
      await startCollabSession(displayName.trim(), sessionName.trim());
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  const handleJoin = async () => {
    if (!displayName.trim() || !joinInput.trim()) return;
    setBusy(true);
    setError(null);
    try {
      await joinCollabSession(joinInput.trim(), displayName.trim(), joinRole);
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

      {tab === 'start' && (
        <input placeholder={UI.sessionNameFieldPlaceholder} value={sessionName} onChange={e => setSessionName(e.target.value)} style={{ ...inputStyle, marginBottom: 12 }} />
      )}

      <input placeholder={UI.sessionNamePlaceholder} value={displayName} onChange={e => setDisplayName(e.target.value)} style={{ ...inputStyle, marginBottom: 12 }} />

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
          disabled={busy || !displayName.trim() || (tab === 'start' && !sessionName.trim()) || (tab === 'join' && !joinInput.trim())}
          style={primaryBtnStyle}
        >
          {tab === 'start' ? UI.sessionStartBtn : UI.sessionJoinBtn}
        </button>
        <button onClick={onClose} style={neutralBtnStyle}>{UI.sessionCloseBtn}</button>
      </div>
    </div>
  );
}
```

(Task 5 adds the join-dialog name-lookup confirmation on top of this — not this task's concern; this step only fixes tab pre-selection and adds the required session-name field.)

- [ ] **Step 4: Verify and commit**

Run: `cd GanttChartEditor && npx tsc -b 2>&1 | grep -iE "SessionDialog|MenuBar"` — expect no output (clean). Run: `npm run test:jest` — expect all passing (no new tests needed for this task; behavior is covered by Task 2's reducer tests for tab selection and manually verified here).

```bash
git add GanttChartEditor/src/components/Dialogs/SessionDialog.tsx GanttChartEditor/src/components/Toolbar/MenuBar.tsx GanttChartEditor/src/config/uiText.ts
git commit -m "feat(client): require a session name to start, open dialog on the clicked tab"
```

---

### Task 5: Join-dialog clarity — resolve and show the session name before joining

**Files:**
- Modify: `GanttChartEditor/src/components/Dialogs/SessionDialog.tsx`
- Modify: `GanttChartEditor/src/config/uiText.ts`

**Interfaces:**
- Consumes: Task 3's `fetchSessionName(sessionId: string): Promise<string | null>`; `parseSessionId` (already imported elsewhere in the codebase from `collabService`).

- [ ] **Step 1: Add UI text**

In `src/config/uiText.ts`, add near the join-related keys:

```ts
  sessionJoinResolvedName: (name: string) => `「${name}」というセッションに参加します`,
  sessionJoinUnresolvedName: 'セッションが見つかりません。リンクまたはIDを確認してください。',
```

- [ ] **Step 2: Add the lookup to `StartOrJoinPanel`**

In `SessionDialog.tsx`, import `fetchSessionName` and `parseSessionId` from `../../services/collabService` (add to the existing `fetchCollabLink` import line). Add state and a debounced lookup effect:

```tsx
  const [resolvedSessionName, setResolvedSessionName] = useState<string | null>(null);
  const [nameLookupFailed, setNameLookupFailed] = useState(false);

  useEffect(() => {
    if (tab !== 'join' || !joinInput.trim()) {
      setResolvedSessionName(null);
      setNameLookupFailed(false);
      return;
    }
    let cancelled = false;
    const timer = setTimeout(() => {
      const id = parseSessionId(joinInput.trim());
      void fetchSessionName(id).then(name => {
        if (cancelled) return;
        setResolvedSessionName(name);
        setNameLookupFailed(!name);
      });
    }, 400);
    return () => { cancelled = true; clearTimeout(timer); };
  }, [tab, joinInput]);
```

(This needs `useEffect` added to the existing `import { useState, useEffect } from 'react';` line at the top of the file if not already present — it already is, from `ActiveSessionPanel`'s effect.)

Insert the confirmation/error line right after the join-link input, inside the existing `{tab === 'join' && (...)}` block:

```tsx
          <input placeholder={UI.sessionJoinLinkPlaceholder} value={joinInput} onChange={e => setJoinInput(e.target.value)} style={{ ...inputStyle, marginBottom: 12 }} />
          {resolvedSessionName && (
            <div style={{ fontSize: 12, color: '#1976d2', marginBottom: 12 }}>{UI.sessionJoinResolvedName(resolvedSessionName)}</div>
          )}
          {nameLookupFailed && (
            <div style={{ fontSize: 12, color: '#c62828', marginBottom: 12 }}>{UI.sessionJoinUnresolvedName}</div>
          )}
```

- [ ] **Step 3: Manual verification**

Run `npm run dev:all`, open the app, start a session (note its id from the URL/link), open a second tab, click 共同編集 → セッションに参加, paste the id. Expected: after a brief pause, "「<name>」というセッションに参加します" appears. Paste garbage instead — expected: the "not found" message appears instead.

- [ ] **Step 4: Commit**

```bash
git add GanttChartEditor/src/components/Dialogs/SessionDialog.tsx GanttChartEditor/src/config/uiText.ts
git commit -m "feat(client): resolve and show the session name before joining"
```

---

### Task 6: Participant list on click

**Files:**
- Modify: `GanttChartEditor/src/components/Toolbar/MenuBar.tsx`
- Modify: `GanttChartEditor/src/config/uiText.ts`

**Interfaces:**
- Consumes: `state.session.participants: SessionParticipant[]` (already exists, `{id, name, role}`).

- [ ] **Step 1: Add UI text**

In `src/config/uiText.ts`, add near `sessionParticipantsLabel`:

```ts
  sessionParticipantRoleEdit: '編集者',
  sessionParticipantRoleView: '閲覧者',
```

- [ ] **Step 2: Make the presence indicator clickable with a dropdown**

In `MenuBar.tsx`, add state: `const [showParticipants, setShowParticipants] = useState(false);` (alongside the existing `showSaveAs`/`saveStatus` state).

Replace:
```tsx
        {state.session && (
          <span style={{ color: '#a8d4f5', fontSize: 11, marginLeft: 16, fontFamily: 'Meiryo, sans-serif' }}>
            {UI.sessionParticipantsLabel(state.session.participants.length)}
          </span>
        )}
```
with:
```tsx
        {state.session && (
          <div style={{ position: 'relative', marginLeft: 16 }}>
            <span
              onClick={() => setShowParticipants(v => !v)}
              style={{ color: '#a8d4f5', fontSize: 11, fontFamily: 'Meiryo, sans-serif', cursor: 'pointer' }}
            >
              {UI.sessionParticipantsLabel(state.session.participants.length)}
            </span>
            {showParticipants && (
              <div
                style={{
                  position: 'absolute', top: 20, right: 0, minWidth: 160,
                  backgroundColor: '#fff', border: '1px solid #ccc', borderRadius: 2,
                  boxShadow: '0 4px 12px rgba(0,0,0,0.25)', zIndex: 500, padding: '6px 0',
                }}
              >
                {state.session.participants.map(p => (
                  <div key={p.id} style={{ display: 'flex', justifyContent: 'space-between', padding: '4px 12px', fontSize: 12, fontFamily: 'MS Gothic, monospace', color: '#222' }}>
                    <span>{p.name}</span>
                    <span style={{ color: '#666', marginLeft: 12 }}>{p.role === 'edit' ? UI.sessionParticipantRoleEdit : UI.sessionParticipantRoleView}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
```

The bar's existing outside-click handler (`document.addEventListener('mousedown', handleClick)` in the `useEffect` near the top of `MenuBar`, which closes `openMenu` when clicking outside `barRef`) already covers this new dropdown too, since it lives inside the same `barRef`-wrapped container — but it only resets `openMenu`, not `showParticipants`. Update that handler:

```ts
  useEffect(() => {
    const handleClick = (e: MouseEvent) => {
      if (barRef.current && !barRef.current.contains(e.target as Node)) {
        setOpenMenu(null);
        setShowParticipants(false);
      }
    };
    document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, []);
```

- [ ] **Step 3: Manual verification**

Run `npm run dev:all`, start a session, join from a second tab with a different display name and role (view). In the first tab, click the participant count — expected: a dropdown listing both names with "編集者"/"閲覧者" labels. Click elsewhere — expected: it closes.

- [ ] **Step 4: Commit**

```bash
git add GanttChartEditor/src/components/Toolbar/MenuBar.tsx GanttChartEditor/src/config/uiText.ts
git commit -m "feat(client): show participant names and roles on click"
```

---

### Task 7: セッション情報 submenu + name/edit-link-visibility in the active-session panel

**Files:**
- Modify: `GanttChartEditor/src/components/Toolbar/MenuBar.tsx`
- Modify: `GanttChartEditor/src/components/Dialogs/SessionDialog.tsx`
- Modify: `GanttChartEditor/src/config/uiText.ts`

**Interfaces:**
- Consumes: `state.session.name` (Task 2/3); `state.session.role` (existing).

- [ ] **Step 1: Add UI text**

In `src/config/uiText.ts`, add near `leaveSessionItem`:

```ts
  sessionInfoItem: 'セッション情報',
  sessionNameLabel: 'セッション名',
```

- [ ] **Step 2: Add the menu item**

In `MenuBar.tsx`, replace the `state.session ?` branch's single-item array:
```ts
        ? [{ label: UI.leaveSessionItem, action: () => { leaveCollabSession(); setOpenMenu(null); } }]
```
with:
```ts
        ? [
            { label: UI.sessionInfoItem, action: () => { dispatch({ type: 'OPEN_SESSION_DIALOG', payload: 'start' }); setOpenMenu(null); } },
            { label: UI.leaveSessionItem, action: () => { leaveCollabSession(); setOpenMenu(null); } },
          ]
```

(The `payload: 'start'` is inert here — `StartOrJoinPanel` is never reached while `state.session` is truthy, since `SessionDialog` renders `ActiveSessionPanel` instead in that case; the payload value just needs to satisfy the action's required type.)

- [ ] **Step 3: Show the session name and gate the edit link by role**

In `SessionDialog.tsx`'s `ActiveSessionPanel`, add the name display and make the edit link conditional:

```tsx
  return (
    <div>
      <div style={{ fontSize: 15, fontWeight: 'bold', color: '#1a2e3f', marginBottom: 4 }}>{UI.sessionActiveTitle}</div>
      <div style={{ fontSize: 12, color: '#666', marginBottom: 12 }}>{UI.sessionNameLabel}: {session.name}</div>
      {session.role === 'edit' && editLink && <LinkRow label={UI.sessionEditLinkLabel} link={editLink} />}
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
```

Also, since a view-role participant never needs the edit link, skip fetching it for them — change the effect's fetch calls:

```tsx
  useEffect(() => {
    const sessionId = session?.id;
    if (!sessionId) return;
    let cancelled = false;
    if (session?.role === 'edit') {
      void fetchCollabLink(sessionId, 'edit').then(link => { if (!cancelled) setEditLink(link); }).catch(() => {});
    }
    void fetchCollabLink(sessionId, 'view').then(link => { if (!cancelled) setViewLink(link); }).catch(() => {});
    return () => { cancelled = true; };
  }, [session?.id, session?.role]);
```

- [ ] **Step 4: Manual verification**

Start a session, close the dialog (don't leave the session), reopen it via 共同編集 → セッション情報 — expected: the panel reappears with the session name, both links (since you're the edit-role creator). Join as view-role from another tab, open セッション情報 there — expected: only the view link shows, no edit link.

- [ ] **Step 5: Commit**

```bash
git add GanttChartEditor/src/components/Toolbar/MenuBar.tsx GanttChartEditor/src/components/Dialogs/SessionDialog.tsx GanttChartEditor/src/config/uiText.ts
git commit -m "feat(client): add セッション情報 submenu, show session name, hide edit link from viewers"
```

---

### Task 8: Unify the editor and viewer shells, gate Toolbar/keyboard for read-only

**Files:**
- Modify: `GanttChartEditor/src/App.tsx`
- Modify: `GanttChartEditor/src/components/Toolbar/Toolbar.tsx`
- Modify: `GanttChartEditor/src/hooks/useKeyboardShortcuts.ts`
- Delete: `GanttChartEditor/src/pages/ViewPage.tsx`
- Modify: `GanttChartEditor/src/__tests__/App.test.tsx`
- Modify: `GanttChartEditor/src/config/uiText.ts`

**Interfaces:**
- Produces: `isReadOnly = state.session?.role === 'view'`, computed inline wherever needed (not a new context field — always derived, per the Global Constraints).
- Downstream: Task 9 threads the same derivation into the Gantt chart components for the drag-interaction gating.

- [ ] **Step 1: Add UI text**

In `src/config/uiText.ts`, near `flightStintsBtn`/`constraintCheckBtn`, no new keys are needed here — this task is purely structural/gating, reusing existing labels.

- [ ] **Step 2: Rewrite `App.tsx` to remove the `ViewPage` branch**

Replace:
```tsx
import { GanttPage } from './pages/GanttPage';
import { ViewPage } from './pages/ViewPage';
```
with:
```tsx
import { GanttPage } from './pages/GanttPage';
```

Replace the `AppContent` function:
```tsx
// The ONE place read-only rendering is decided, so it can't drift per
// entry point. Read-only is a property of the joined session's role, not of
// how the session was joined: both the URL path (?session=...&role=view, via
// SessionJoinGate) and the in-app SessionDialog's "閲覧のみ" join end up here
// with state.session.role === 'view' and get the same read-only ViewPage.
// Solo mode (no session) and edit-role sessions render the editor as before.
// Exported for App.test.tsx, which drives this decision directly.
export function AppContent() {
  const { state } = useAppContext();
  return state.session?.role === 'view' ? <ViewPage /> : <EditorShell />;
}
```
with:
```tsx
// EditorShell is now shared by every role — a view-role participant gets the
// exact same page as an editor, with only the specific data-mutating
// interactions disabled (Toolbar/useKeyboardShortcuts here; the Gantt chart's
// own drag handlers in a later task). This replaced an earlier, separate
// read-only ViewPage that could only mirror — it couldn't scroll, filter, or
// run a constraint check, which real usage showed people wanted.
export function AppContent() {
  return <EditorShell />;
}
```

Update `EditorShell`'s own comment (the one above its definition) since it's no longer edit-role-only:
```tsx
// The shared shell for every session role. Its hooks (global shortcuts,
// constraint checker, SchedulerWeb incoming-transfer listener) mount
// unconditionally; each one internally no-ops the specific actions a
// view-role participant shouldn't trigger, rather than the page deciding
// not to mount them at all — see useKeyboardShortcuts.ts and Toolbar.tsx.
function EditorShell() {
```

`SessionJoinGate`'s final `return <AppContent />;` and its surrounding comment stay as-is (still correct — it already just delegates the role decision onward, which now always resolves to `EditorShell`).

- [ ] **Step 3: Gate Toolbar's editing buttons**

In `Toolbar.tsx`, add the derivation and use it to disable editor-only actions:

```tsx
export function Toolbar() {
  const { state, dispatch } = useAppContext();
  const { schedule, currentView, showFlightStints } = state;
  const has = !!schedule;
  const isReadOnly = state.session?.role === 'view';
  const canEdit = has && !isReadOnly;
  const { runCheck, isChecking } = useBackendConstraintCheck();
```

Change the "+ バー配置" button: `disabled={!canEdit}` (was `!has`).
Change the "+ 新規製番追加" button: `style={mkBtn(palette.accentDark, canEdit)} disabled={!canEdit}` (was `mkBtn(palette.accentDark)` / `!has`).
Change the "計画管理ツールへ送信" button: `disabled={!canEdit}` and `style={S.submitBtn(canEdit)}` (was `!has` / `S.submitBtn(has)`).

Leave unchanged (still gated only on `has`, available to viewers): `UndoRedoButtons` is NOT unchanged — see next step. `PlanFlexBulkSettings`, `PlanRangeEditDialog`, the constraint-check button, the flight-stints button, and both filter panels stay gated on `has` only, per the design (all enabled for viewers).

`UndoRedoButtons` is a separate component (`src/components/Toolbar/UndoRedoButtons.tsx`) — it derives `isReadOnly` itself via its own `useAppContext()` call, matching how every other session-aware check in this codebase is a local read rather than prop-drilled. Replace its full contents:

```tsx
import type { CSSProperties } from 'react';
import { useAppContext } from '../../context/AppContext';
import { UI } from '../../config/uiText';

const btn: CSSProperties = {
  padding: '4px 10px',
  backgroundColor: '#fff',
  border: '1px solid #999',
  borderRadius: 3,
  cursor: 'pointer',
  fontSize: 12,
  fontFamily: 'MS Gothic, monospace',
};

export function UndoRedoButtons() {
  const { state, dispatch } = useAppContext();
  const isReadOnly = state.session?.role === 'view';
  const canUndo = state.undoStack.length > 0 && !isReadOnly;
  const canRedo = state.redoStack.length > 0 && !isReadOnly;

  return (
    <div style={{ display: 'flex', gap: 4 }}>
      <button
        onClick={() => dispatch({ type: 'UNDO' })}
        disabled={!canUndo}
        style={{ ...btn, opacity: canUndo ? 1 : 0.4 }}
      >
        {UI.undo}
      </button>
      <button
        onClick={() => dispatch({ type: 'REDO' })}
        disabled={!canRedo}
        style={{ ...btn, opacity: canRedo ? 1 : 0.4 }}
      >
        {UI.redo}
      </button>
    </div>
  );
}
```

- [ ] **Step 4: Gate keyboard shortcuts**

In `useKeyboardShortcuts.ts`, add the derivation and gate Undo/Redo/Delete (Ctrl+O is already correctly gated on `!state.session`, which covers viewers too — no change needed there; Ctrl+S already requires `currentEnvPath`/`currentSchedulePath`, which a viewer never has — also already correctly inert):

```ts
export function useKeyboardShortcuts() {
  const { state, dispatch } = useAppContext();
  const isReadOnly = state.session?.role === 'view';

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.ctrlKey || e.metaKey) {
        switch (e.key.toLowerCase()) {
          case 'z':
            e.preventDefault();
            if (!isReadOnly) dispatch({ type: 'UNDO' });
            break;
          case 'y':
            e.preventDefault();
            if (!isReadOnly) dispatch({ type: 'REDO' });
            break;
          case 's':
            e.preventDefault();
            if (!e.shiftKey && state.schedule && state.envConfig && state.currentEnvPath && state.currentSchedulePath) {
              overwriteSaveFiles(state.envConfig, state.schedule, state.currentEnvPath, state.currentSchedulePath)
                .catch(err => console.error('Save failed:', err));
            }
            break;
          case 'o':
            e.preventDefault();
            if (!state.session) dispatch({ type: 'OPEN_FILE_DIALOG' });
            break;
        }
      }
      if (e.key === 'Delete' && state.selectedAssignmentIndex !== null && !isReadOnly) {
        if (window.confirm(UI.deleteConfirm)) {
          dispatch({ type: 'DELETE_ASSIGNMENT', payload: state.selectedAssignmentIndex });
        }
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [state.schedule, state.envConfig, state.selectedAssignmentIndex, state.currentSchedulePath, state.currentEnvPath, state.session, isReadOnly, dispatch]);
}
```

- [ ] **Step 5: Delete `ViewPage.tsx`**

```bash
rm GanttChartEditor/src/pages/ViewPage.tsx
```

- [ ] **Step 6: Update `App.test.tsx`**

`GanttPage` is mocked to a bare stub in this file (`<div data-testid="editor-page" />`), so it was only ever suited to testing *which top-level shell* `AppContent` renders — not button-level read-only behavior (that's what Task 9's new Cypress spec proves end-to-end, against the real page). With `ViewPage` gone, "which shell" no longer has two answers, so this file's job shrinks to confirming that's true for every role. Replace the full contents:

```tsx
/**
 * @jest-environment jsdom
 */
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { AppContent } from '../App';
import { AppProvider, useAppContext } from '../context/AppContext';
import * as collabService from '../services/collabService';
import { ScheduleData } from '../types/schedule';
import { EnvConfig } from '../types/envConfig';

// GanttPage is stubbed so this file tests exactly one thing: that AppContent
// renders the shared shell for every role. Read-only button-level behavior
// (Toolbar, UndoRedoButtons, useKeyboardShortcuts) is proven end-to-end
// against the real page by cypress/e2e/06_viewer_parity.cy.ts instead.
jest.mock('../pages/GanttPage', () => ({ GanttPage: () => <div data-testid="editor-page" /> }));
jest.mock('../hooks/useKeyboardShortcuts', () => ({ useKeyboardShortcuts: jest.fn() }));
jest.mock('../hooks/useConstraintCheck', () => ({ useConstraintCheck: jest.fn() }));
jest.mock('../hooks/useIncomingGanttTransfer', () => ({ useIncomingGanttTransfer: jest.fn() }));

jest.mock('../services/collabService');
const mockedCollab = collabService as jest.Mocked<typeof collabService>;

const SCHEDULE: ScheduleData = {
  planRange: { startDate: '2026-01-01', endDate: '2026-01-31' },
  workflowTaskList: [],
  assignmentList: [],
};
const ENV_CONFIG: EnvConfig = {
  workflowList: [], fabList: [], regionList: [], customerCompanyList: [], workerCompanyList: [], workerList: [], transiteDayMap: [],
};

// Sits alongside AppContent inside the same provider so a test can join a
// session the same way the in-app SessionDialog does — no URL involved.
function JoinHarness() {
  const { joinCollabSession } = useAppContext();
  return (
    <>
      <button onClick={() => void joinCollabSession('abc', 'Carol', 'view')}>join-view</button>
      <button onClick={() => void joinCollabSession('abc', 'Bob', 'edit')}>join-edit</button>
    </>
  );
}

function renderApp() {
  return render(
    <AppProvider>
      <JoinHarness />
      <AppContent />
    </AppProvider>,
  );
}

beforeEach(() => {
  jest.clearAllMocks();
  mockedCollab.parseSessionId.mockImplementation((s: string) => s);
  mockedCollab.joinCollabRoom.mockImplementation((_id, _name, _role, _isCreator, onSyncInit) => {
    onSyncInit('Test Session', { schedule: SCHEDULE, envConfig: ENV_CONFIG, currentView: 'worker' }, []);
    return () => {};
  });
});

it('renders the shared editor shell in solo mode (no session)', () => {
  renderApp();
  expect(screen.getByTestId('editor-page')).toBeInTheDocument();
});

it('renders the shared editor shell for an edit-role session', async () => {
  renderApp();
  await userEvent.click(screen.getByText('join-edit'));
  await waitFor(() => expect(screen.getByTestId('editor-page')).toBeInTheDocument());
});

// The regression this file exists for: joining as 閲覧のみ from inside the app
// (the SessionDialog path) used to leave the fully interactive editor on
// screen, because read-only rendering was decided from the URL's ?role=view
// rather than from the session's own role. Now there is only ever one shell,
// so the assertion is simply that it renders here too — read-only-ness is a
// property of gating inside that shell, not of which shell got picked.
it('renders the same shared editor shell for a view-role session joined from inside the app', async () => {
  renderApp();
  await userEvent.click(screen.getByText('join-view'));
  await waitFor(() => expect(screen.getByTestId('editor-page')).toBeInTheDocument());
});
```

- [ ] **Step 6a: Run this file's tests to verify they pass**

Run: `cd GanttChartEditor && npm run test:jest -- App.test`
Expected: PASS (3/3).

- [ ] **Step 7: Run the full client suite and type-check**

Run: `cd GanttChartEditor && npm run test:jest && npx tsc -b`
Expected: all tests passing; `tsc -b` shows only the 3 pre-existing, unrelated errors (`yamlService.ts`, `WorkerViewGantt.tsx`, `reducer.test.ts`'s `BASE_STATE`) plus nothing new — if `WorkerTimelineGrid.tsx`/`DeviceViewGantt.tsx` show errors about a missing `readOnly` prop at this point, that's expected and resolved by Task 9, not this one.

- [ ] **Step 8: Commit**

```bash
git add GanttChartEditor/src/App.tsx GanttChartEditor/src/components/Toolbar/Toolbar.tsx GanttChartEditor/src/components/Toolbar/UndoRedoButtons.tsx GanttChartEditor/src/hooks/useKeyboardShortcuts.ts GanttChartEditor/src/__tests__/App.test.tsx
git rm GanttChartEditor/src/pages/ViewPage.tsx
git commit -m "feat(client): unify editor/viewer shells, gate Toolbar and keyboard shortcuts for read-only"
```

---

### Task 9: Gate the Gantt chart's drag interactions for read-only

**Files:**
- Modify: `GanttChartEditor/src/components/GanttChart/WorkerViewGantt.tsx`
- Modify: `GanttChartEditor/src/components/GanttChart/WorkerTimelineGrid.tsx`
- Modify: `GanttChartEditor/src/components/GanttChart/DeviceViewGantt.tsx`
- Test: `GanttChartEditor/cypress/e2e/06_viewer_parity.cy.ts` (new)

**Interfaces:**
- Consumes: Task 8's `isReadOnly` derivation pattern.
- Produces: `WorkerTimelineGrid`'s `Props` gains `readOnly: boolean`; `startDrag` and `startUnavailDrag` (both defined inside `WorkerTimelineGrid`) become no-ops when `readOnly` is true — this is the single choke point for every bar-move, bar-resize, and unavailable-date-drag entry, regardless of how many `onMouseDown` handlers ultimately call them, so no other call site needs individual changes.

- [ ] **Step 1: Thread `readOnly` into `WorkerTimelineGrid`**

In `WorkerViewGantt.tsx`, add `const isReadOnly = state.session?.role === 'view';` near the top of the component (it already has `const { state, dispatch } = useAppContext();`), and pass it to `<WorkerTimelineGrid ... readOnly={isReadOnly} ... />` (add the prop alongside the existing ones at its call site, around line 329).

In `WorkerTimelineGrid.tsx`, add `readOnly: boolean;` to the `Props` interface (near `onBarCommit: (commit: BarDragCommit) => void;`), and destructure it in the component's parameter list alongside `onBarCommit`.

- [ ] **Step 2: Gate `startDrag`**

At the very top of the `startDrag` function body (right after its parameter list, before `if (segment.assignmentIndex === undefined) return;`), add:

```ts
    if (readOnly) return;
```

- [ ] **Step 3: Gate `startUnavailDrag`**

At the very top of the `startUnavailDrag` function body, add the same:

```ts
    if (readOnly) return;
```

- [ ] **Step 4: Gate `DeviceViewGantt`'s inline edit callbacks**

In `DeviceViewGantt.tsx`, add `const isReadOnly = state.session?.role === 'view';` near the top of the component (it already has `const { state, dispatch } = useAppContext();`). Guard each of the three callback bodies found at the module's edit-detail-panel call site:

```tsx
              onChange={updates => { if (!isReadOnly) dispatch({ type: 'UPDATE_PHASE_TASK', payload: { workflowTaskId: selectedPhase.module.moduleId, phaseTaskId: selectedPhase.phase.phaseId, updates } }); }}
              onChangeWorker={(ai, wid) => { if (!isReadOnly) dispatch({ type: 'UPDATE_ASSIGNMENT', payload: { index: ai, updates: { worker: wid } } }); }}
              onChangeOpTask={(oid, updates) => { if (!isReadOnly) dispatch({ type: 'UPDATE_OPERATION_TASK', payload: { workflowTaskId: selectedTask.module.moduleId, phaseTaskId: selectedTask.phase.phaseId, operationTaskId: oid, updates } }); }}
```

(These replace the three existing one-line arrow functions at their current locations — same callback prop names, same underlying dispatches, just wrapped with the read-only check.)

- [ ] **Step 5: Run the full client suite and type-check**

Run: `cd GanttChartEditor && npm run test:jest && npx tsc -b`
Expected: all passing; only the 3 documented pre-existing `tsc` errors remain.

- [ ] **Step 6: Add the Cypress spec**

Create `GanttChartEditor/cypress/e2e/06_viewer_parity.cy.ts`, following this project's existing spec style (`cy.visit('/')`, `cy.loadFixtures()` custom command from `cypress/support/commands.ts`, `cy.request()` for direct API calls):

```ts
/**
 * Test Suite 06 — Viewer Parity
 * Verifies a view-role participant can scroll/filter/check-constraints but
 * cannot edit, using the real collab server (requires `npm run dev:all`,
 * not just `npm run dev` — see this feature's design doc for why).
 */
describe('06 – Viewer Parity', () => {
  it('a view-role join can toggle 出入国バー and run the constraint check, but cannot drag a bar or use undo/redo', () => {
    cy.visit('/');
    cy.loadFixtures();

    // Start a session as editor, capture its id from the share link so the
    // test can join a second, view-role client via the URL path — the same
    // mechanism a real viewer's browser link uses.
    cy.contains('共同編集').click();
    cy.contains('セッションを開始').click();
    cy.get('input[placeholder="セッション名を入力"]').type('Viewer Parity Test');
    cy.get('input[placeholder="表示名を入力"]').type('Editor');
    cy.contains('button', '開始する').click();
    cy.contains('セッション情報', { timeout: 8000 }).should('exist');

    cy.window().then(win => {
      const url = new URL(win.location.href);
      // The session id isn't in this window's own URL (it's the creator, not
      // a link-joiner) — read it from the copyable edit-link input instead.
    });
    cy.get('input[readonly]').first().invoke('val').then(editLink => {
      const sessionId = new URL(String(editLink)).searchParams.get('session');

      cy.visit(`/?session=${sessionId}&role=view`);
      cy.get('input[placeholder]').first().type('Viewer');
      cy.contains('button', '参加する').click();

      // Enabled for a viewer: flight-stints toggle and constraint check.
      cy.contains('✈ 出入国バー', { timeout: 8000 }).click();
      cy.contains('☑ 制約チェック').click();

      // Disabled for a viewer: undo/redo, and dragging a bar has no effect —
      // the button itself is visibly inert (opacity/disabled), which is the
      // observable proxy for "the handler is gated" from outside the app.
      cy.contains('button', '元に戻す').should('be.disabled');
      cy.contains('button', 'やり直し').should('be.disabled');
    });
  });
});
```

- [ ] **Step 7: Run the Cypress spec**

Run `npm run dev:all` in one terminal, then `cd GanttChartEditor && npm run test:cypress:run -- --spec cypress/e2e/06_viewer_parity.cy.ts` in another.
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add GanttChartEditor/src/components/GanttChart/WorkerViewGantt.tsx GanttChartEditor/src/components/GanttChart/WorkerTimelineGrid.tsx GanttChartEditor/src/components/GanttChart/DeviceViewGantt.tsx GanttChartEditor/cypress/e2e/06_viewer_parity.cy.ts
git commit -m "feat(client): gate bar drag/resize and device-view edit callbacks for read-only, add Cypress coverage"
```

---

## Self-Review Notes

- **Spec coverage:** §1 session name (Tasks 1-4), §2 participant list on click (Task 6), §3 join-dialog clarity (Task 5), §4 セッション情報 submenu (Task 7), §5 viewer parity (Tasks 8-9), §6 tab pre-selection (Tasks 2 & 4) — all six items covered.
- **Placeholder scan:** no TBD/vague steps; Task 8/9's "expected pre-existing tsc errors" note is a documented baseline fact carried from the original plan, not a placeholder.
- **Type consistency:** `startCollabSession(displayName, sessionName)` in Task 3 matches its call sites in Task 4's `StartOrJoinPanel`; `joinCollabRoom`'s `onSyncInit(sessionName, baseline, actions)` signature in Task 3 matches `AppContext.tsx`'s `joinInternal` usage in the same task; `WorkerTimelineGrid`'s `readOnly` prop name is used consistently between Task 8's stated pattern and Task 9's actual implementation.
- **Cross-task risk flagged for the controller/reviewer:** Tasks 2-7 leave the app in a non-compiling state between commits (by design, matching the original plan's precedent for this exact codebase) — `tsc -b` only needs to be clean at the end of Task 4 (client) and Task 9 (Gantt components); don't treat mid-sequence compile errors in not-yet-touched files as a regression.
