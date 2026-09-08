# Online Collaboration on Azure (ACA) — Web Client Plan (Phase 2)

> Follows `GanttChartEditor_OnlineCollabAzure_ACA_ImplementationPlan20260908.md` (Phase 1 backend, done).
> **Goal:** the desktop app + browser talk to the new ACA1 session API instead of LAN share links — a session list, create-by-YAML-upload, open, live co-edit, owner lock/unlock — all testable locally over the LAN via `npm run dev:mock`.

**Date:** 2026-09-08 · **Branch:** `online-collab-aca`

## What changes for the user

| Before (LAN share links) | After (ACA session list) |
|---|---|
| "Start Session" → copy a `http://192.168.x.x:5173/?session=…&role=…` link, send it | "セッション" dialog shows a **list** of online sessions with status; click 開く |
| "Join Session" → paste the link | pick from the list (or open `http://<host>:5173` and choose) |
| Roles: edit / view chosen at join | same, plus a session-wide **Lock** the owner toggles from inside |
| Session lives on the host PC | session lives in ACA1/ACA2 + storage; host can close the app |

## Config

- New `VITE_ACA1_URL`. Resolution in `collabService`:
  `import.meta.env.VITE_ACA1_URL ?? \`\${location.protocol}//\${location.hostname}:4000\``
  (LAN-friendly default — matches `dev:mock`'s ACA1 port).
- `openSession()` rewrites a `localhost`/`127.0.0.1` host in ACA1's returned `relayUrl`
  to `location.hostname`, so a LAN participant connects to the host, not their own machine.
  A real Azure FQDN is left untouched.

## File-by-file

### `src/services/collabService.ts` (rewrite the transport half)
- **Add:**
  - `listSessions(): Promise<SessionSummary[]>` — `GET {ACA1}/api/sessions`
  - `createSessionFromYaml(name, scheduleFile: File, envConfigFile: File): Promise<CreateResult>` — multipart `POST {ACA1}/api/sessions`
  - `createSessionFromState(name, baseline): Promise<CreateResult>` — JSON `POST {ACA1}/api/sessions`
  - `openSession(sessionId): Promise<{ relayUrl: string; status: SessionStatus }>` — `POST {ACA1}/api/sessions/:id/open` + localhost→hostname rewrite
  - `deleteSession(sessionId, ownerToken): Promise<void>` — `DELETE` with `x-owner-token`
  - `sendCollabLock()` / `sendCollabUnlock()` — `socket.emit('lock'|'unlock')`
  - `CreateResult = { sessionId: string; ownerToken: string }`
  - `SessionSummary` re-declared client-side (`id,name,status,createdAt,lastActivityAt,participantCount`)
- **Change `joinCollabRoom`:** new signature
  `joinCollabRoom(sessionId, name, role, isCreator, relayUrl, ownerToken, { onSyncInit, onAction, onPresence, onStatusChange, onSessionStatus })`.
  Connect the socket to `relayUrl` (not `hostname:3010`). Emit `join` with `{ sessionId, name, role, ownerToken }`. Handle inbound `session-status` → `onSessionStatus(status)`. `sync-init` payload now also carries `status` → forward via `onSessionStatus` on first sync.
- **Remove:** `fetchCollabLink`, `parseSessionOrigin`, `getSocketOrigin` (→ replaced by ACA1 URL + relayUrl), `/api/network-info` use.
- **Keep:** `fetchSessionName` (repoint to `GET {ACA1}/api/sessions/:id` → `.session.name`), `sendCollabAction`, `parseSessionId`.

### `src/types/appState.ts`
- `export type SessionStatus = 'open' | 'lock' | 'close';`
- `SessionState` gains `status: SessionStatus` and `ownerToken?: string`.
- New action `{ type: 'SET_SESSION_STATUS'; payload: SessionStatus }`.

### `src/context/reducer.ts`
- `SET_SESSION_STATUS` → `state.session ? { ...state, session: { ...state.session, status } } : state`.
- `SET_SESSION` default `status: 'open'` when a session is set without one.

### `src/context/AppContext.tsx`
- `startCollabSession(displayName, sessionName)` → `createSessionFromState(sessionName, {schedule,envConfig,currentView})` → store `ownerToken` → `openSession` → `joinInternal(..., relayUrl, ownerToken, isCreator=true)`.
- **Add** `createUploadSession(displayName, sessionName, scheduleFile, envConfigFile)` — same but `createSessionFromYaml`.
- `joinCollabSession(sessionId, name, role)` → `openSession(sessionId)` → `joinInternal(..., relayUrl, undefined, false)`.
- **Add** `lockSession()` / `unlockSession()` → `sendCollabLock/Unlock()`.
- `joinInternal` passes `relayUrl` + `ownerToken` through; wires `onSessionStatus` → `dispatch({type:'SET_SESSION_STATUS'})`.
- Outgoing dispatch gate: also block `SYNCABLE_ACTION_TYPES` + UNDO/REDO when `session.status === 'lock'` (mirror the existing disconnected-gate; server drops them anyway, this stops the optimistic local apply).
- Drop `fetchCollabLink` import.

### `src/lib/sessionReadOnly.ts` (new, tiny)
```ts
import type { AppState } from '../types/appState';
export const isSessionReadOnly = (s: AppState): boolean =>
  s.session != null && (s.session.role === 'view' || s.session.status === 'lock');
```
Replace `state.session?.role === 'view'` with `isSessionReadOnly(state)` in:
`Toolbar.tsx`, `UndoRedoButtons.tsx`, `SidePanel/SidePanel.tsx`, `hooks/useKeyboardShortcuts.ts`,
`GanttChart/WorkerViewGantt.tsx`, `GanttChart/DeviceViewGantt.tsx`, and the MenuBar viewer indicator.

### `src/components/Dialogs/SessionDialog.tsx` (rebuild)
- **No active session:** two tabs.
  - **一覧 (List):** `listSessions()` on open + a 更新 button + 5 s poll. Rows: name · status chip (`開催中`/`ロック`/`停止中`) · 参加者 N. 開く button → `joinCollabSession(id, displayName, role)` (role radio: 編集/閲覧). Needs a display-name field.
  - **新規作成 (Create):** name + display-name + two `<input type="file" accept=".yaml,.yml">` → `createUploadSession`. If `state.schedule` exists, also a "現在のスケジュールから作成" button → `startCollabSession`.
- **Active session:** name, status chip, participant list (kept). If `state.session.ownerToken` set → ロック / 解除 button (`lockSession`/`unlockSession`). 退出 button (kept). Remove the LinkRow / edit-link / view-link UI entirely.

### `src/App.tsx`
- `SessionJoinGate`: `?session=<id>&role=` still supported for LAN convenience — now calls `joinCollabSession(id, name, role)` (which does `openSession` internally). No other change.

### `src/components/Toolbar/MenuBar.tsx`
- Menu items: keep セッション情報 (opens dialog), rename しての "Start"/"Join" to just セッション一覧 opening the dialog on the list tab. Viewer indicator uses `isSessionReadOnly` and shows ロック中 vs 閲覧のみ.

### `src/config/uiText.ts`
- Add: `sessionListTab`, `sessionCreateTab`, `sessionStatusOpen/Lock/Close`, `sessionOpenBtn`, `sessionRefreshBtn`, `sessionFromCurrentBtn`, `sessionScheduleFileLabel`, `sessionEnvFileLabel`, `sessionLockBtn`, `sessionUnlockBtn`, `sessionLockedBanner`, `sessionNoSessions`.
- Remove/repurpose: `copyLinkBtn`, `sessionEditLinkLabel`, `sessionViewLinkLabel`, `sessionJoinLinkPlaceholder` (leave constants if other code references; delete dead ones).

### `src/config/appConfig.ts`
- Note: `SERVICE_BASE_URL` (`:3010`) stays for the local Node service (constraints/handoff). ACA1 URL is resolved in `collabService`, not here.

## Tests (jest + RTL)
- `__tests__/services/collabService.test.ts` — update `joinCollabRoom` calls to the new signature (relayUrl + options object); add `listSessions` / `openSession` (mock `fetch`, assert URL + localhost→hostname rewrite) and `createSessionFromYaml` (assert `FormData` parts).
- `__tests__/components/sessionDialog.test.tsx` — list renders rows from a mocked `listSessions`; 開く calls `joinCollabSession`; create tab posts files; active panel shows ロック button only with an `ownerToken`.
- `__tests__/components/sessionJoinGate.test.tsx` — `?session=` still joins via the new path (mock `openSession`).
- `__tests__/components/readOnlyGating.test.tsx` — add a `status: 'lock'` case alongside the existing `role: 'view'` cases (Toolbar/UndoRedo/SidePanel disabled).
- `__tests__/context/AppContext.test.tsx` — `lockSession()` emits; inbound `session-status` flips `state.session.status`; a syncable dispatch is blocked while locked.
- `__tests__/components/menuBar.test.tsx` — indicator text for lock vs view.

## Manual acceptance (over the LAN)
1. `npm run dev:mock` on the host. Note the host's LAN IP.
2. Host browser `http://<lan-ip>:5173` → セッション → 新規作成 → upload `Test_data/Schedule.yaml` + `EnvConfig.yaml` → 作成 → lands in the editor.
3. Second machine `http://<lan-ip>:5173` → セッション → the session is listed 開催中 → 開く as 編集 → both edit, bars move within ~1 s.
4. Host (owner) → セッション → ロック → second machine goes read-only with a banner; host edits are refused too. 解除 → editing resumes.
5. Everyone 退出 → session shows 停止中. Re-open → last state is back (replayed from `server/mock-blob/`).

## Commits
1. `collabService.ts` + its test — new ACA1 transport
2. `appState.ts` + `reducer.ts` — `SessionStatus`, `ownerToken`, `SET_SESSION_STATUS`
3. `AppContext.tsx` + test — create/open/lock wiring, locked-edit gate
4. `sessionReadOnly.ts` + 7 call sites + `readOnlyGating.test.tsx`
5. `SessionDialog.tsx` + test — list / create / lock UI
6. `App.tsx` + `MenuBar.tsx` + `uiText.ts` + remaining tests
