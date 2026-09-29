import { createContext, useContext, useReducer, useCallback, useEffect, useRef, Dispatch, ReactNode } from 'react';
import {
  AppState, ActionType, SessionRole,
  DEFAULT_WORKER_VIEW_FILTER, DEFAULT_MODULE_VIEW_FILTER, DEFAULT_WORKER_COLUMN_FILTER,
} from '../types/appState';
import { reducer } from './reducer';
import { captureUndoEntry, hasConflict, buildRevertAction } from './undoEntries';
import {
  createSessionFromState, createSessionFromYaml, joinCollabRoom, sendCollabAction,
  sendCollabLock, sendCollabUnlock, sendCollabCheckpoint, sendCollabSessionUpdate,
  parseYamlBaseline, openSession, overwriteSessionState, parseSessionId,
  probeAzureReachability,
} from '../services/collabService';
import type { SessionBaseline } from '../types/appState';
import { UI } from '../config/uiText';
import { generateId } from '../utils/id';

const initialState: AppState = {
  envConfig: null,
  schedule: null,
  currentView: 'worker',
  violations: [],
  myPendingUndo: [],
  myPendingRedo: [],
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
  sessionDialog: null,
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
  // Revert vehicles for undo/redo (see undoEntries.ts's buildRevertAction).
  // These two key on real shared ids (workflowTask.id / worker.id), so
  // forwarding them to other participants is safe.
  'REMOVE_WORKFLOW_TASKS_BY_ID', 'RESTORE_WORKER_UNAVAILABLE_DATES',
  // NOTE: RESTORE_ASSIGNMENT_FIELDS and REVERT_MERGE are deliberately NOT
  // included here — they key on Assignment._id, which each client mints
  // independently for baseline-loaded assignments and isn't guaranteed to
  // match across participants. They're blocked outright while in a session
  // (see the dispatch wrapper's UNDO handling below) rather than being
  // silently forwarded and diverging everyone else's copy.
]);

interface ContextType {
  state: AppState;
  dispatch: Dispatch<ActionType>;
  startCollabSession: (displayName: string, sessionName: string) => Promise<{ sessionId: string }>;
  createUploadSession: (displayName: string, sessionName: string, scheduleFile: File, envConfigFile: File) => Promise<{ sessionId: string }>;
  overwriteAndJoinSession: (displayName: string, existingSessionId: string, sessionName: string, baseline: SessionBaseline) => Promise<{ sessionId: string }>;
  joinCollabSession: (sessionId: string, name: string, role: SessionRole) => Promise<void>;
  lockSession: () => void;
  unlockSession: () => void;
  updateSessionFromYaml: (scheduleFile: File, envConfigFile: File) => Promise<void>;
  leaveCollabSession: () => void;
}

// Periodic backup: an active session's durable snapshot (current.json —
// there's no action-by-action log, see server/src/collab/persistence.ts)
// otherwise only refreshes on an explicit checkpoint moment (leave, a
// server-requested checkpoint, a session-data update). A long-running
// session with nobody triggering one of those could sit far behind what's
// actually on everyone's screen, so an edit-role participant also checkpoints
// on its own every N actions or after M minutes, whichever comes first.
const BACKUP_ACTION_THRESHOLD = 50;
const BACKUP_INTERVAL_MS = 60 * 60 * 1000; // 1 hour

const AppContext = createContext<ContextType | undefined>(undefined);

export function AppProvider({ children }: { children: ReactNode }) {
  const [state, rawDispatch] = useReducer(reducer, initialState);
  const stateRef = useRef(state);
  stateRef.current = state;
  const disconnectRef = useRef<(() => void) | null>(null);
  const actionsSinceCheckpointRef = useRef(0);
  const lastCheckpointAtRef = useRef(Date.now());

  // This session's current schedule/envConfig, ready to hand to
  // sendCollabCheckpoint — null when there's nothing loaded to send. Doesn't
  // check role/session state itself; callers decide when a checkpoint makes
  // sense (e.g. only for an edit-role participant).
  const currentCheckpointBaseline = useCallback((): SessionBaseline | null => {
    const { schedule, envConfig, currentView } = stateRef.current;
    return schedule && envConfig ? { schedule, envConfig, currentView } : null;
  }, []);

  const markCheckpointed = useCallback(() => {
    actionsSinceCheckpointRef.current = 0;
    lastCheckpointAtRef.current = Date.now();
  }, []);

  // Resolve once, in the background, whether this app can actually reach the
  // build-time Azure URL (if any was baked in) — before the user opens a
  // session dialog, so 参加/作成 already point the right way by the time they
  // click. A packaged installer with no network/VPN falls back to its own
  // bundled local server instead of hanging on an unreachable host.
  useEffect(() => { void probeAzureReachability(); }, []);

  // Outgoing: apply locally as normal, and if we're an editor in an active
  // session, also forward data-mutating actions to the server. UNDO/REDO are
  // own-action, conflict-aware: each records a per-entry UndoEntry (see
  // undoEntries.ts) describing exactly what changed, so undoing/redoing it
  // later can check whether anyone else has touched the same object since
  // and forward the specific reverted action instead of a whole-schedule
  // snapshot.
  const dispatch: Dispatch<ActionType> = useCallback((action: ActionType) => {
    // Loading a file mid-session would silently swap this client's document
    // while everyone else keeps editing the old one — real corruption.
    if (action.type === 'LOAD_FILES' && stateRef.current.session) {
      rawDispatch({ type: 'SET_ERROR', payload: UI.collabActiveLoadBlockedError });
      return;
    }
    const session = stateRef.current.session;
    const isSyncingEdit = session?.role === 'edit';
    const needsLiveConnection =
      action.type === 'UNDO' || action.type === 'REDO' || SYNCABLE_ACTION_TYPES.has(action.type);
    // A locked session is read-only for everyone. The relay drops these
    // anyway; blocking here stops the optimistic local apply that would
    // otherwise diverge this client until the next sync-init.
    if (isSyncingEdit && needsLiveConnection && session?.status === 'lock') {
      rawDispatch({ type: 'SET_ERROR', payload: UI.collabLockedEditBlockedError });
      return;
    }
    // socket.io buffers emits while disconnected, so a bar dragged after the
    // connection dropped would move locally and look synced yet never leave
    // this machine. Block such actions until the connection is back.
    if (isSyncingEdit && needsLiveConnection && session?.connectionStatus !== 'connected') {
      rawDispatch({ type: 'SET_ERROR', payload: UI.collabDisconnectedEditBlockedError });
      return;
    }

    if (action.type === 'UNDO' || action.type === 'REDO') {
      const direction = action.type === 'UNDO' ? 'undo' : 'redo';
      const pending = direction === 'undo' ? stateRef.current.myPendingUndo : stateRef.current.myPendingRedo;
      const entry = pending[pending.length - 1];
      if (!entry) return;

      // Solo mode: nobody else could have touched anything, so revert
      // unconditionally — no conflict check, no error message, no server
      // round-trip. Same behavior solo mode already has today, just
      // implemented via captured entries instead of a snapshot stack.
      if (!stateRef.current.session) {
        const revertAction = buildRevertAction(entry, stateRef.current, direction);
        if (!revertAction) {
          rawDispatch({ type: 'SET_ERROR', payload: direction === 'undo' ? UI.undoUnsupportedError : UI.redoUnsupportedError });
          rawDispatch({ type: direction === 'undo' ? 'CONSUME_UNDO_ENTRY' : 'CONSUME_REDO_ENTRY' });
          return;
        }
        rawDispatch(revertAction);
        rawDispatch({ type: direction === 'undo' ? 'CONSUME_UNDO_ENTRY' : 'CONSUME_REDO_ENTRY' });
        return;
      }

      // Undoing a bulk-flexibility or merge-data edit reverts via an action
      // keyed on Assignment._id — an id each client mints independently for
      // baseline-loaded assignments, so it isn't guaranteed to match across
      // participants. Reverting locally and forwarding it would silently
      // diverge everyone else's copy with no error. Block it outright while
      // in a session (solo mode above is unaffected) until _id is made
      // session-consistent.
      if (direction === 'undo' && (entry.kind === 'bulkFlex' || entry.kind === 'mergeData')) {
        rawDispatch({ type: 'SET_ERROR', payload: UI.bulkUndoUnsupportedInSessionError });
        return;
      }

      if (hasConflict(entry, stateRef.current, direction)) {
        rawDispatch({ type: 'SET_ERROR', payload: direction === 'undo' ? UI.undoBlockedError : UI.redoBlockedError });
        return;
      }
      const revertAction = buildRevertAction(entry, stateRef.current, direction);
      if (!revertAction) {
        rawDispatch({ type: 'SET_ERROR', payload: direction === 'undo' ? UI.undoUnsupportedError : UI.redoUnsupportedError });
        rawDispatch({ type: direction === 'undo' ? 'CONSUME_UNDO_ENTRY' : 'CONSUME_REDO_ENTRY' });
        return;
      }
      rawDispatch(revertAction);
      rawDispatch({ type: direction === 'undo' ? 'CONSUME_UNDO_ENTRY' : 'CONSUME_REDO_ENTRY' });
      if (isSyncingEdit) {
        sendCollabAction(revertAction.type, (revertAction as { payload?: unknown }).payload);
      }
      return;
    }

    // ADD_ASSIGNMENT and MERGE_DATA can create brand-new assignments —
    // assign their stable _id here, once, before either capture or the real
    // dispatch see them, so both agree on the same id.
    let effectiveAction = action;
    if (action.type === 'ADD_ASSIGNMENT' && !action.payload._id) {
      effectiveAction = { ...action, payload: { ...action.payload, _id: generateId() } };
    } else if (action.type === 'MERGE_DATA' && action.payload.schedule) {
      effectiveAction = {
        ...action,
        payload: {
          ...action.payload,
          schedule: {
            ...action.payload.schedule,
            assignmentList: action.payload.schedule.assignmentList.map(a => (a._id ? a : { ...a, _id: generateId() })),
          },
        },
      };
    }

    // Undo/redo tracking: solo mode captures unconditionally (nothing else
    // could have touched anything); an active session only captures for an
    // edit-role participant. A view-role participant never reaches here with
    // a mutating action type in the first place — every control that could
    // dispatch one is already gated by isReadOnly — so `!isSyncingEdit` here
    // can only mean solo mode, matching the `!stateRef.current.session` check.
    const shouldCapture = SYNCABLE_ACTION_TYPES.has(effectiveAction.type) && (!stateRef.current.session || isSyncingEdit);
    const capturedEntry = shouldCapture
      ? captureUndoEntry(effectiveAction.type, (effectiveAction as { payload?: unknown }).payload, stateRef.current)
      : null;

    rawDispatch(effectiveAction);

    if (capturedEntry) {
      rawDispatch({ type: 'RECORD_UNDO_ENTRY', payload: capturedEntry });
    }
    if (isSyncingEdit && SYNCABLE_ACTION_TYPES.has(effectiveAction.type)) {
      sendCollabAction(effectiveAction.type, (effectiveAction as { payload?: unknown }).payload);
      actionsSinceCheckpointRef.current += 1;
      const dueByCount = actionsSinceCheckpointRef.current >= BACKUP_ACTION_THRESHOLD;
      const dueByTime = Date.now() - lastCheckpointAtRef.current >= BACKUP_INTERVAL_MS;
      if (dueByCount || dueByTime) {
        const baseline = currentCheckpointBaseline();
        if (baseline) {
          markCheckpointed();
          void sendCollabCheckpoint(baseline);
        }
      }
    }
  }, [currentCheckpointBaseline, markCheckpointed]);

  // Inbound guard. The relay forwards whatever `type` string an edit-role
  // participant emits without validating it, so an inbound action is
  // untrusted input. SYNCABLE_ACTION_TYPES is the single source of truth for
  // "a real cross-participant edit" in both directions; anything else is ignored.
  const applyRemoteAction = useCallback((action: { type: string; payload: unknown }) => {
    if (!SYNCABLE_ACTION_TYPES.has(action.type as ActionType['type'])) return;
    rawDispatch({ type: action.type, payload: action.payload } as ActionType);
  }, []);

  const joinInternal = useCallback((
    sessionId: string, name: string, role: SessionRole, isCreator: boolean,
    relayUrl: string, ownerToken: string | undefined,
  ) => {
    disconnectRef.current?.();
    disconnectRef.current = joinCollabRoom(sessionId, name, role, isCreator, relayUrl, ownerToken, {
      onSyncInit: (sessionName, baseline, actions) => {
        // Incoming: applied via the raw dispatch, never the wrapped one —
        // otherwise a remote action would be immediately re-forwarded and echo forever.
        rawDispatch({ type: 'SET_SESSION_NAME', payload: sessionName });
        rawDispatch({ type: 'SET_SESSION_BASELINE', payload: baseline });
        for (const a of actions) applyRemoteAction(a);
        // Fresh baseline just landed (join, reconnect, or someone else's
        // update/overwrite) — the periodic-backup clock starts over from here.
        markCheckpointed();
      },
      onAction: applyRemoteAction,
      onPresence: (participants) => rawDispatch({ type: 'SET_SESSION_PARTICIPANTS', payload: participants }),
      onStatusChange: (status) => rawDispatch({ type: 'SET_SESSION_CONNECTION_STATUS', payload: status }),
      onSessionStatus: (status) => rawDispatch({ type: 'SET_SESSION_STATUS', payload: status }),
      onCheckpointRequest: () => {
        const { session } = stateRef.current;
        const baseline = currentCheckpointBaseline();
        if (session?.role !== 'edit' || !baseline) return;
        markCheckpointed();
        void sendCollabCheckpoint(baseline);
      },
    });
  }, [applyRemoteAction, currentCheckpointBaseline, markCheckpointed]);

  const startCollabSession = useCallback(async (displayName: string, sessionName: string) => {
    const { schedule, envConfig, currentView } = stateRef.current;
    if (!schedule || !envConfig) throw new Error(UI.collabNoScheduleError);
    const { sessionId, ownerToken } = await createSessionFromState(sessionName, { schedule, envConfig, currentView });
    const { relayUrl, status } = await openSession(sessionId);
    rawDispatch({ type: 'SET_SESSION', payload: { id: sessionId, name: sessionName, role: 'edit', connectionStatus: 'connecting', participants: [], status, ownerToken } });
    joinInternal(sessionId, displayName, 'edit', true, relayUrl, ownerToken);
    return { sessionId };
  }, [joinInternal]);

  const createUploadSession = useCallback(async (
    displayName: string, sessionName: string, scheduleFile: File, envConfigFile: File,
  ) => {
    const { sessionId, ownerToken } = await createSessionFromYaml(sessionName, scheduleFile, envConfigFile);
    const { relayUrl, status } = await openSession(sessionId);
    rawDispatch({ type: 'SET_SESSION', payload: { id: sessionId, name: sessionName, role: 'edit', connectionStatus: 'connecting', participants: [], status, ownerToken } });
    joinInternal(sessionId, displayName, 'edit', false, relayUrl, ownerToken);
    return { sessionId };
  }, [joinInternal]);

  // Overwriting a duplicate-named session (create dialog, after the user
  // confirms the warning) reuses the existing session's id rather than
  // minting a new one — same tail as startCollabSession/createUploadSession,
  // just skipping the create step.
  const overwriteAndJoinSession = useCallback(async (
    displayName: string, existingSessionId: string, sessionName: string, baseline: SessionBaseline,
  ) => {
    await overwriteSessionState(existingSessionId, baseline);
    const { relayUrl, status } = await openSession(existingSessionId);
    rawDispatch({ type: 'SET_SESSION', payload: { id: existingSessionId, name: sessionName, role: 'edit', connectionStatus: 'connecting', participants: [], status } });
    joinInternal(existingSessionId, displayName, 'edit', false, relayUrl, undefined);
    return { sessionId: existingSessionId };
  }, [joinInternal]);

  const joinCollabSession = useCallback(async (idOrLink: string, name: string, role: SessionRole) => {
    const sessionId = parseSessionId(idOrLink);
    const { relayUrl, status } = await openSession(sessionId);
    rawDispatch({ type: 'SET_SESSION', payload: { id: sessionId, name: '', role, connectionStatus: 'connecting', participants: [], status } });
    joinInternal(sessionId, name, role, false, relayUrl, undefined);
  }, [joinInternal]);

  const lockSession = useCallback(() => sendCollabLock(), []);
  const unlockSession = useCallback(() => sendCollabUnlock(), []);

  // Explicit "replace this session's whole data" push (only takes effect
  // while locked — the server enforces that; import-only, see
  // SessionUpdateDialog's own comment for why there's no "use the current
  // Gantt" option here). The server broadcasts a fresh sync-init back to
  // everyone including us, which the existing onSyncInit handler in
  // joinInternal already applies — no local state update here.
  const updateSessionFromYaml = useCallback(async (scheduleFile: File, envConfigFile: File) => {
    const baseline = await parseYamlBaseline(scheduleFile, envConfigFile);
    sendCollabSessionUpdate(baseline);
  }, []);

  const leaveCollabSession = useCallback(async () => {
    // Best-effort: if I'm the last one here (an editor, with data to give),
    // hand the server a final snapshot before disconnecting — the server
    // only ever persists one "current state" file per session, not a growing
    // action log, so this is what makes this session's edits durable. If it
    // can't fire (abrupt disconnect, or a view-only participant is the one
    // left), the persisted state just stays as of the last checkpoint. Waits
    // for the server's ack (bounded by sendCollabCheckpoint's own timeout)
    // before disconnecting, so the write has actually landed rather than
    // racing the socket teardown right behind it.
    const { session } = stateRef.current;
    const baseline = currentCheckpointBaseline();
    if (session?.role === 'edit' && session.participants.length <= 1 && baseline) {
      await sendCollabCheckpoint(baseline);
    }
    disconnectRef.current?.();
    disconnectRef.current = null;
    rawDispatch({ type: 'SET_SESSION', payload: null });
  }, [currentCheckpointBaseline]);

  useEffect(() => () => disconnectRef.current?.(), []);

  // Electron only: main.cts intercepts the window's X button / Alt+F4 and
  // waits for 'app:ready-to-close' instead of closing immediately, so a
  // click on the close button can't silently tear down an in-progress
  // session's socket before this checkpoint has a chance to reach the
  // server (main.cts still force-closes after a few seconds if this never
  // fires — e.g. a wedged renderer — so this can't hang the app shut).
  useEffect(() => {
    const api = window.electronAPI;
    if (!api) return;
    api.onBeforeClose(() => {
      const { session } = stateRef.current;
      const baseline = currentCheckpointBaseline();
      if (session?.role === 'edit' && baseline) {
        void sendCollabCheckpoint(baseline).finally(() => api.notifyReadyToClose());
      } else {
        api.notifyReadyToClose();
      }
    });
  }, [currentCheckpointBaseline]);

  return (
    <AppContext.Provider value={{
      state, dispatch, startCollabSession, createUploadSession, overwriteAndJoinSession, joinCollabSession,
      lockSession, unlockSession, updateSessionFromYaml, leaveCollabSession,
    }}>
      {children}
    </AppContext.Provider>
  );
}

export function useAppContext(): ContextType {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error('useAppContext must be used within AppProvider');
  return ctx;
}
