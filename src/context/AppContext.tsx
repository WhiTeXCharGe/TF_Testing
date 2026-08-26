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
    // Loading a file is inherently local (it's not in SYNCABLE_ACTION_TYPES),
    // but nothing else stops it from firing mid-session — the menu, Ctrl+O,
    // and the incoming-transfer hook from the sibling SchedulerWeb app can
    // all dispatch LOAD_FILES with no session awareness. If it went through
    // while a session is active, this client would silently swap to a
    // different document while every other participant keeps sending
    // index/id-based edits against what is now the wrong document here —
    // real data corruption, not just a UI nuisance. Blocked regardless of
    // role: even a viewer silently swapping documents mid-session is broken.
    if (action.type === 'LOAD_FILES' && stateRef.current.session) {
      rawDispatch({ type: 'SET_ERROR', payload: UI.collabActiveLoadBlockedError });
      return;
    }
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
