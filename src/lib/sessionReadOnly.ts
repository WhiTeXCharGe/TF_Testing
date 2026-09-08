import type { AppState } from '../types/appState';

// The single "this participant cannot edit" decision. True when in a session
// AND either joined as a viewer, or the session is locked (owner-toggled,
// read-only for everyone). Solo editing (no session) is never read-only.
export function isSessionReadOnly(state: AppState): boolean {
  const s = state.session;
  return s != null && (s.role === 'view' || s.status === 'lock');
}
