// Shared collaboration types used by both roles of the server:
//   - ROLE=aca2 (the live Socket.IO relay + persistence)
//   - ROLE=aca1 (the HTTP session API)
// Kept dependency-free and I/O-free so either entrypoint can import it.

/** Lifecycle status of a session, as recorded in `sessions/<id>/status.json`. */
export type SessionStatus = 'open' | 'lock' | 'close';

/** The immutable starting point a session is built from. Captured at creation. */
export interface SessionBaseline {
  schedule: unknown;
  envConfig: unknown;
  currentView: 'worker' | 'device';
}

/** One entry in a session's append-only, monotonically-sequenced action log. */
export interface LoggedAction {
  seq: number;
  type: string;
  payload: unknown;
}

/** `sessions/<id>/meta.json` — written once by ACA1 on create, never changed. */
export interface SessionMeta {
  id: string;
  name: string;
  createdAt: number;
  /** sha256 hex of the ownerToken handed back to the creator. */
  ownerTokenHash: string;
}

/** `sessions/<id>/status.json` — mutable; ACA1 seeds it, ACA2 owns it thereafter. */
export interface SessionStatusRecord {
  status: SessionStatus;
  /** Identifier of the ACA2 replica currently hosting the session, or null. */
  relayInstance: string | null;
  /** URL clients should open their socket to while the session is live, or null. */
  relayUrl: string | null;
  lastActivityAt: number;
}

/** One row in `GET /api/sessions`. */
export interface SessionSummary {
  id: string;
  name: string;
  status: SessionStatus;
  createdAt: number;
  lastActivityAt: number;
  /** Live participant count from ACA2; null when ACA2 is asleep or unreachable. */
  participantCount: number | null;
}
