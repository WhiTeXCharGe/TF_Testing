import { resolve } from 'node:path';
import { randomUUID } from 'node:crypto';

// Single place that reads and validates process.env. Every entrypoint
// (local | aca1 | aca2) calls loadConfig() once at boot and passes the
// result down — nothing else touches process.env.

export type Role = 'local' | 'aca1' | 'aca2';

export type StorageConfig =
  | { kind: 'memory' }
  | { kind: 'fs'; rootDir: string }
  | { kind: 'blob'; connectionString: string; container: string };

export interface AppConfig {
  role: Role;
  port: number;
  storage: StorageConfig;
  /** CORS allow-origin for the aca1/aca2 cloud builds; null → LAN allowlist (local). */
  webOrigin: string | null;
  /** Shared secret guarding ACA2's /internal/* routes. Empty only in local mode. */
  internalKey: string;
  /** aca1 → aca2 base URL, e.g. http://localhost:4010 */
  aca2Url: string;
  /** URL clients open their socket to; aca2 writes this into status.json. */
  publicRelayUrl: string;
  /** This replica's identifier, recorded as status.json.relayInstance. */
  instanceId: string;
  absoluteSessionMaxMs: number;
  idleSweepMs: number;
  idleSessionTimeoutMs: number;
  limits: {
    /** POST /api/sessions per IP per hour (ACA1). */
    createPerHourPerIp: number;
    /** Reject session creation past this many live records. */
    maxConcurrentSessions: number;
    /** Reject a socket `join` past this many participants in a session. */
    maxParticipants: number;
    /** multipart YAML upload cap, bytes. */
    maxUploadBytes: number;
    /** Socket.IO maxHttpBufferSize, bytes. */
    maxSocketMessageBytes: number;
  };
}

const ROLES: readonly Role[] = ['local', 'aca1', 'aca2'];

const DEFAULT_PORT: Record<Role, number> = { local: 3010, aca1: 4000, aca2: 4010 };

function parsePort(raw: string | undefined, fallback: number): number {
  if (raw === undefined || raw === '') return fallback;
  const n = Number(raw);
  if (!Number.isInteger(n) || n < 0 || n > 65535) {
    throw new Error(`PORT must be an integer 0-65535, got: ${raw}`);
  }
  return n;
}

function parseMs(raw: string | undefined, fallback: number): number {
  if (raw === undefined || raw === '') return fallback;
  const n = Number(raw);
  if (!Number.isFinite(n) || n <= 0) throw new Error(`expected a positive number of ms, got: ${raw}`);
  return n;
}

function parsePositiveInt(raw: string | undefined, fallback: number): number {
  if (raw === undefined || raw === '') return fallback;
  const n = Number(raw);
  if (!Number.isInteger(n) || n <= 0) throw new Error(`expected a positive integer, got: ${raw}`);
  return n;
}

export function loadConfig(env: NodeJS.ProcessEnv = process.env): AppConfig {
  const role = (env.ROLE ?? 'local') as Role;
  if (!ROLES.includes(role)) {
    throw new Error(`ROLE must be one of ${ROLES.join(' | ')}, got: ${env.ROLE}`);
  }

  const internalKey = env.INTERNAL_KEY ?? '';
  if (role !== 'local' && internalKey === '') {
    throw new Error(`INTERNAL_KEY is required when ROLE=${role}`);
  }

  // Local mode keeps session state in-process (as it did before persistence
  // existed); the cloud roles default to the folder-backed mock store.
  const storageKind = env.STORAGE ?? (role === 'local' ? 'memory' : 'fs');

  let storage: StorageConfig;
  if (storageKind === 'blob') {
    const connectionString = env.BLOB_CONNECTION_STRING ?? '';
    if (connectionString === '') throw new Error('BLOB_CONNECTION_STRING is required when STORAGE=blob');
    storage = { kind: 'blob', connectionString, container: env.BLOB_CONTAINER ?? 'sessions' };
  } else if (storageKind === 'fs') {
    storage = { kind: 'fs', rootDir: resolve(process.cwd(), env.MOCK_BLOB_DIR ?? '../mock-blob') };
  } else if (storageKind === 'memory') {
    storage = { kind: 'memory' };
  } else {
    throw new Error(`STORAGE must be memory | fs | blob, got: ${storageKind}`);
  }

  // A real deployment (STORAGE=blob) must pin CORS and use a strong shared
  // key — the mock (memory/fs) stays frictionless for local/LAN testing.
  if (storage.kind === 'blob' && !env.WEB_ORIGIN) {
    throw new Error('WEB_ORIGIN is required when STORAGE=blob (CORS must be pinned to the web origin)');
  }
  if (storage.kind === 'blob' && internalKey.length < 16) {
    throw new Error('INTERNAL_KEY must be at least 16 characters when STORAGE=blob');
  }

  const port = parsePort(env.PORT, DEFAULT_PORT[role]);

  return {
    role,
    port,
    storage,
    webOrigin: env.WEB_ORIGIN ?? null,
    internalKey,
    aca2Url: env.ACA2_URL ?? 'http://localhost:4010',
    publicRelayUrl: env.PUBLIC_RELAY_URL ?? `http://localhost:${port}`,
    instanceId: env.CONTAINER_APP_REPLICA_NAME ?? env.HOSTNAME ?? randomUUID(),
    absoluteSessionMaxMs: parseMs(env.ABSOLUTE_SESSION_MAX_MS, 8 * 60 * 60 * 1000),
    idleSweepMs: parseMs(env.IDLE_SWEEP_MS, 5 * 60 * 1000),
    idleSessionTimeoutMs: parseMs(env.IDLE_SESSION_TIMEOUT_MS, 30 * 60 * 1000),
    limits: {
      createPerHourPerIp: parsePositiveInt(env.CREATE_RATE_PER_HOUR, 10),
      maxConcurrentSessions: parsePositiveInt(env.MAX_SESSIONS, 100),
      maxParticipants: parsePositiveInt(env.MAX_PARTICIPANTS, 25),
      maxUploadBytes: parsePositiveInt(env.MAX_UPLOAD_MB, 5) * 1024 * 1024,
      maxSocketMessageBytes: parsePositiveInt(env.MAX_SOCKET_MB, 10) * 1024 * 1024,
    },
  };
}
