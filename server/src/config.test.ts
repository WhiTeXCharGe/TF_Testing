import { describe, it, expect } from 'vitest';
import { loadConfig } from './config.js';

describe('loadConfig', () => {
  it('defaults to local + in-memory storage + port 3010', () => {
    const c = loadConfig({});
    expect(c.role).toBe('local');
    expect(c.port).toBe(3010);
    expect(c.storage).toEqual({ kind: 'memory' });
  });

  it('cloud roles default to fs storage under mock-blob', () => {
    const c = loadConfig({ ROLE: 'aca2', INTERNAL_KEY: 'k' });
    expect(c.storage).toEqual({ kind: 'fs', rootDir: expect.stringContaining('mock-blob') });
  });

  it('honours an explicit STORAGE=fs in local mode', () => {
    const c = loadConfig({ STORAGE: 'fs', MOCK_BLOB_DIR: '/tmp/x' });
    expect(c.storage).toEqual({ kind: 'fs', rootDir: expect.stringContaining('x') });
  });

  it('reads role and port', () => {
    const c = loadConfig({ ROLE: 'aca1', PORT: '4000', INTERNAL_KEY: 'k', ACA2_URL: 'http://localhost:4010' });
    expect(c.role).toBe('aca1');
    expect(c.port).toBe(4000);
    expect(c.aca2Url).toBe('http://localhost:4010');
  });

  it('allows PORT=0 for ephemeral binding', () => {
    const c = loadConfig({ ROLE: 'aca2', PORT: '0', INTERNAL_KEY: 'k' });
    expect(c.port).toBe(0);
  });

  it('throws when aca2 has no INTERNAL_KEY', () => {
    expect(() => loadConfig({ ROLE: 'aca2' })).toThrow(/INTERNAL_KEY/);
  });

  it('throws when aca1 has no INTERNAL_KEY', () => {
    expect(() => loadConfig({ ROLE: 'aca1', ACA2_URL: 'x' })).toThrow(/INTERNAL_KEY/);
  });

  it('throws on unknown role', () => {
    expect(() => loadConfig({ ROLE: 'nope' })).toThrow(/ROLE/);
  });

  it('parses blob storage config', () => {
    const c = loadConfig({
      ROLE: 'aca1', INTERNAL_KEY: 'a-strong-enough-key-1234', ACA2_URL: 'x', WEB_ORIGIN: 'https://x',
      STORAGE: 'blob', BLOB_CONNECTION_STRING: 'cs', BLOB_CONTAINER: 'sessions',
    });
    expect(c.storage).toEqual({ kind: 'blob', connectionString: 'cs', container: 'sessions' });
  });

  it('throws on STORAGE=blob without a connection string', () => {
    expect(() => loadConfig({ ROLE: 'aca1', INTERNAL_KEY: 'k', ACA2_URL: 'x', STORAGE: 'blob' })).toThrow(/BLOB_CONNECTION_STRING/);
  });

  it('local mode does not require an internal key', () => {
    expect(() => loadConfig({ ROLE: 'local' })).not.toThrow();
    expect(loadConfig({}).role).toBe('local');
  });

  it('exposes default limits', () => {
    expect(loadConfig({}).limits).toEqual({
      createPerHourPerIp: 10,
      maxConcurrentSessions: 100,
      maxParticipants: 25,
      maxUploadBytes: 5 * 1024 * 1024,
      maxSocketMessageBytes: 10 * 1024 * 1024,
    });
  });

  it('reads limit overrides from env', () => {
    const c = loadConfig({ MAX_PARTICIPANTS: '4', MAX_SESSIONS: '2', CREATE_RATE_PER_HOUR: '3', MAX_UPLOAD_MB: '1', MAX_SOCKET_MB: '2' });
    expect(c.limits).toMatchObject({ maxParticipants: 4, maxConcurrentSessions: 2, createPerHourPerIp: 3, maxUploadBytes: 1048576, maxSocketMessageBytes: 2097152 });
  });

  it('a fs/memory aca2 does NOT require WEB_ORIGIN or a long key (mock stays frictionless)', () => {
    expect(() => loadConfig({ ROLE: 'aca2', INTERNAL_KEY: 'short' })).not.toThrow();
  });

  it('STORAGE=blob requires WEB_ORIGIN', () => {
    expect(() => loadConfig({
      ROLE: 'aca2', INTERNAL_KEY: 'a-strong-enough-key', STORAGE: 'blob', BLOB_CONNECTION_STRING: 'cs',
    })).toThrow(/WEB_ORIGIN/);
  });

  it('STORAGE=blob requires an INTERNAL_KEY of at least 16 chars', () => {
    expect(() => loadConfig({
      ROLE: 'aca2', INTERNAL_KEY: 'tooshort', STORAGE: 'blob', BLOB_CONNECTION_STRING: 'cs', WEB_ORIGIN: 'https://x',
    })).toThrow(/INTERNAL_KEY/);
  });

  it('STORAGE=blob passes with a pinned origin and a strong key', () => {
    expect(() => loadConfig({
      ROLE: 'aca2', INTERNAL_KEY: 'a-strong-enough-key-1234', STORAGE: 'blob', BLOB_CONNECTION_STRING: 'cs', WEB_ORIGIN: 'https://collab.example.com',
    })).not.toThrow();
  });
});
