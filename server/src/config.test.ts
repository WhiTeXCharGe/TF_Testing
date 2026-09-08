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
      ROLE: 'aca1', INTERNAL_KEY: 'k', ACA2_URL: 'x',
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
});
