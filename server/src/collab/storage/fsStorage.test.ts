import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createFsStorage } from './fsStorage.js';
import type { StorageClient } from './storageClient.js';

let root: string;
let storage: StorageClient;

beforeEach(async () => {
  root = await mkdtemp(join(tmpdir(), 'gantt-fsstore-'));
  storage = createFsStorage(root);
});
afterEach(() => rm(root, { recursive: true, force: true }));

describe('fsStorage', () => {
  it('returns null for a missing key', async () => {
    expect(await storage.getJson('sessions/x/meta.json')).toBeNull();
  });

  it('round-trips JSON, creating parent directories', async () => {
    await storage.putJson('sessions/x/meta.json', { id: 'x', n: 1 });
    expect(await storage.getJson('sessions/x/meta.json')).toEqual({ id: 'x', n: 1 });
  });

  it('overwrites an existing key', async () => {
    await storage.putJson('k.json', { v: 1 });
    await storage.putJson('k.json', { v: 2 });
    expect(await storage.getJson('k.json')).toEqual({ v: 2 });
  });

  it('lists keys under a prefix, sorted, full paths', async () => {
    await storage.putJson('sessions/b/meta.json', {});
    await storage.putJson('sessions/a/meta.json', {});
    await storage.putJson('sessions/a/log.json', []);
    expect(await storage.listPrefix('sessions/')).toEqual([
      'sessions/a/log.json',
      'sessions/a/meta.json',
      'sessions/b/meta.json',
    ]);
  });

  it('lists by a partial (non-directory) prefix', async () => {
    await storage.putJson('sessions/abc/meta.json', {});
    await storage.putJson('sessions/abd/meta.json', {});
    expect(await storage.listPrefix('sessions/abc')).toEqual(['sessions/abc/meta.json']);
  });

  it('returns [] listing a prefix that does not exist', async () => {
    expect(await storage.listPrefix('nope/')).toEqual([]);
  });

  it('delete removes one key and is a no-op when absent', async () => {
    await storage.putJson('k.json', { v: 1 });
    await storage.delete('k.json');
    await storage.delete('k.json');
    expect(await storage.getJson('k.json')).toBeNull();
  });

  it('deletePrefix removes the whole subtree', async () => {
    await storage.putJson('sessions/a/meta.json', {});
    await storage.putJson('sessions/a/log.json', []);
    await storage.deletePrefix('sessions/a');
    expect(await storage.listPrefix('sessions/a')).toEqual([]);
  });

  it('rejects keys that escape the root', async () => {
    await expect(storage.putJson('../evil.json', {})).rejects.toThrow();
    await expect(storage.getJson('../../etc/passwd')).rejects.toThrow();
  });
});
