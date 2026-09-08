import { describe, it, expect, beforeAll, afterAll, beforeEach } from 'vitest';
import { spawn, type ChildProcess } from 'node:child_process';
import { createServer } from 'node:http';
import type { AddressInfo } from 'node:net';
import { createBlobStorage } from './blobStorage.js';
import type { StorageClient } from './storageClient.js';

// Runs the StorageClient contract against a real Azure Blob API, served by
// Azurite (npm, in-memory — no Docker). If Azurite can't start, the suite
// fails loudly rather than silently skipping.

const AZURITE_KEY =
  'Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==';

let azurite: ChildProcess;
let connectionString: string;

async function freePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const s = createServer();
    s.listen(0, () => {
      const { port } = s.address() as AddressInfo;
      s.close((e) => (e ? reject(e) : resolve(port)));
    });
  });
}

async function waitForPort(port: number, timeoutMs = 15_000): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      // Azurite answers anonymous requests with 400/403 — any HTTP response means it's up.
      await fetch(`http://127.0.0.1:${port}/devstoreaccount1`);
      return;
    } catch {
      await new Promise((r) => setTimeout(r, 200));
    }
  }
  throw new Error(`Azurite did not come up on :${port} within ${timeoutMs}ms`);
}

beforeAll(async () => {
  const port = await freePort();
  const bin = process.platform === 'win32' ? 'azurite-blob.cmd' : 'azurite-blob';
  azurite = spawn(
    bin,
    ['--inMemoryPersistence', '--silent', '--blobHost', '127.0.0.1', '--blobPort', String(port)],
    { cwd: process.cwd(), stdio: 'ignore', shell: process.platform === 'win32' },
  );
  await waitForPort(port);
  connectionString =
    `DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;AccountKey=${AZURITE_KEY};` +
    `BlobEndpoint=http://127.0.0.1:${port}/devstoreaccount1;`;
}, 30_000);

afterAll(() => {
  azurite?.kill();
});

let storage: StorageClient;
let container: string;

beforeEach(() => {
  // Fresh container per test so cases don't see each other's blobs.
  container = `t${Date.now().toString(36)}${Math.random().toString(36).slice(2, 7)}`;
  storage = createBlobStorage(connectionString, container);
});

describe('blobStorage (contract, via Azurite)', () => {
  it('returns null for a missing key', async () => {
    expect(await storage.getJson('sessions/x/meta.json')).toBeNull();
  });

  it('round-trips JSON', async () => {
    await storage.putJson('sessions/x/meta.json', { id: 'x', n: 1 });
    expect(await storage.getJson('sessions/x/meta.json')).toEqual({ id: 'x', n: 1 });
  });

  it('overwrites an existing key', async () => {
    await storage.putJson('k.json', { v: 1 });
    await storage.putJson('k.json', { v: 2 });
    expect(await storage.getJson('k.json')).toEqual({ v: 2 });
  });

  it('lists keys under a prefix, sorted', async () => {
    await storage.putJson('sessions/b/meta.json', {});
    await storage.putJson('sessions/a/meta.json', {});
    await storage.putJson('sessions/a/log.json', []);
    expect(await storage.listPrefix('sessions/')).toEqual([
      'sessions/a/log.json',
      'sessions/a/meta.json',
      'sessions/b/meta.json',
    ]);
    expect(await storage.listPrefix('sessions/a')).toEqual(['sessions/a/log.json', 'sessions/a/meta.json']);
  });

  it('delete removes one key and is a no-op when absent', async () => {
    await storage.putJson('k.json', { v: 1 });
    await storage.delete('k.json');
    await storage.delete('k.json');
    expect(await storage.getJson('k.json')).toBeNull();
  });

  it('deletePrefix removes the subtree', async () => {
    await storage.putJson('sessions/a/meta.json', {});
    await storage.putJson('sessions/a/log.json', []);
    await storage.deletePrefix('sessions/a');
    expect(await storage.listPrefix('sessions/a')).toEqual([]);
  });
});
