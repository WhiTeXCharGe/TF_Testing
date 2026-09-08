import { mkdir, readFile, writeFile, rename, rm, readdir, unlink } from 'node:fs/promises';
import { dirname, join, resolve, sep } from 'node:path';
import { randomUUID } from 'node:crypto';
import type { StorageClient } from './storageClient.js';

// Folder-backed StorageClient. Keys map 1:1 to paths under `rootDir`. Small
// internal scale (dozens of sessions, a handful of files each), so listPrefix
// walks the whole tree and filters — simple and always correct. The Blob
// implementation uses native prefix listing instead.
export function createFsStorage(rootDir: string): StorageClient {
  const root = resolve(rootDir);

  const toPath = (key: string): string => {
    const p = resolve(root, key);
    if (p !== root && !p.startsWith(root + sep)) {
      throw new Error(`storage key escapes root: ${key}`);
    }
    return p;
  };

  // Root-relative, '/'-joined keys for every file under `dir`. Missing dir → [].
  const walkKeys = async (dir: string): Promise<string[]> => {
    let entries: import('node:fs').Dirent[];
    try {
      entries = await readdir(dir, { withFileTypes: true });
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code === 'ENOENT') return [];
      throw err;
    }
    const out: string[] = [];
    for (const e of entries) {
      const full = join(dir, e.name);
      if (e.isDirectory()) {
        out.push(...(await walkKeys(full)));
      } else {
        out.push(full.slice(root.length + 1).split(sep).join('/'));
      }
    }
    return out;
  };

  const getJson = async <T>(key: string): Promise<T | null> => {
    try {
      return JSON.parse(await readFile(toPath(key), 'utf-8')) as T;
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code === 'ENOENT') return null;
      throw err;
    }
  };

  const putJson = async (key: string, value: unknown): Promise<void> => {
    const path = toPath(key);
    await mkdir(dirname(path), { recursive: true });
    const tmp = `${path}.${randomUUID()}.tmp`;
    await writeFile(tmp, JSON.stringify(value, null, 2), 'utf-8');
    await rename(tmp, path);
  };

  const listPrefix = async (prefix: string): Promise<string[]> => {
    const all = await walkKeys(root);
    return all.filter((k) => k.startsWith(prefix)).sort();
  };

  const deleteKey = async (key: string): Promise<void> => {
    try {
      await unlink(toPath(key));
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code !== 'ENOENT') throw err;
    }
  };

  const deletePrefix = async (prefix: string): Promise<void> => {
    // Directory-shaped prefixes are the common case (`sessions/<id>`); rm -rf
    // the directory, and also drop any sibling files that share the prefix.
    await rm(toPath(prefix), { recursive: true, force: true });
    const stragglers = await listPrefix(prefix);
    await Promise.all(stragglers.map((k) => deleteKey(k)));
  };

  return { getJson, putJson, listPrefix, delete: deleteKey, deletePrefix };
}
