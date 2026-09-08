import type { StorageClient } from './storageClient.js';

// In-process StorageClient backed by a Map. Used by ROLE=local (keeps the
// Electron/LAN relay's state purely in memory, as it was before persistence
// existed) and by unit tests that don't want a temp directory.
export function createMemStorage(): StorageClient {
  const map = new Map<string, string>();
  return {
    async getJson<T>(key: string): Promise<T | null> {
      const v = map.get(key);
      return v === undefined ? null : (JSON.parse(v) as T);
    },
    async putJson(key: string, value: unknown): Promise<void> {
      map.set(key, JSON.stringify(value));
    },
    async listPrefix(prefix: string): Promise<string[]> {
      return [...map.keys()].filter((k) => k.startsWith(prefix)).sort();
    },
    async delete(key: string): Promise<void> {
      map.delete(key);
    },
    async deletePrefix(prefix: string): Promise<void> {
      for (const k of [...map.keys()]) if (k.startsWith(prefix)) map.delete(k);
    },
  };
}
