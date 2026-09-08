// A minimal key/value-of-JSON store. A "key" is always a '/'-joined path of
// segments, e.g. `sessions/<id>/meta.json`. Two implementations:
//   - fsStorage   — a local folder (the Azure-free mock, and local dev)
//   - blobStorage — Azure Blob Storage (added in the Azure deploy phase)
// Everything above the storage layer depends only on this interface.

export interface StorageClient {
  /** Parsed JSON at `key`, or null when the key is absent. */
  getJson<T>(key: string): Promise<T | null>;
  /** Write `value` as JSON at `key`, creating parents. Atomic (write + rename). */
  putJson(key: string, value: unknown): Promise<void>;
  /** Every key that starts with `prefix`, sorted ascending. `''` = all keys. */
  listPrefix(prefix: string): Promise<string[]>;
  /** Remove one key. No-op when it is already absent. */
  delete(key: string): Promise<void>;
  /** Remove every key that starts with `prefix`. */
  deletePrefix(prefix: string): Promise<void>;
}
