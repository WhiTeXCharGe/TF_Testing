import type { StorageConfig } from '../../config.js';
import type { StorageClient } from './storageClient.js';
import { createFsStorage } from './fsStorage.js';
import { createMemStorage } from './memStorage.js';
import { createBlobStorage } from './blobStorage.js';

/** Build the StorageClient a role should use, from its resolved config. */
export function makeStorage(cfg: StorageConfig): StorageClient {
  switch (cfg.kind) {
    case 'fs':
      return createFsStorage(cfg.rootDir);
    case 'memory':
      return createMemStorage();
    case 'blob':
      return createBlobStorage(cfg.connectionString, cfg.container);
  }
}

export { createFsStorage } from './fsStorage.js';
export { createMemStorage } from './memStorage.js';
export { createBlobStorage } from './blobStorage.js';
export type { StorageClient } from './storageClient.js';
