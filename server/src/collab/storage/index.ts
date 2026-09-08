import type { StorageConfig } from '../../config.js';
import type { StorageClient } from './storageClient.js';
import { createFsStorage } from './fsStorage.js';
import { createMemStorage } from './memStorage.js';

/** Build the StorageClient a role should use, from its resolved config. */
export function makeStorage(cfg: StorageConfig): StorageClient {
  switch (cfg.kind) {
    case 'fs':
      return createFsStorage(cfg.rootDir);
    case 'memory':
      return createMemStorage();
    case 'blob':
      throw new Error('blob storage is not implemented until the Azure deploy phase');
  }
}

export { createFsStorage } from './fsStorage.js';
export { createMemStorage } from './memStorage.js';
export type { StorageClient } from './storageClient.js';
