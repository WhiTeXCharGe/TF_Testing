import { BlobServiceClient, ContainerClient, RestError } from '@azure/storage-blob';
import type { StorageClient } from './storageClient.js';

// Azure Blob Storage implementation of StorageClient. One block blob per key;
// the key is the blob name verbatim (e.g. "sessions/<id>/meta.json"). The
// container is created on first use so deployment doesn't need a separate
// provisioning step for it.
export function createBlobStorage(connectionString: string, containerName: string): StorageClient {
  const service = BlobServiceClient.fromConnectionString(connectionString);
  const container: ContainerClient = service.getContainerClient(containerName);

  let ensured: Promise<void> | null = null;
  const ensureContainer = (): Promise<void> => {
    if (!ensured) ensured = container.createIfNotExists().then(() => undefined);
    return ensured;
  };

  const is404 = (err: unknown): boolean =>
    err instanceof RestError && (err.statusCode === 404 || err.code === 'BlobNotFound' || err.code === 'ContainerNotFound');

  const getJson = async <T>(key: string): Promise<T | null> => {
    await ensureContainer();
    try {
      const buf = await container.getBlockBlobClient(key).downloadToBuffer();
      return JSON.parse(buf.toString('utf-8')) as T;
    } catch (err) {
      if (is404(err)) return null;
      throw err;
    }
  };

  const putJson = async (key: string, value: unknown): Promise<void> => {
    await ensureContainer();
    const body = JSON.stringify(value, null, 2);
    await container.getBlockBlobClient(key).upload(body, Buffer.byteLength(body), {
      blobHTTPHeaders: { blobContentType: 'application/json' },
    });
  };

  const listPrefix = async (prefix: string): Promise<string[]> => {
    await ensureContainer();
    const names: string[] = [];
    for await (const blob of container.listBlobsFlat({ prefix })) {
      names.push(blob.name);
    }
    return names.sort();
  };

  const deleteKey = async (key: string): Promise<void> => {
    await ensureContainer();
    await container.getBlockBlobClient(key).deleteIfExists();
  };

  const deletePrefix = async (prefix: string): Promise<void> => {
    const keys = await listPrefix(prefix);
    await Promise.all(keys.map((k) => container.getBlockBlobClient(k).deleteIfExists()));
  };

  return { getJson, putJson, listPrefix, delete: deleteKey, deletePrefix };
}
