import { describe, it, expect } from 'vitest';
import { createAca2Client } from './aca2Client.js';

const cfg = { aca2Url: 'http://aca2:4010', internalKey: 'k' };
const jsonResponse = (body: unknown, status = 200): Response =>
  ({ ok: status >= 200 && status < 300, status, json: async () => body } as Response);

describe('aca2Client', () => {
  it('activate returns relayUrl + status', async () => {
    const c = createAca2Client(cfg, (async () => jsonResponse({ ok: true, relayUrl: 'http://r', status: 'open' })) as typeof fetch);
    expect(await c.activate('s1')).toEqual({ relayUrl: 'http://r', status: 'open' });
  });

  it('activate maps 404 to notFound', async () => {
    const c = createAca2Client(cfg, (async () => jsonResponse({ ok: false }, 404)) as typeof fetch);
    expect(await c.activate('s1')).toEqual({ notFound: true });
  });

  it('live returns the live view', async () => {
    const c = createAca2Client(cfg, (async () => jsonResponse({ ok: true, active: true, participantCount: 3, status: 'lock' })) as typeof fetch);
    expect(await c.live('s1')).toEqual({ active: true, participantCount: 3, status: 'lock' });
  });

  it('live maps a thrown fetch to unreachable', async () => {
    const c = createAca2Client(cfg, (async () => { throw new Error('ECONNREFUSED'); }) as typeof fetch);
    expect(await c.live('s1')).toEqual({ unreachable: true });
  });

  it('live maps a non-2xx to unreachable', async () => {
    const c = createAca2Client(cfg, (async () => jsonResponse({ ok: false }, 503)) as typeof fetch);
    expect(await c.live('s1')).toEqual({ unreachable: true });
  });

  it('evict swallows network errors', async () => {
    const c = createAca2Client(cfg, (async () => { throw new Error('down'); }) as typeof fetch);
    await expect(c.evict('s1')).resolves.toBeUndefined();
  });

  it('sends the internal key header and hits the right URL', async () => {
    let seenUrl = '';
    let seenInit: RequestInit | undefined;
    const c = createAca2Client(cfg, (async (url: string | URL | Request, init?: RequestInit) => {
      seenUrl = String(url);
      seenInit = init;
      return jsonResponse({ ok: true, active: false, participantCount: 0, status: 'close' });
    }) as typeof fetch);
    await c.live('abc 123');
    expect(seenUrl).toBe('http://aca2:4010/internal/sessions/abc%20123/live');
    expect((seenInit?.headers as Record<string, string>)['x-internal-key']).toBe('k');
  });
});
