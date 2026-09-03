import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { createServer, Server as HttpServer } from 'node:http';
import { AddressInfo } from 'node:net';
import { io as ioClient, Socket as ClientSocket } from 'socket.io-client';
import { createCollabSocketServer } from './collabSocket.js';
import { createSession, _resetForTests } from './sessionStore.js';

const BASELINE = { schedule: { assignments: [] }, envConfig: { workers: [] }, currentView: 'worker' as const };

let httpServer: HttpServer;
let port: number;

beforeEach(async () => {
  _resetForTests();
  httpServer = createServer();
  createCollabSocketServer(httpServer);
  await new Promise<void>(resolve => httpServer.listen(0, resolve));
  port = (httpServer.address() as AddressInfo).port;
});

afterEach(async () => {
  await new Promise<void>(resolve => httpServer.close(() => resolve()));
});

function connect(): ClientSocket {
  return ioClient(`http://localhost:${port}`, { path: '/collab/socket.io', transports: ['websocket'] });
}

// Attaches the 'connect' and 'sync-init' listeners in the same tick the
// socket is created, before anything else gets a chance to run. Attaching
// them later (e.g. after awaiting another socket's full round trip) is a
// race: on a fast local connection, 'connect' can fire and be dropped before
// the listener exists, and the join is never sent — so this helper is used
// for every multi-socket test below, not just the first one to join.
function joinAndWaitForSync(client: ClientSocket, payload: { sessionId: string; name: string; role: 'edit' | 'view' }): Promise<any> {
  return new Promise<any>(resolve => {
    client.on('connect', () => client.emit('join', payload));
    client.on('sync-init', resolve);
  });
}

describe('join', () => {
  it('replies with the baseline, session name, empty action log, and participant list for a fresh session', async () => {
    const sessionId = createSession('Test Session', BASELINE);
    const client = connect();
    const syncInit = await new Promise<any>(resolve => {
      client.on('connect', () => client.emit('join', { sessionId, name: 'Alice', role: 'edit' }));
      client.on('sync-init', resolve);
    });
    expect(syncInit).toEqual({ ok: true, name: 'Test Session', baseline: BASELINE, actions: [], participants: [{ id: expect.any(String), name: 'Alice', role: 'edit' }] });
    client.disconnect();
  });

  it('replies with ok:false for an unknown session id', async () => {
    const client = connect();
    const syncInit = await new Promise<any>(resolve => {
      client.on('connect', () => client.emit('join', { sessionId: 'nope', name: 'Alice', role: 'edit' }));
      client.on('sync-init', resolve);
    });
    expect(syncInit).toEqual({ ok: false });
    client.disconnect();
  });
});

describe('action relay', () => {
  it('broadcasts an edit-role action to other participants but not back to the sender', async () => {
    const sessionId = createSession('Test Session', BASELINE);
    const alice = connect();
    const bob = connect();
    const aliceReady = joinAndWaitForSync(alice, { sessionId, name: 'Alice', role: 'edit' });
    const bobReady = joinAndWaitForSync(bob, { sessionId, name: 'Bob', role: 'edit' });
    await aliceReady;
    await bobReady;

    const bobReceived = new Promise<any>(resolve => bob.on('action', resolve));
    let aliceReceivedOwnAction = false;
    alice.on('action', () => { aliceReceivedOwnAction = true; });

    alice.emit('action', { type: 'UPDATE_PLAN_RANGE', payload: { startDate: '2026-02-01', endDate: '2026-02-28' } });

    expect(await bobReceived).toEqual({ type: 'UPDATE_PLAN_RANGE', payload: { startDate: '2026-02-01', endDate: '2026-02-28' } });
    expect(aliceReceivedOwnAction).toBe(false);
    alice.disconnect();
    bob.disconnect();
  });

  it('ignores actions from view-role participants', async () => {
    const sessionId = createSession('Test Session', BASELINE);
    const alice = connect();
    const viewer = connect();
    const aliceReady = joinAndWaitForSync(alice, { sessionId, name: 'Alice', role: 'edit' });
    const viewerReady = joinAndWaitForSync(viewer, { sessionId, name: 'Viewer', role: 'view' });
    await aliceReady;
    await viewerReady;

    let aliceReceivedAction = false;
    alice.on('action', () => { aliceReceivedAction = true; });
    viewer.emit('action', { type: 'UPDATE_PLAN_RANGE', payload: { startDate: '2026-03-01', endDate: '2026-03-31' } });

    await new Promise(resolve => setTimeout(resolve, 200));
    expect(aliceReceivedAction).toBe(false);
    alice.disconnect();
    viewer.disconnect();
  });
});

describe('presence', () => {
  it('notifies remaining participants when someone disconnects', async () => {
    const sessionId = createSession('Test Session', BASELINE);
    const alice = connect();
    const bob = connect();
    const aliceReady = joinAndWaitForSync(alice, { sessionId, name: 'Alice', role: 'edit' });
    const bobReady = joinAndWaitForSync(bob, { sessionId, name: 'Bob', role: 'edit' });
    await aliceReady;
    await bobReady;

    // Bob's own join also broadcasts a 'presence' update to Alice (now [Alice,
    // Bob]) — filter for the one that reflects Bob leaving ([Alice] alone) so
    // this isn't racing that unrelated join notification.
    const aliceSawPresenceDrop = new Promise<any>(resolve => {
      alice.on('presence', (participants: unknown[]) => {
        if (participants.length === 1) resolve(participants);
      });
    });
    bob.disconnect();
    expect(await aliceSawPresenceDrop).toEqual([{ id: expect.any(String), name: 'Alice', role: 'edit' }]);
    alice.disconnect();
  });
});
