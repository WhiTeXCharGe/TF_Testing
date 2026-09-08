import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { createServer, Server as HttpServer } from 'node:http';
import { AddressInfo } from 'node:net';
import { io as ioClient, Socket as ClientSocket } from 'socket.io-client';
import { createCollabSocketServer } from './collabSocket.js';
import { createSessionStore } from './sessionStore.js';
import { createMemStorage } from './storage/memStorage.js';
import { createSessionRecord, hashOwnerToken, loadSessionRecord } from './persistence.js';
import { loadConfig } from '../config.js';

const BASELINE = { schedule: { assignments: [] }, envConfig: { workers: [] }, currentView: 'worker' as const };
const OWNER_TOKEN = 'owner-secret-123';
const config = loadConfig({}); // local: memory storage config, but we inject our own store

let httpServer: HttpServer;
let port: number;
let storage: ReturnType<typeof createMemStorage>;
let store: ReturnType<typeof createSessionStore>;
let sessionId: string;

beforeEach(async () => {
  storage = createMemStorage();
  store = createSessionStore({ storage });
  sessionId = await createSessionRecord(storage, {
    name: 'Test Session', baseline: BASELINE, ownerTokenHash: hashOwnerToken(OWNER_TOKEN),
  });
  httpServer = createServer();
  createCollabSocketServer(httpServer, store, config);
  await new Promise<void>((resolve) => httpServer.listen(0, resolve));
  port = (httpServer.address() as AddressInfo).port;
});

afterEach(async () => {
  await new Promise<void>((resolve) => httpServer.close(() => resolve()));
});

function connect(): ClientSocket {
  return ioClient(`http://localhost:${port}`, { path: '/collab/socket.io', transports: ['websocket'] });
}

// Attach connect + sync-init listeners in the same tick the socket is made,
// before anything else runs — attaching later is a race on a fast local
// connection (see the note kept from the original suite).
function joinAndWaitForSync(
  client: ClientSocket,
  payload: { sessionId: string; name: string; role: 'edit' | 'view'; ownerToken?: string },
): Promise<any> {
  return new Promise<any>((resolve) => {
    client.on('connect', () => client.emit('join', payload));
    client.on('sync-init', resolve);
  });
}

describe('join', () => {
  it('lazily activates a session that only exists in storage and replies with baseline + status open', async () => {
    expect(store.isLoaded(sessionId)).toBe(false);
    const client = connect();
    const syncInit = await joinAndWaitForSync(client, { sessionId, name: 'Alice', role: 'edit' });
    expect(syncInit).toEqual({
      ok: true,
      name: 'Test Session',
      baseline: BASELINE,
      actions: [],
      participants: [{ id: expect.any(String), name: 'Alice', role: 'edit' }],
      status: 'open',
    });
    client.disconnect();
  });

  it('replies with ok:false for an unknown session id', async () => {
    const client = connect();
    const syncInit = await joinAndWaitForSync(client, { sessionId: 'nope', name: 'Alice', role: 'edit' });
    expect(syncInit).toEqual({ ok: false });
    client.disconnect();
  });
});

describe('action relay', () => {
  it('broadcasts an edit-role action to other participants but not back to the sender', async () => {
    const alice = connect();
    const bob = connect();
    const aliceReady = joinAndWaitForSync(alice, { sessionId, name: 'Alice', role: 'edit' });
    const bobReady = joinAndWaitForSync(bob, { sessionId, name: 'Bob', role: 'edit' });
    await aliceReady;
    await bobReady;

    const bobReceived = new Promise<any>((resolve) => bob.on('action', resolve));
    let aliceReceivedOwnAction = false;
    alice.on('action', () => { aliceReceivedOwnAction = true; });

    alice.emit('action', { type: 'UPDATE_PLAN_RANGE', payload: { startDate: '2026-02-01', endDate: '2026-02-28' } });

    expect(await bobReceived).toEqual({ type: 'UPDATE_PLAN_RANGE', payload: { startDate: '2026-02-01', endDate: '2026-02-28' } });
    expect(aliceReceivedOwnAction).toBe(false);
    alice.disconnect();
    bob.disconnect();
  });

  it('ignores actions from view-role participants', async () => {
    const alice = connect();
    const viewer = connect();
    const aliceReady = joinAndWaitForSync(alice, { sessionId, name: 'Alice', role: 'edit' });
    const viewerReady = joinAndWaitForSync(viewer, { sessionId, name: 'Viewer', role: 'view' });
    await aliceReady;
    await viewerReady;

    let aliceReceivedAction = false;
    alice.on('action', () => { aliceReceivedAction = true; });
    viewer.emit('action', { type: 'UPDATE_PLAN_RANGE', payload: { startDate: '2026-03-01', endDate: '2026-03-31' } });

    await new Promise((resolve) => setTimeout(resolve, 200));
    expect(aliceReceivedAction).toBe(false);
    alice.disconnect();
    viewer.disconnect();
  });
});

describe('lock / unlock', () => {
  it('an owner emitting "lock" flips status and broadcasts session-status to the room', async () => {
    const owner = connect();
    const other = connect();
    const ownerReady = joinAndWaitForSync(owner, { sessionId, name: 'Owner', role: 'edit', ownerToken: OWNER_TOKEN });
    const otherReady = joinAndWaitForSync(other, { sessionId, name: 'Other', role: 'edit' });
    await ownerReady;
    await otherReady;

    const ownerSaw = new Promise<any>((resolve) => owner.on('session-status', resolve));
    const otherSaw = new Promise<any>((resolve) => other.on('session-status', resolve));
    owner.emit('lock');
    expect(await ownerSaw).toEqual({ status: 'lock' });
    expect(await otherSaw).toEqual({ status: 'lock' });
    expect(store.getSession(sessionId)?.status).toBe('lock');
    owner.disconnect();
    other.disconnect();
  });

  it('a non-owner emitting "lock" is ignored', async () => {
    const notOwner = connect();
    await joinAndWaitForSync(notOwner, { sessionId, name: 'NotOwner', role: 'edit', ownerToken: 'wrong' });
    let gotStatus = false;
    notOwner.on('session-status', () => { gotStatus = true; });
    notOwner.emit('lock');
    await new Promise((resolve) => setTimeout(resolve, 150));
    expect(gotStatus).toBe(false);
    expect(store.getSession(sessionId)?.status).toBe('open');
    notOwner.disconnect();
  });

  it('while locked no action is broadcast; after unlock it flows again', async () => {
    const owner = connect();
    const bob = connect();
    const ownerReady = joinAndWaitForSync(owner, { sessionId, name: 'Owner', role: 'edit', ownerToken: OWNER_TOKEN });
    const bobReady = joinAndWaitForSync(bob, { sessionId, name: 'Bob', role: 'edit' });
    await ownerReady;
    await bobReady;

    await new Promise<void>((resolve) => { owner.on('session-status', () => resolve()); owner.emit('lock'); });

    let bobGot = false;
    bob.on('action', () => { bobGot = true; });
    owner.emit('action', { type: 'UPDATE_PLAN_RANGE', payload: { startDate: 'a', endDate: 'b' } });
    await new Promise((resolve) => setTimeout(resolve, 200));
    expect(bobGot).toBe(false);

    await new Promise<void>((resolve) => { owner.on('session-status', () => resolve()); owner.emit('unlock'); });
    const bobReceives = new Promise<any>((resolve) => bob.on('action', resolve));
    owner.emit('action', { type: 'UPDATE_PLAN_RANGE', payload: { startDate: 'c', endDate: 'd' } });
    expect(await bobReceives).toEqual({ type: 'UPDATE_PLAN_RANGE', payload: { startDate: 'c', endDate: 'd' } });
    owner.disconnect();
    bob.disconnect();
  });
});

describe('participant cap', () => {
  it('rejects a non-owner join past maxParticipants but lets the owner in', async () => {
    const capServer = createServer();
    const capStore = createSessionStore({ storage });
    const capConfig = { ...config, limits: { ...config.limits, maxParticipants: 1 } };
    createCollabSocketServer(capServer, capStore, capConfig);
    await new Promise<void>((resolve) => capServer.listen(0, resolve));
    const capPort = (capServer.address() as AddressInfo).port;
    const cc = () => ioClient(`http://localhost:${capPort}`, { path: '/collab/socket.io', transports: ['websocket'] });

    const first = cc();
    await new Promise<any>((res) => { first.on('connect', () => first.emit('join', { sessionId, name: 'One', role: 'edit' })); first.on('sync-init', res); });

    const second = cc();
    const secondSync = await new Promise<any>((res) => { second.on('connect', () => second.emit('join', { sessionId, name: 'Two', role: 'edit' })); second.on('sync-init', res); });
    expect(secondSync).toEqual({ ok: false, error: 'session is full' });

    const owner = cc();
    const ownerSync = await new Promise<any>((res) => { owner.on('connect', () => owner.emit('join', { sessionId, name: 'Owner', role: 'edit', ownerToken: OWNER_TOKEN })); owner.on('sync-init', res); });
    expect(ownerSync.ok).toBe(true);

    first.disconnect(); second.disconnect(); owner.disconnect();
    await new Promise<void>((resolve) => capServer.close(() => resolve()));
  });
});

describe('presence + last-leave persistence', () => {
  it('notifies remaining participants when someone disconnects', async () => {
    const alice = connect();
    const bob = connect();
    const aliceReady = joinAndWaitForSync(alice, { sessionId, name: 'Alice', role: 'edit' });
    const bobReady = joinAndWaitForSync(bob, { sessionId, name: 'Bob', role: 'edit' });
    await aliceReady;
    await bobReady;

    const aliceSawPresenceDrop = new Promise<any>((resolve) => {
      alice.on('presence', (participants: unknown[]) => {
        if (participants.length === 1) resolve(participants);
      });
    });
    bob.disconnect();
    expect(await aliceSawPresenceDrop).toEqual([{ id: expect.any(String), name: 'Alice', role: 'edit' }]);
    alice.disconnect();
  });

  it('flushes the log and sets status close when the last participant leaves', async () => {
    const alice = connect();
    await joinAndWaitForSync(alice, { sessionId, name: 'Alice', role: 'edit' });
    alice.emit('action', { type: 'SET_SCHEDULE', payload: { v: 1 } });
    // Wait until the relay has actually recorded the action before leaving,
    // so this isn't racing the in-flight 'action' packet against disconnect.
    for (let i = 0; i < 40 && (store.getSession(sessionId)?.actions.length ?? 0) === 0; i++) {
      await new Promise((r) => setTimeout(r, 25));
    }
    expect(store.getSession(sessionId)?.actions).toHaveLength(1);
    alice.disconnect();

    // The last-leave handler flushes then evicts, so "no longer loaded here"
    // is the signal that the async flush + markClosed have completed.
    for (let i = 0; i < 80 && store.isLoaded(sessionId); i++) {
      await new Promise((r) => setTimeout(r, 25));
    }
    expect(store.isLoaded(sessionId)).toBe(false);

    const rec = await loadSessionRecord(storage, sessionId);
    expect(rec?.status.status).toBe('close');
    expect(rec?.log).toEqual([{ seq: 0, type: 'SET_SCHEDULE', payload: { v: 1 } }]);
  });
});
