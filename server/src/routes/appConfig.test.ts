import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import express from 'express';
import request from 'supertest';
import { mkdtempSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { parseConfigText, appConfigRouter } from './appConfig.js';

describe('parseConfigText', () => {
  it('reads online mode with a url, ignoring comments, blank lines and trailing slashes', () => {
    const r = parseConfigText('# comment\r\n\r\n; another\r\nmode = online\r\nazure_url = https://x.example.com/\r\n');
    expect(r).toEqual({ ok: true, config: { mode: 'online', azureUrl: 'https://x.example.com' } });
  });

  it('reads local mode and ignores any azure_url', () => {
    expect(parseConfigText('mode=local\nazure_url=https://x.example.com')).toEqual({ ok: true, config: { mode: 'local' } });
  });

  it('is case-insensitive for keys and mode, tolerates quotes and a BOM', () => {
    const r = parseConfigText('﻿MODE="Online"\nAzure-Url=\'https://x.example.com\'');
    expect(r).toEqual({ ok: true, config: { mode: 'online', azureUrl: 'https://x.example.com' } });
  });

  it('a # inside the url value is not treated as a comment', () => {
    const r = parseConfigText('mode=online\nazure_url=https://x.example.com/#a');
    expect(r.ok && r.config.mode === 'online' && r.config.azureUrl).toBe('https://x.example.com/#a');
  });

  it('rejects a missing mode rather than silently choosing local', () => {
    expect(parseConfigText('azure_url=https://x.example.com').ok).toBe(false);
  });

  it('rejects an unknown mode (typo) rather than silently choosing local', () => {
    expect(parseConfigText('mode=onlne\nazure_url=https://x.example.com').ok).toBe(false);
  });

  it('rejects online mode without a usable url', () => {
    expect(parseConfigText('mode=online').ok).toBe(false);
    expect(parseConfigText('mode=online\nazure_url=not a url').ok).toBe(false);
    expect(parseConfigText('mode=online\nazure_url=ftp://x.example.com').ok).toBe(false);
  });
});

describe('GET /api/app-config', () => {
  let dir: string;
  beforeEach(() => { dir = mkdtempSync(path.join(tmpdir(), 'cfg-')); });
  afterEach(() => { rmSync(dir, { recursive: true, force: true }); });

  const appFor = (getPath: () => string | null) => {
    const app = express();
    app.use('/api', appConfigRouter(getPath));
    return app;
  };

  it('is plain local when no config path is set (dev)', async () => {
    const res = await request(appFor(() => null)).get('/api/app-config');
    expect(res.body).toEqual({ ok: true, mode: 'local', azureUrl: null });
  });

  it('reports an error (not local) when a configured file is missing', async () => {
    const res = await request(appFor(() => path.join(dir, 'config.txt'))).get('/api/app-config');
    expect(res.body.ok).toBe(false);
    expect(res.body.error).toMatch(/config\.txt/);
  });

  it('serves the file contents and re-reads on every request', async () => {
    const file = path.join(dir, 'config.txt');
    const app = appFor(() => file);
    writeFileSync(file, 'mode=online\nazure_url=https://a.example.com');
    expect((await request(app).get('/api/app-config')).body).toEqual({ ok: true, mode: 'online', azureUrl: 'https://a.example.com' });
    writeFileSync(file, 'mode=local');
    expect((await request(app).get('/api/app-config')).body).toEqual({ ok: true, mode: 'local', azureUrl: null });
  });

  it('reports a config error instead of falling back to local', async () => {
    const file = path.join(dir, 'config.txt');
    writeFileSync(file, 'mode=onlne');
    const res = await request(appFor(() => file)).get('/api/app-config');
    expect(res.body.ok).toBe(false);
    expect(res.body.error).toMatch(/mode/);
  });
});
