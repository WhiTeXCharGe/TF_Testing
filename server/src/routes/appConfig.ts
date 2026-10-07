import { Router } from 'express';
import { readFile } from 'node:fs/promises';

// config.txt — the one place an installed copy of the app is pointed at
// "local" (this PC's bundled server / LAN) or "online" (the Azure session
// server), and at WHICH Azure server. It sits next to the exe so it can be
// edited after install with no rebuild and no UI. Plain `key=value` lines;
// blank lines and lines starting with # or ; are ignored.
//
//   mode=online
//   azure_url=https://example.azurecontainerapps.io

export type RuntimeConfig =
  | { mode: 'local' }
  | { mode: 'online'; azureUrl: string };

export type ParsedConfig =
  | { ok: true; config: RuntimeConfig }
  | { ok: false; error: string };

export function parseConfigText(text: string): ParsedConfig {
  const values = new Map<string, string>();
  for (const rawLine of text.replace(/^﻿/, '').split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith('#') || line.startsWith(';')) continue;
    const eq = line.indexOf('=');
    if (eq <= 0) continue;
    const key = line.slice(0, eq).trim().toLowerCase().replace(/-/g, '_');
    let value = line.slice(eq + 1).trim();
    if (value.length >= 2 && /^(["']).*\1$/.test(value)) value = value.slice(1, -1).trim();
    values.set(key, value);
  }

  const mode = (values.get('mode') ?? '').toLowerCase();
  if (mode === '') return { ok: false, error: 'config.txt に mode が指定されていません。mode=local または mode=online を指定してください。' };
  if (mode === 'local') return { ok: true, config: { mode: 'local' } };
  if (mode !== 'online') {
    return { ok: false, error: `config.txt の mode が正しくありません（"${values.get('mode')}"）。local または online を指定してください。` };
  }

  const url = values.get('azure_url') ?? '';
  if (url === '') return { ok: false, error: 'config.txt の mode=online には azure_url の指定が必要です。' };
  try {
    const parsed = new URL(url);
    if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') throw new Error('protocol');
  } catch {
    return { ok: false, error: `config.txt の azure_url が正しいURLではありません（"${url}"）。https:// から始まるURLを指定してください。` };
  }
  return { ok: true, config: { mode: 'online', azureUrl: url.replace(/\/+$/, '') } };
}

// GET /api/app-config — read fresh on every request so an edit to config.txt
// is picked up by the next window/restart without touching the server. No
// configured path (plain dev) means plain local mode, exactly the
// pre-config.txt behavior.
export function appConfigRouter(getPath: () => string | null | undefined): Router {
  const router = Router();
  router.get('/app-config', async (_req, res) => {
    const filePath = getPath();
    if (!filePath) {
      res.json({ ok: true, mode: 'local', azureUrl: null });
      return;
    }
    let text: string;
    try {
      text = await readFile(filePath, 'utf-8');
    } catch (err) {
      // A configured path that has no file is an error, not "local": the
      // desktop app recreates config.txt at launch, so a missing one means
      // something is wrong and silently going local would hide it.
      const detail = (err as NodeJS.ErrnoException).code === 'ENOENT' ? `ファイルが見つかりません（${filePath}）` : (err as Error).message;
      res.json({ ok: false, error: `config.txt を読み込めませんでした: ${detail}` });
      return;
    }
    const parsed = parseConfigText(text);
    if (!parsed.ok) {
      res.json({ ok: false, error: parsed.error });
      return;
    }
    res.json({
      ok: true,
      mode: parsed.config.mode,
      azureUrl: parsed.config.mode === 'online' ? parsed.config.azureUrl : null,
    });
  });
  return router;
}
