import fs from 'fs';

// config.txt — the one place an installed copy of the app is pointed at the
// "local" solver API (this PC, http://localhost:3001) or the "online" one
// (Azure), and at WHICH Azure server. It sits next to the exe so it can be
// edited after install with no rebuild and no UI — same convention as
// GanttChartEditor's config.txt. Plain `key=value` lines; blank lines and
// lines starting with # or ; are ignored.
//
//   mode=online
//   azure_url=https://example.azurecontainerapps.io
//   local_url=http://localhost:3001   (optional, mode=local only)

export const DEFAULT_LOCAL_URL = 'http://localhost:3001';

export interface RuntimeConfig {
  mode: 'local' | 'online';
  apiBaseUrl: string;
}

export type ParsedConfig =
  | { ok: true; config: RuntimeConfig }
  | { ok: false; error: string };

function validUrl(url: string): boolean {
  try {
    const parsed = new URL(url);
    return parsed.protocol === 'http:' || parsed.protocol === 'https:';
  } catch {
    return false;
  }
}

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

  if (mode === 'local') {
    const url = values.get('local_url') || DEFAULT_LOCAL_URL;
    if (!validUrl(url)) return { ok: false, error: `config.txt の local_url が正しいURLではありません（"${url}"）。` };
    return { ok: true, config: { mode: 'local', apiBaseUrl: url.replace(/\/+$/, '') } };
  }
  if (mode !== 'online') {
    return { ok: false, error: `config.txt の mode が正しくありません（"${values.get('mode')}"）。local または online を指定してください。` };
  }

  const url = values.get('azure_url') ?? '';
  if (url === '') return { ok: false, error: 'config.txt の mode=online には azure_url の指定が必要です。' };
  if (!validUrl(url)) {
    return { ok: false, error: `config.txt の azure_url が正しいURLではありません（"${url}"）。https:// から始まるURLを指定してください。` };
  }
  return { ok: true, config: { mode: 'online', apiBaseUrl: url.replace(/\/+$/, '') } };
}

/**
 * Read config.txt fresh (called on every GET /api/app-config) so an edit is
 * picked up by the next window reload/restart. No configured path means the
 * caller should fall back to its build-time default. A configured path with
 * no file is an error, not "local": the desktop app recreates config.txt at
 * launch, so a missing one means something is wrong.
 */
export function readAppConfig(filePath: string | null | undefined):
  | { ok: true; mode: 'local' | 'online'; apiBaseUrl: string; configPath: string }
  | { ok: true; mode: null; apiBaseUrl: null; configPath: null }
  | { ok: false; error: string; configPath: string } {
  if (!filePath) return { ok: true, mode: null, apiBaseUrl: null, configPath: null };
  let text: string;
  try {
    text = fs.readFileSync(filePath, 'utf-8');
  } catch (err) {
    const detail = (err as NodeJS.ErrnoException).code === 'ENOENT' ? `ファイルが見つかりません（${filePath}）` : (err as Error).message;
    return { ok: false, error: `config.txt を読み込めませんでした: ${detail}`, configPath: filePath };
  }
  const parsed = parseConfigText(text);
  if (!parsed.ok) return { ok: false, error: parsed.error, configPath: filePath };
  return { ok: true, mode: parsed.config.mode, apiBaseUrl: parsed.config.apiBaseUrl, configPath: filePath };
}
