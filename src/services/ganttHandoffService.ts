// Sends a run's EnvConfig/Schedule YAML to GanttChartEditor (コピーファイル表示 / 結果を表示),
// auto-launching it via the dev-only /api/handoff/create route in vite.config.ts if needed.

export async function fetchText(url: string): Promise<string> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`ファイル取得に失敗しました: ${url}`);
  return res.text();
}

// Desktop mode returns { url: null } — the transfer URL was already delivered
// straight to GanttChartEditor's one window via IPC (see electron/main.cts's
// single-instance-lock handling), so the caller must NOT window.open() it too
// (that used to open a second, blank GanttChartEditor window racing the real
// one for the one-time token — the actual cause of "opens twice / empty data").
// Web mode returns { url } for the caller to window.open() as before.
export async function sendToGanttEditor(envYaml: string, scheduleYaml: string): Promise<{ url: string | null }> {
  if (window.electronAPI) {
    // Make sure GanttChartEditor is up before minting a token — /api/handoff/create's
    // own dev-only "spawn npm run dev:all" fallback won't work in a packaged app.
    const ensureUp = await window.electronAPI.launchGanttEditor();
    if (!ensureUp.ok) throw new Error(ensureUp.error ?? 'GanttChartEditorの起動に失敗しました');
  }

  const res = await fetch('/api/handoff/create', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ envYaml, scheduleYaml }),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok || !data.ok || !data.url) {
    throw new Error(data.error ?? 'GanttChartEditorへの送信に失敗しました');
  }

  if (window.electronAPI) {
    // Deliver the token straight to GanttChartEditor's window instead of window.open().
    const deliver = await window.electronAPI.launchGanttEditor(data.url);
    if (!deliver.ok) throw new Error(deliver.error ?? 'GanttChartEditorへの送信に失敗しました');
    return { url: null };
  }
  return { url: data.url };
}
