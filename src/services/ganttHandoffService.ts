// Sends a run's EnvConfig/Schedule YAML to GanttChartEditor (コピーファイル表示 / 結果を表示),
// auto-launching it via the dev-only /api/handoff/create route in vite.config.ts if needed.

export async function fetchText(url: string): Promise<string> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`ファイル取得に失敗しました: ${url}`);
  return res.text();
}

export async function sendToGanttEditor(envYaml: string, scheduleYaml: string): Promise<{ url: string }> {
  const res = await fetch('/api/handoff/create', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ envYaml, scheduleYaml }),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok || !data.ok || !data.url) {
    throw new Error(data.error ?? 'GanttChartEditorへの送信に失敗しました');
  }
  return { url: data.url };
}
