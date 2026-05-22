// Date utility functions — all pure, no external deps.

/** Parse YYYY/MM/DD or YYYY-MM-DD → Date (local midnight). */
export function parseDate(s: string): Date {
  const clean = s.replace(/\//g, '-');
  const [y, m, d] = clean.split('-').map(Number);
  return new Date(y, m - 1, d);
}

/** Format Date → "YYYY/MM/DD". */
export function formatDate(d: Date): string {
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, '0');
  const dd = String(d.getDate()).padStart(2, '0');
  return `${y}/${m}/${dd}`;
}

/** Date → "YYYY-MM-DD" key for Maps. */
export function toKey(d: Date): string {
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, '0');
  const dd = String(d.getDate()).padStart(2, '0');
  return `${y}-${m}-${dd}`;
}

/** Generate array of dates from start to end inclusive. */
export function dateRange(start: Date, end: Date): Date[] {
  const dates: Date[] = [];
  const cur = new Date(start);
  while (cur <= end) {
    dates.push(new Date(cur));
    cur.setDate(cur.getDate() + 1);
  }
  return dates;
}

/** true if Saturday (6) or Sunday (0). */
export function isWeekend(d: Date): boolean {
  const dow = d.getDay();
  return dow === 0 || dow === 6;
}

/** Short weekday abbreviation: Mon, Tue, … */
export function dowAbbr(d: Date): string {
  return ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'][d.getDay()];
}

/** Short month abbreviation: Jan, Feb, … */
export function monthName(d: Date): string {
  return ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][d.getMonth()];
}

/** Format elapsed seconds as "HH:MM:SS". */
export function formatTimer(sec: number): string {
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = sec % 60;
  return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

/**
 * Human-readable elapsed duration from two ISO strings.
 * Returns e.g. "1h 23m" or "45m 10s".
 */
export function formatElapsed(startIso: string, endIso: string | null): string {
  if (!endIso) return '—';
  const diffSec = Math.floor((new Date(endIso).getTime() - new Date(startIso).getTime()) / 1000);
  if (diffSec < 0) return '—';
  const h = Math.floor(diffSec / 3600);
  const m = Math.floor((diffSec % 3600) / 60);
  const s = diffSec % 60;
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

/** ISO string → "YYYY/MM/DD HH:MM" display label. */
export function nowLabel(iso: string): string {
  const d = new Date(iso);
  const date = formatDate(d);
  const hh = String(d.getHours()).padStart(2, '0');
  const mm = String(d.getMinutes()).padStart(2, '0');
  return `${date} ${hh}:${mm}`;
}
