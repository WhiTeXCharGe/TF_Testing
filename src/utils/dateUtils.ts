// Normalize any date string to YYYY-MM-DD
export function normalizeDate(raw: string): string {
  if (!raw) return '';
  // Already YYYY-MM-DD
  if (/^\d{4}-\d{2}-\d{2}$/.test(raw)) return raw;
  // YYYY/MM/DD or YYYY/M/D
  const parts = raw.split('/');
  if (parts.length === 3) {
    const [y, m, d] = parts;
    return `${y}-${m.padStart(2, '0')}-${d.padStart(2, '0')}`;
  }
  return raw;
}

// Parse a normalized YYYY-MM-DD string into a Date (midnight UTC)
export function parseDate(dateStr: string): Date {
  return new Date(dateStr + 'T00:00:00');
}

// Format a Date to YYYY-MM-DD
export function formatDate(date: Date): string {
  const y = date.getFullYear();
  const m = String(date.getMonth() + 1).padStart(2, '0');
  const d = String(date.getDate()).padStart(2, '0');
  return `${y}-${m}-${d}`;
}

// Format for display: MM/DD
export function formatDateShort(dateStr: string): string {
  const [, m, d] = dateStr.split('-');
  return `${m}/${d}`;
}

// Add N calendar days to a YYYY-MM-DD string
export function addDays(dateStr: string, days: number): string {
  const date = parseDate(dateStr);
  date.setDate(date.getDate() + days);
  return formatDate(date);
}

// Difference in days between two YYYY-MM-DD strings (to - from)
export function diffDays(from: string, to: string): number {
  return Math.round(
    (parseDate(to).getTime() - parseDate(from).getTime()) / 86_400_000
  );
}

// Generate all calendar dates between start and end (inclusive)
export function generateDateRange(start: string, end: string): string[] {
  const dates: string[] = [];
  let current = start;
  while (current <= end) {
    dates.push(current);
    current = addDays(current, 1);
  }
  return dates;
}

// Day-of-week (0=Sun, 6=Sat)
export function getDayOfWeek(dateStr: string): number {
  return parseDate(dateStr).getDay();
}

export function isWeekend(dateStr: string): boolean {
  const dow = getDayOfWeek(dateStr);
  return dow === 0 || dow === 6;
}

// Check if a date string is in a list (handles YYYY/MM/DD and YYYY-MM-DD)
export function isDateInList(dateStr: string, list: string[]): boolean {
  const normalized = normalizeDate(dateStr);
  return list.some(d => normalizeDate(d) === normalized);
}
