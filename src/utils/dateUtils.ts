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

export interface RangeOverlayGeom {
  left: number;
  width: number;
  /** true when rangeStart itself is one of the visible dates (boundary marker should show) */
  startInView: boolean;
  /** true when rangeEnd itself is one of the visible dates (boundary marker should show) */
  endInView: boolean;
  startDate: string;
  endDate: string;
}

// Compute the pixel geometry of a [rangeStart, rangeEnd] span clipped to the visible `dates` window.
// Returns null when the range doesn't overlap the visible dates at all.
export function getRangeOverlayGeom(dates: string[], rangeStart: string, rangeEnd: string, cellWidth: number): RangeOverlayGeom | null {
  if (!rangeStart || !rangeEnd || dates.length === 0) return null;
  const viewStart = dates[0];
  const viewEnd = dates[dates.length - 1];
  const clampedStart = rangeStart < viewStart ? viewStart : rangeStart;
  const clampedEnd = rangeEnd > viewEnd ? viewEnd : rangeEnd;
  if (clampedEnd < clampedStart) return null;
  const startIdx = dates.indexOf(clampedStart);
  const endIdx = dates.indexOf(clampedEnd);
  if (startIdx === -1 || endIdx === -1) return null;
  return {
    left: startIdx * cellWidth,
    width: (endIdx - startIdx + 1) * cellWidth,
    startInView: rangeStart >= viewStart && rangeStart <= viewEnd,
    endInView: rangeEnd >= viewStart && rangeEnd <= viewEnd,
    startDate: clampedStart,
    endDate: clampedEnd,
  };
}
