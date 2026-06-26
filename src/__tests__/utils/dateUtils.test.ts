import {
  normalizeDate,
  parseDate,
  formatDate,
  formatDateShort,
  addDays,
  diffDays,
  generateDateRange,
  getDayOfWeek,
  isWeekend,
  isDateInList,
} from '../../utils/dateUtils';

describe('normalizeDate', () => {
  it('returns YYYY-MM-DD as-is', () => {
    expect(normalizeDate('2025-09-15')).toBe('2025-09-15');
  });
  it('converts YYYY/MM/DD to YYYY-MM-DD', () => {
    expect(normalizeDate('2025/09/15')).toBe('2025-09-15');
  });
  it('pads single-digit month and day', () => {
    expect(normalizeDate('2025/1/5')).toBe('2025-01-05');
  });
  it('returns empty string for empty input', () => {
    expect(normalizeDate('')).toBe('');
  });
});

describe('parseDate', () => {
  it('returns a Date object at midnight local time', () => {
    const d = parseDate('2025-01-01');
    expect(d.getFullYear()).toBe(2025);
    expect(d.getMonth()).toBe(0);
    expect(d.getDate()).toBe(1);
  });
});

describe('formatDate', () => {
  it('formats a Date to YYYY-MM-DD', () => {
    const d = new Date('2025-07-04T00:00:00');
    expect(formatDate(d)).toBe('2025-07-04');
  });
  it('pads single digit month/day', () => {
    const d = new Date('2025-01-03T00:00:00');
    expect(formatDate(d)).toBe('2025-01-03');
  });
});

describe('formatDateShort', () => {
  it('returns MM/DD format', () => {
    expect(formatDateShort('2025-09-05')).toBe('09/05');
  });
});

describe('addDays', () => {
  it('adds positive days', () => {
    expect(addDays('2025-01-01', 5)).toBe('2025-01-06');
  });
  it('adds negative days (subtracts)', () => {
    expect(addDays('2025-01-10', -3)).toBe('2025-01-07');
  });
  it('crosses month boundary', () => {
    expect(addDays('2025-01-30', 3)).toBe('2025-02-02');
  });
  it('crosses year boundary', () => {
    expect(addDays('2025-12-31', 1)).toBe('2026-01-01');
  });
  it('adding 0 returns same date', () => {
    expect(addDays('2025-06-15', 0)).toBe('2025-06-15');
  });
});

describe('diffDays', () => {
  it('returns 0 for same date', () => {
    expect(diffDays('2025-01-01', '2025-01-01')).toBe(0);
  });
  it('returns positive for later date', () => {
    expect(diffDays('2025-01-01', '2025-01-06')).toBe(5);
  });
  it('returns negative for earlier date', () => {
    expect(diffDays('2025-01-10', '2025-01-01')).toBe(-9);
  });
  it('works across months', () => {
    expect(diffDays('2025-01-31', '2025-03-02')).toBe(30);
  });
});

describe('generateDateRange', () => {
  it('returns single date for same start/end', () => {
    expect(generateDateRange('2025-01-01', '2025-01-01')).toEqual(['2025-01-01']);
  });
  it('returns correct range of dates', () => {
    expect(generateDateRange('2025-01-01', '2025-01-05')).toEqual([
      '2025-01-01', '2025-01-02', '2025-01-03', '2025-01-04', '2025-01-05',
    ]);
  });
  it('returns empty array when start is after end', () => {
    expect(generateDateRange('2025-01-05', '2025-01-01')).toEqual([]);
  });
  it('crosses month boundary', () => {
    const range = generateDateRange('2025-01-30', '2025-02-02');
    expect(range).toEqual(['2025-01-30', '2025-01-31', '2025-02-01', '2025-02-02']);
  });
});

describe('getDayOfWeek', () => {
  it('2025-01-05 is Sunday (0)', () => {
    expect(getDayOfWeek('2025-01-05')).toBe(0);
  });
  it('2025-01-06 is Monday (1)', () => {
    expect(getDayOfWeek('2025-01-06')).toBe(1);
  });
  it('2025-01-11 is Saturday (6)', () => {
    expect(getDayOfWeek('2025-01-11')).toBe(6);
  });
});

describe('isWeekend', () => {
  it('returns true for Sunday', () => {
    expect(isWeekend('2025-01-05')).toBe(true);
  });
  it('returns true for Saturday', () => {
    expect(isWeekend('2025-01-11')).toBe(true);
  });
  it('returns false for Monday', () => {
    expect(isWeekend('2025-01-06')).toBe(false);
  });
  it('returns false for Friday', () => {
    expect(isWeekend('2025-01-10')).toBe(false);
  });
});

describe('isDateInList', () => {
  it('finds exact match', () => {
    expect(isDateInList('2025-01-01', ['2025-01-01', '2025-01-02'])).toBe(true);
  });
  it('returns false when not in list', () => {
    expect(isDateInList('2025-01-03', ['2025-01-01', '2025-01-02'])).toBe(false);
  });
  it('normalizes YYYY/MM/DD format', () => {
    expect(isDateInList('2025/01/01', ['2025-01-01'])).toBe(true);
  });
  it('handles empty list', () => {
    expect(isDateInList('2025-01-01', [])).toBe(false);
  });
});
