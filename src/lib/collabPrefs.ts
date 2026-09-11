// Per-browser convenience: remember the display name so a returning
// participant doesn't retype it every session. localStorage only — never
// leaves this browser.
const DISPLAY_NAME_KEY = 'gantt.collab.displayName';

export function loadDisplayName(): string {
  try {
    return localStorage.getItem(DISPLAY_NAME_KEY) ?? '';
  } catch {
    return '';
  }
}

export function saveDisplayName(name: string): void {
  const trimmed = name.trim();
  if (!trimmed) return;
  try {
    localStorage.setItem(DISPLAY_NAME_KEY, trimmed);
  } catch {
    /* private mode / storage disabled — not worth surfacing */
  }
}
