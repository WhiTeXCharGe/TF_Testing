// Shared design tokens — import in other style files, never inline magic numbers
import type React from 'react';

export const palette = {
  // Brand
  accent: '#1976d2',
  accentDark: '#1565c0',
  accentLight: '#e3f0fb',
  success: '#2e7d32',
  danger: '#b71c1c',

  // Neutrals
  bg: '#f0f2f5',
  bgCard: '#fafbfc',
  bgWhite: '#ffffff',
  bgDark: '#1c2b3a',
  bgStatus: '#1a2e3f',

  // Borders
  border: '#d0d5dd',
  borderLight: '#e4ebf4',
  borderMid: '#c9d5e3',

  // Text
  textPrimary: '#1e334b',
  textSecondary: '#5a7fa0',
  textMuted: '#9aa8b8',
  textWhite: '#ffffff',

  // Weekend
  weekend: '#fff5f5',
  weekendText: '#b54747',
} as const;

export const font = {
  family: 'MS Gothic, monospace',
  size: 12,
  sizeSm: 11,
  sizeLg: 13,
} as const;

export const radius = {
  sm: 3,
  md: 4,
  pill: 16,
} as const;

export const spacing = {
  xs: 4,
  sm: 6,
  md: 10,
  lg: 16,
} as const;

export const shadow = {
  dropdown: '0 4px 12px rgba(0,0,0,0.18)',
  modal: '0 8px 28px rgba(0,0,0,0.35)',
} as const;

// Base button style builder
export function mkBtn(bg: string, disabled = false): React.CSSProperties {
  return {
    padding: '4px 10px',
    backgroundColor: bg,
    color: palette.textWhite,
    border: 'none',
    borderRadius: radius.sm,
    cursor: disabled ? 'default' : 'pointer',
    fontSize: font.size,
    fontFamily: font.family,
    opacity: disabled ? 0.4 : 1,
  };
}
