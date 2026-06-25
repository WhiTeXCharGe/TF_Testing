import type React from 'react';
import { palette, font, radius, spacing } from './common';

export const toolbarStyles = {
  root: {
    backgroundColor: palette.bg,
    borderBottom: `1px solid ${palette.border}`,
    padding: `${spacing.xs}px ${spacing.md}px`,
    display: 'flex',
    flexDirection: 'column',
    gap: spacing.xs,
    flexShrink: 0,
  } as React.CSSProperties,

  row: {
    display: 'flex',
    gap: 8,
    alignItems: 'center',
  } as React.CSSProperties,

  divider: {
    width: 1,
    height: 20,
    backgroundColor: palette.border,
    flexShrink: 0,
  } as React.CSSProperties,

  actionBtn: (active: boolean): React.CSSProperties => ({
    padding: '4px 10px',
    backgroundColor: active ? palette.accent : '#bdbdbd',
    color: palette.textWhite,
    border: 'none',
    borderRadius: radius.sm,
    cursor: active ? 'pointer' : 'default',
    fontSize: font.size,
    fontFamily: font.family,
  }),

  submitBtn: (active: boolean): React.CSSProperties => ({
    padding: '4px 14px',
    backgroundColor: active ? palette.accentDark : '#bdbdbd',
    color: palette.textWhite,
    border: 'none',
    borderRadius: radius.sm,
    cursor: active ? 'pointer' : 'default',
    fontSize: font.size,
    fontFamily: font.family,
  }),
} as const;

export const filterBarStyles = {
  root: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: 6,
    alignItems: 'center',
    padding: `4px 0`,
  } as React.CSSProperties,

  textInput: {
    padding: '3px 8px',
    border: `1px solid ${palette.border}`,
    borderRadius: radius.pill,
    fontSize: font.size,
    fontFamily: font.family,
    width: 140,
    outline: 'none',
    background: palette.bgWhite,
  } as React.CSSProperties,

  dateInput: {
    padding: '2px 6px',
    border: `1px solid ${palette.border}`,
    borderRadius: radius.sm,
    fontSize: font.sizeSm,
    fontFamily: font.family,
    outline: 'none',
    background: palette.bgWhite,
  } as React.CSSProperties,

  dateSep: {
    fontSize: font.sizeSm,
    color: palette.textSecondary,
    flexShrink: 0,
  } as React.CSSProperties,

  dateGroup: {
    display: 'flex',
    alignItems: 'center',
    gap: 4,
    padding: '2px 8px',
    border: `1px solid ${palette.border}`,
    borderRadius: radius.pill,
    background: palette.bgWhite,
  } as React.CSSProperties,

  dateLabel: {
    fontSize: 10,
    color: palette.textSecondary,
    flexShrink: 0,
  } as React.CSSProperties,

  chip: (active: boolean): React.CSSProperties => ({
    padding: '3px 10px',
    border: `1px solid ${active ? palette.accent : palette.border}`,
    borderRadius: radius.pill,
    background: active ? palette.accentLight : palette.bgWhite,
    color: active ? palette.accentDark : '#444',
    fontWeight: active ? 700 : 400,
    cursor: 'pointer',
    fontSize: font.size,
    fontFamily: font.family,
    display: 'flex',
    alignItems: 'center',
    gap: 4,
    whiteSpace: 'nowrap',
    userSelect: 'none',
  }),

  chipClear: {
    cursor: 'pointer',
    color: palette.accent,
    fontWeight: 700,
    fontSize: 13,
    lineHeight: 1,
    padding: '0 2px',
  } as React.CSSProperties,

  clearAll: {
    padding: '3px 10px',
    border: `1px solid ${palette.border}`,
    borderRadius: radius.pill,
    background: 'transparent',
    color: palette.danger,
    cursor: 'pointer',
    fontSize: font.sizeSm,
    fontFamily: font.family,
    fontWeight: 700,
  } as React.CSSProperties,

  // Portal dropdown
  dropdown: {
    position: 'fixed',
    background: palette.bgWhite,
    border: `1px solid ${palette.borderMid}`,
    borderRadius: radius.md,
    boxShadow: '0 4px 12px rgba(0,0,0,0.18)',
    zIndex: 9999,
    minWidth: 200,
    maxWidth: 280,
    maxHeight: 300,
    display: 'flex',
    flexDirection: 'column',
    overflow: 'hidden',
  } as React.CSSProperties,

  dropdownSearch: {
    padding: '6px 8px',
    border: 'none',
    borderBottom: `1px solid ${palette.borderLight}`,
    fontSize: font.sizeSm,
    fontFamily: font.family,
    outline: 'none',
    width: '100%',
    boxSizing: 'border-box',
  } as React.CSSProperties,

  dropdownList: {
    overflowY: 'auto',
    flex: 1,
  } as React.CSSProperties,

  dropdownItem: (checked: boolean): React.CSSProperties => ({
    display: 'flex',
    alignItems: 'center',
    gap: 6,
    padding: '5px 10px',
    cursor: 'pointer',
    fontSize: font.sizeSm,
    fontFamily: font.family,
    background: checked ? palette.accentLight : 'transparent',
    color: checked ? palette.accentDark : '#333',
  }),

  dropdownEmpty: {
    padding: '10px',
    color: palette.textMuted,
    fontSize: font.sizeSm,
    fontFamily: font.family,
    textAlign: 'center',
  } as React.CSSProperties,
} as const;
