// Dialog styles
// TODO: migrate inline styles from NewScheduleDialog, TaskAddDialog, FileOpenDialog, ErrorDialog here
import type React from 'react';
import { palette, font, radius, shadow } from './common';

export const dialogStyles = {
  overlay: {
    position: 'fixed',
    inset: 0,
    backgroundColor: 'rgba(0,0,0,0.45)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 900,
  } as React.CSSProperties,

  modal: (width = 540): React.CSSProperties => ({
    backgroundColor: palette.bgWhite,
    borderRadius: radius.md,
    width,
    maxHeight: '90vh',
    display: 'flex',
    flexDirection: 'column',
    boxShadow: shadow.modal,
    fontFamily: font.family,
    overflow: 'hidden',
  }),

  titleBar: {
    backgroundColor: palette.bgDark,
    color: palette.textWhite,
    padding: '10px 16px',
    fontSize: font.sizeLg,
    fontWeight: 'bold',
    flexShrink: 0,
  } as React.CSSProperties,

  body: {
    padding: 16,
    flex: 1,
    overflowY: 'auto',
  } as React.CSSProperties,

  footer: {
    display: 'flex',
    justifyContent: 'flex-end',
    gap: 8,
    padding: '12px 16px',
    borderTop: `1px solid ${palette.borderLight}`,
    backgroundColor: '#fafafa',
    flexShrink: 0,
  } as React.CSSProperties,

  input: {
    width: '100%',
    padding: '4px 6px',
    border: `1px solid ${palette.border}`,
    borderRadius: radius.sm,
    fontSize: font.size,
    boxSizing: 'border-box',
    fontFamily: font.family,
  } as React.CSSProperties,

  fieldLabel: {
    display: 'block',
    fontSize: font.sizeSm,
    color: '#555',
    fontWeight: 'bold',
    marginBottom: 3,
  } as React.CSSProperties,

  primaryBtn: {
    padding: '6px 20px',
    backgroundColor: palette.accent,
    color: palette.textWhite,
    border: 'none',
    borderRadius: radius.sm,
    cursor: 'pointer',
    fontSize: font.size,
    fontFamily: font.family,
  } as React.CSSProperties,

  cancelBtn: {
    padding: '6px 16px',
    border: `1px solid #aaa`,
    borderRadius: radius.sm,
    cursor: 'pointer',
    fontSize: font.size,
    backgroundColor: palette.bgWhite,
    fontFamily: font.family,
  } as React.CSSProperties,
} as const;
