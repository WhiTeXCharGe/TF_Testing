// SidePanel styles
// TODO: migrate inline styles from SidePanel.tsx here
import type React from 'react';
import { palette, font, radius } from './common';

export const sidePanelStyles = {
  root: {
    position: 'absolute' as const,
    right: 0,
    top: 0,
    bottom: 0,
    width: 300,
    borderLeft: `1px solid ${palette.borderMid}`,
    background: palette.bgWhite,
    overflowY: 'auto' as const,
    fontFamily: font.family,
    fontSize: font.size,
    zIndex: 10,
  } as React.CSSProperties,

  title: {
    fontSize: font.sizeLg,
    fontWeight: 'bold',
    color: palette.accentDark,
    borderBottom: `1px solid ${palette.border}`,
    paddingBottom: 6,
    marginBottom: 12,
  } as React.CSSProperties,

  fieldLabel: {
    color: '#666',
    fontSize: font.sizeSm,
    fontWeight: 700,
    display: 'block',
    marginBottom: 2,
  } as React.CSSProperties,

  input: {
    padding: '4px 6px',
    border: `1px solid ${palette.borderMid}`,
    borderRadius: radius.sm,
    fontSize: font.sizeSm,
    fontFamily: font.family,
    boxSizing: 'border-box' as const,
  } as React.CSSProperties,

  deleteBtn: {
    padding: '4px 12px',
    backgroundColor: palette.danger,
    color: palette.textWhite,
    border: 'none',
    borderRadius: radius.sm,
    cursor: 'pointer',
    fontSize: font.sizeSm,
    fontFamily: font.family,
    marginTop: 8,
  } as React.CSSProperties,
} as const;
