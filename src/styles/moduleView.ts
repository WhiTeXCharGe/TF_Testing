// Module (Device) view styles
// TODO: migrate inline styles from DeviceViewGantt.tsx here
import type React from 'react';
import { palette, font } from './common';

export const moduleViewStyles = {
  leftHeader: {
    display: 'flex',
    alignItems: 'center',
    padding: '0 10px',
    fontWeight: 700,
    fontSize: font.size,
    color: palette.textPrimary,
    background: '#f2f6fb',
    borderBottom: `1px solid ${palette.borderMid}`,
  } as React.CSSProperties,

  labelRowKoutei: {
    background: '#e8eef5',
    fontWeight: 700,
    cursor: 'pointer',
  } as React.CSSProperties,

  labelRowTask: {
    background: palette.bgWhite,
    fontWeight: 400,
    cursor: 'pointer',
  } as React.CSSProperties,

  weekendTint: {
    background: palette.weekend,
  } as React.CSSProperties,
} as const;
