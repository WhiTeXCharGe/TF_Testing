// Worker timeline view styles
// TODO: migrate inline styles from WorkerTimelineGrid.tsx and WorkerViewGantt.tsx here
import type React from 'react';
import { palette, font } from './common';

export const workerViewStyles = {
  rowHeaderCell: {
    display: 'flex',
    alignItems: 'center',
    padding: '0 8px',
    fontSize: font.sizeSm,
    fontFamily: font.family,
    borderBottom: `1px solid ${palette.borderLight}`,
    background: palette.bgWhite,
    overflow: 'hidden',
    whiteSpace: 'nowrap' as const,
    textOverflow: 'ellipsis',
  } as React.CSSProperties,

  headerCell: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: 10,
    fontFamily: font.family,
    borderRight: `1px solid ${palette.borderLight}`,
    color: palette.textPrimary,
  } as React.CSSProperties,

  weekendCell: {
    background: palette.weekend,
    color: palette.weekendText,
  } as React.CSSProperties,
} as const;
