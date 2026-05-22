/**
 * Simple inline SVG icon component.
 * Uses a small built-in set — extend as needed.
 * No external icon library required.
 */

type IconName =
  | 'dataset'
  | 'plus'
  | 'pipeline'
  | 'settings'
  | 'chevron-left'
  | 'chevron-right'
  | 'download'
  | 'gantt'
  | 'check'
  | 'x'
  | 'clock'
  | 'comment'
  | 'folder'
  | 'play';

interface IconProps {
  name: IconName;
  size?: number;
  color?: string;
  className?: string;
}

const PATHS: Record<IconName, string> = {
  'dataset':       'M3 5h18M3 10h18M3 15h18M3 20h18',
  'plus':          'M12 5v14M5 12h14',
  'pipeline':      'M4 6h16M4 12h10M4 18h13',
  'settings':      'M12 15.5A3.5 3.5 0 1 0 12 8.5a3.5 3.5 0 0 0 0 7zm7.43-2.37c.04-.32.07-.64.07-.63a7 7 0 0 0-.07-1L20.45 10a1 1 0 0 0-.27-1.34l-2-1.38a6.8 6.8 0 0 0-.8-1.38l.42-2.2a1 1 0 0 0-.7-1.18l-2.28-.61a6.86 6.86 0 0 0-1.5-.8L12.75 1a1 1 0 0 0-1.5 0l-1.57 1.11a6.86 6.86 0 0 0-1.5.8l-2.28.61a1 1 0 0 0-.7 1.18l.42 2.2a6.8 6.8 0 0 0-.8 1.38l-2 1.38A1 1 0 0 0 2.55 10l1.02 1.63c-.04.32-.07.64-.07.87s.03.55.07.87L2.55 14.63a1 1 0 0 0 .27 1.34l2 1.38c.23.5.5.96.8 1.38l-.42 2.2a1 1 0 0 0 .7 1.18l2.28.61c.47.3.97.55 1.5.8l1.57 1.11a1 1 0 0 0 1.5 0l1.57-1.11a6.86 6.86 0 0 0 1.5-.8l2.28-.61a1 1 0 0 0 .7-1.18l-.42-2.2a6.8 6.8 0 0 0 .8-1.38l2-1.38a1 1 0 0 0 .27-1.34l-1.02-1.63z',
  'chevron-left':  'M15 18l-6-6 6-6',
  'chevron-right': 'M9 18l6-6-6-6',
  'download':      'M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M7 10l5 5 5-5M12 15V3',
  'gantt':         'M3 3h18v4H3zM3 10h12v4H3zM3 17h15v4H3z',
  'check':         'M20 6L9 17l-5-5',
  'x':             'M18 6 6 18M6 6l12 12',
  'clock':         'M12 22c5.523 0 10-4.477 10-10S17.523 2 12 2 2 6.477 2 12s4.477 10 10 10zM12 6v6l4 2',
  'comment':       'M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z',
  'folder':        'M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z',
  'play':          'M5 3l14 9-14 9z',
};

export function Icon({ name, size = 16, color = 'currentColor', className }: IconProps) {
  const d = PATHS[name];
  const isFilled = name === 'gantt' || name === 'play';
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill={isFilled ? color : 'none'}
      stroke={isFilled ? 'none' : color}
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      aria-hidden="true"
    >
      <path d={d} />
    </svg>
  );
}
