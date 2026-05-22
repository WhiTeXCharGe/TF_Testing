import type { RunStatus } from '@/types';
import { UI } from '@/config/uiConfig';

interface BadgeProps {
  status: RunStatus | null;
}

const classMap: Record<RunStatus, string> = {
  Completed: 'badge b-comp',
  Failed:    'badge b-fail',
  Executing: 'badge b-exec',
};

export function Badge({ status }: BadgeProps) {
  if (!status) return <span className="badge" style={{ color: 'var(--text-dis)' }}>{UI.status.none}</span>;
  return (
    <span className={classMap[status]}>
      <span className="dot" />
      {UI.status[status]}
    </span>
  );
}
