import { useMemo } from 'react';
import { useAppContext } from '../context/AppContext';
import { DeviceViewGantt } from '../components/GanttChart/DeviceViewGantt';
import { WorkerViewGantt } from '../components/GanttChart/WorkerViewGantt';
import { generateDateRange } from '../utils/dateUtils';
import { ScheduleData } from '../types/schedule';

// Same extent calculation GanttPage uses, duplicated here rather than shared
// since ViewPage intentionally has no dependency on the editable page's props.
function getScheduleExtent(schedule: ScheduleData): { startDate: string; endDate: string } | null {
  let min: string | null = null;
  let max: string | null = null;
  const consider = (d?: string) => {
    if (!d) return;
    if (min === null || d < min) min = d;
    if (max === null || d > max) max = d;
  };
  consider(schedule.planRange.startDate);
  consider(schedule.planRange.endDate);
  for (const wt of schedule.workflowTaskList) {
    for (const pt of wt.phaseTaskList) {
      consider(pt.startDate);
      consider(pt.endDate);
    }
  }
  for (const a of schedule.assignmentList) {
    consider(a.startDate);
    consider(a.endDate);
  }
  return min && max ? { startDate: min, endDate: max } : null;
}

const statusLabel: Record<string, string> = {
  connecting: '接続中…',
  connected: 'ライブ — ホストに接続済み',
  disconnected: '未接続 — ホストの共有待ち',
};

const statusColor: Record<string, string> = {
  connecting: '#e0a95c',
  connected: '#6bbf8a',
  disconnected: '#e0a95c',
};

export function ViewPage() {
  const { state } = useAppContext();
  const { schedule, currentView, session } = state;
  const viewConnectionStatus = session?.connectionStatus ?? 'disconnected';

  const scheduleExtent = useMemo(() => (schedule ? getScheduleExtent(schedule) : null), [schedule]);
  const dates = useMemo(() => {
    if (!scheduleExtent) return [];
    return generateDateRange(scheduleExtent.startDate, scheduleExtent.endDate);
  }, [scheduleExtent]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', overflow: 'hidden' }}>
      <div
        style={{
          padding: '6px 12px',
          backgroundColor: '#1a2e3f',
          color: '#fff',
          fontSize: 12,
          fontFamily: 'MS Gothic, monospace',
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          flexShrink: 0,
        }}
      >
        <span style={{ color: statusColor[viewConnectionStatus] }}>●</span>
        <span>ライブビュー（閲覧専用） — {statusLabel[viewConnectionStatus]}</span>
      </div>

      {/* pointer-events: none makes this a genuine read-only mirror — no click/drag
          handler in the reused Gantt components can fire, without touching any of
          their (fairly involved) internal drag logic. Known trade-off for this fast
          first cut: it also blocks the grid's own scroll/pan, so a viewer currently
          sees whatever portion of the timeline the host's default view renders. */}
      <div style={{ flex: 1, overflow: 'hidden', position: 'relative', pointerEvents: 'none' }}>
        {!schedule ? (
          <div
            style={{
              height: '100%',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: '#aaa',
              fontSize: 14,
              fontFamily: 'MS Gothic, monospace',
              backgroundColor: '#f8f9fa',
            }}
          >
            ホストがスケジュールを共有するのを待っています…
          </div>
        ) : currentView === 'device' ? (
          <DeviceViewGantt dates={dates} />
        ) : (
          <WorkerViewGantt dates={dates} />
        )}
      </div>
    </div>
  );
}
