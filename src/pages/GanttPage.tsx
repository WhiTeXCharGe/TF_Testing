import { useMemo } from 'react';
import { useAppContext } from '../context/AppContext';
import { MenuBar } from '../components/Toolbar/MenuBar';
import { Toolbar } from '../components/Toolbar/Toolbar';
import { SidePanel } from '../components/SidePanel/SidePanel';
import { DeviceViewGantt } from '../components/GanttChart/DeviceViewGantt';
import { WorkerViewGantt } from '../components/GanttChart/WorkerViewGantt';
import { FileOpenDialog } from '../components/Dialogs/FileOpenDialog';
import { UI } from '../config/uiText';
import { generateDateRange } from '../utils/dateUtils';
import { ScheduleData } from '../types/schedule';

// The Gantt chart must always show the full extent of actual schedule data
// (every assignment + phase task), regardless of plan_range. plan_range is only
// a Timefold-side calculation boundary — in this editor it's a visual highlight,
// never something that clips or resizes the visible timeline.
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

export function GanttPage() {
  const { state, dispatch } = useAppContext();
  const { schedule, currentView, undoStack, redoStack, selectedAssignmentIndex, selectedUnavailableInfo } = state;
  const showStickySidePanel = !!schedule && currentView === 'worker' &&
    (selectedAssignmentIndex !== null || selectedUnavailableInfo !== null);

  const scheduleExtent = useMemo(() => schedule ? getScheduleExtent(schedule) : null, [schedule]);

  const dates = useMemo(() => {
    const { currentView, workerViewFilter, moduleViewFilter } = state;
    const viewFilter = currentView === 'worker' ? workerViewFilter : moduleViewFilter;
    const start = viewFilter.startDate ?? scheduleExtent?.startDate;
    const end = viewFilter.endDate ?? scheduleExtent?.endDate;
    if (!start || !end) return [];
    return generateDateRange(start, end);
  }, [state.currentView, state.workerViewFilter, state.moduleViewFilter, scheduleExtent]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', overflow: 'hidden' }}>
      {/* Menu bar (contains app title + File/Edit/View/Help menus) */}
      <MenuBar />

      {/* Toolbar */}
      <Toolbar />

      {/* Main content */}
      <div style={{ flex: 1, display: 'flex', overflow: 'hidden', position: 'relative' }}>
        {!schedule ? (
          <div style={{
            flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center',
            color: '#aaa', fontSize: 14, fontFamily: 'MS Gothic, monospace',
            backgroundColor: '#f8f9fa',
          }}>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: 48, marginBottom: 16, opacity: 0.4 }}>📂</div>
              <div style={{ fontSize: 14, color: '#666', marginBottom: 8 }}>
                {UI.emptyStateInstruction(UI.fileMenu, UI.open)}
              </div>
              <div style={{ fontSize: 12, color: '#aaa' }}>
                {UI.shortcutOpenHint}
              </div>
            </div>
          </div>
        ) : (
          <>
            <div
              style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', overflow: 'hidden', marginRight: showStickySidePanel ? 300 : 0 }}
              onClick={() => {
                if (selectedAssignmentIndex !== null) dispatch({ type: 'SELECT_ASSIGNMENT', payload: null });
                if (selectedUnavailableInfo !== null) dispatch({ type: 'SELECT_UNAVAILABLE', payload: null });
              }}
            >
              {currentView === 'device'
                ? <DeviceViewGantt dates={dates} />
                : <WorkerViewGantt dates={dates} />
              }
            </div>
            {showStickySidePanel && <SidePanel />}
          </>
        )}
      </div>

      {/* Status bar */}
      <div style={{
        backgroundColor: '#1a2e3f',
        color: '#7a9bb5',
        padding: '2px 12px',
        fontSize: 11,
        fontFamily: 'MS Gothic, monospace',
        flexShrink: 0,
        display: 'flex',
        gap: 16,
        alignItems: 'center',
      }}>
        <span style={{ color: schedule ? '#6bbf8a' : '#7a9bb5' }}>
          {schedule ? `✓ ${UI.fileLoaded}` : UI.noFile}
        </span>
        <span>{UI.undoCount(undoStack.length)}</span>
        <span>{UI.redoCount(redoStack.length)}</span>
        <span style={{ marginLeft: 'auto', color: '#4a6a85' }}>{UI.shortcutHint}</span>
      </div>

      {/* File open dialog (reads state.isFileOpenDialogOpen) */}
      <FileOpenDialog />
    </div>
  );
}