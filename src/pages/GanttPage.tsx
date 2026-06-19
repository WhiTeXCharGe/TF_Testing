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


export function GanttPage() {
  const { state } = useAppContext();
  const { schedule, currentView, undoStack, redoStack } = state;

  const dates = useMemo(() => {
    const start = state.displayStartDate ?? schedule?.planRange.startDate;
    const end = state.displayEndDate ?? schedule?.planRange.endDate;
    if (!start || !end) return [];
    return generateDateRange(start, end);
  }, [state.displayStartDate, state.displayEndDate, schedule]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', overflow: 'hidden' }}>
      {/* Menu bar (contains app title + File/Edit/View/Help menus) */}
      <MenuBar />

      {/* Toolbar */}
      <Toolbar />

      {/* Main content */}
      <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
        {!schedule ? (
          <div style={{
            flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center',
            color: '#aaa', fontSize: 14, fontFamily: 'MS Gothic, monospace',
            backgroundColor: '#f8f9fa',
          }}>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: 48, marginBottom: 16, opacity: 0.4 }}>📂</div>
              <div style={{ fontSize: 14, color: '#666', marginBottom: 8 }}>
                「{UI.fileMenu}」→「{UI.open}」でファイルを読み込んでください
              </div>
              <div style={{ fontSize: 12, color: '#aaa' }}>
                ショートカット: Ctrl+O
              </div>
            </div>
          </div>
        ) : (
          <>
            {currentView === 'device'
              ? <DeviceViewGantt dates={dates} />
              : <WorkerViewGantt dates={dates} />
            }
            <SidePanel />
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
