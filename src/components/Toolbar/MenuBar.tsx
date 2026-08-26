import { useState, useEffect, useRef } from 'react';
import { useAppContext } from '../../context/AppContext';
import { overwriteSaveFiles } from '../../services/fileService';
import { exportScheduleToExcel } from '../../services/excelExportService';
import { generateDateRange } from '../../utils/dateUtils';
import { SaveAsDialog } from '../Dialogs/SaveAsDialog';
import { UI } from '../../config/uiText';

const NAVY = '#1c2b3a';
const NAVY_HOVER = '#2c4060';
const DROPDOWN_W = 220;

interface MenuItem {
  label: string;
  shortcut?: string;
  action?: () => void;
  disabled?: boolean;
  separator?: never;
}
interface SeparatorItem { separator: true; label?: never; shortcut?: never; action?: never; disabled?: never; }
type MenuEntry = MenuItem | SeparatorItem;

interface MenuDef {
  id: string;
  label: string;
  items: MenuEntry[];
}

export function MenuBar() {
  const { state, dispatch, leaveCollabSession } = useAppContext();
  const [openMenu, setOpenMenu] = useState<string | null>(null);
  const [showSaveAs, setShowSaveAs] = useState(false);
  const [saveStatus, setSaveStatus] = useState<string | null>(null);
  const barRef = useRef<HTMLDivElement>(null);

  const canSave = !!(state.schedule && state.envConfig);

  const openFileDialog = () => {
    dispatch({ type: 'OPEN_FILE_DIALOG' });
    setOpenMenu(null);
  };

  const saveFile = async () => {
    setOpenMenu(null);
    if (!state.schedule || !state.envConfig) return;
    if (!state.currentEnvPath || !state.currentSchedulePath) {
      setSaveStatus(UI.savePathUnknownMessage);
      setTimeout(() => setSaveStatus(null), 4000);
      return;
    }
    try {
      await overwriteSaveFiles(state.envConfig, state.schedule, state.currentEnvPath, state.currentSchedulePath);
      dispatch({ type: 'MARK_SAVED' });
      setSaveStatus(UI.savedMessage);
      setTimeout(() => setSaveStatus(null), 2000);
    } catch (err) {
      setSaveStatus(UI.saveFailedMessage(err instanceof Error ? err.message : String(err)));
      setTimeout(() => setSaveStatus(null), 5000);
    }
  };

  const saveFileAs = () => {
    setOpenMenu(null);
    setShowSaveAs(true);
  };

  const exportExcel = async () => {
    setOpenMenu(null);
    if (!state.schedule || !state.envConfig) return;
    try {
      const dates = generateDateRange(state.schedule.planRange.startDate, state.schedule.planRange.endDate);
      const base = state.currentSchedulePath?.split(/[/\\]/).pop()?.replace(/\.ya?ml$/i, '') ?? 'Schedule';
      await exportScheduleToExcel(state.envConfig, state.schedule, dates, `${base}.xlsx`);
      setSaveStatus(UI.excelExportedMessage);
      setTimeout(() => setSaveStatus(null), 2000);
    } catch (err) {
      setSaveStatus(UI.exportFailedMessage(err instanceof Error ? err.message : String(err)));
      setTimeout(() => setSaveStatus(null), 5000);
    }
  };

  const menus: MenuDef[] = [
    {
      id: 'file',
      label: UI.fileMenu,
      items: [
        { label: UI.open, shortcut: 'Ctrl+O', action: openFileDialog, disabled: !!state.session },
        { separator: true },
        { label: UI.save, shortcut: 'Ctrl+S', action: saveFile, disabled: !canSave },
        { label: UI.saveAs, shortcut: 'Ctrl+Shift+S', action: saveFileAs, disabled: !canSave },
        { separator: true },
        { label: UI.exportExcel, action: () => void exportExcel(), disabled: !canSave },
      ],
    },
    { id: 'edit', label: UI.editMenu, items: [] },
    { id: 'view', label: UI.viewMenu, items: [] },
    {
      id: 'collab',
      label: UI.collabMenu,
      items: state.session
        ? [{ label: UI.leaveSessionItem, action: () => { leaveCollabSession(); setOpenMenu(null); } }]
        : [
            { label: UI.startSessionItem, action: () => { dispatch({ type: 'OPEN_SESSION_DIALOG' }); setOpenMenu(null); }, disabled: !canSave },
            { label: UI.joinSessionItem, action: () => { dispatch({ type: 'OPEN_SESSION_DIALOG' }); setOpenMenu(null); } },
          ],
    },
    { id: 'help', label: UI.helpMenu, items: [] },
  ];

  useEffect(() => {
    const handleClick = (e: MouseEvent) => {
      if (barRef.current && !barRef.current.contains(e.target as Node)) {
        setOpenMenu(null);
      }
    };
    document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, []);

  const defaultEnvName = state.currentEnvPath
    ? state.currentEnvPath.split(/[/\\]/).pop() ?? 'EnvConfig.yaml'
    : 'EnvConfig.yaml';
  const defaultScheduleName = state.currentSchedulePath
    ? state.currentSchedulePath.split(/[/\\]/).pop() ?? 'Schedule.yaml'
    : 'Schedule.yaml';

  return (
    <>
      <div
        ref={barRef}
        style={{
          backgroundColor: NAVY,
          display: 'flex',
          alignItems: 'center',
          height: 28,
          paddingLeft: 4,
          flexShrink: 0,
          userSelect: 'none',
        }}
      >
        <span style={{
          color: '#e0e8f0',
          fontSize: 12,
          fontWeight: 'bold',
          fontFamily: 'MS Gothic, monospace',
          paddingLeft: 8,
          paddingRight: 20,
          letterSpacing: 1,
        }}>
          {UI.appTitle}
        </span>

        {menus.map(menu => (
          <div key={menu.id} style={{ position: 'relative' }}>
            <button
              onClick={() => setOpenMenu(openMenu === menu.id ? null : menu.id)}
              style={{
                background: openMenu === menu.id ? NAVY_HOVER : 'transparent',
                border: 'none',
                color: '#e0e8f0',
                padding: '0 10px',
                height: 28,
                cursor: 'pointer',
                fontSize: 12,
                fontFamily: 'MS Gothic, monospace',
                outline: 'none',
              }}
              onMouseEnter={() => openMenu && setOpenMenu(menu.id)}
            >
              {menu.label}
            </button>

            {openMenu === menu.id && menu.items.length > 0 && (
              <div
                style={{
                  position: 'absolute',
                  top: 28,
                  left: 0,
                  width: DROPDOWN_W,
                  backgroundColor: '#fff',
                  border: '1px solid #ccc',
                  boxShadow: '0 4px 12px rgba(0,0,0,0.25)',
                  zIndex: 500,
                  borderRadius: 2,
                  paddingTop: 4,
                  paddingBottom: 4,
                }}
              >
                {menu.items.map((item, idx) =>
                  'separator' in item ? (
                    <div key={idx} style={{ height: 1, backgroundColor: '#e0e0e0', margin: '4px 0' }} />
                  ) : (
                    <div
                      key={idx}
                      onClick={() => !item.disabled && item.action?.()}
                      style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        padding: '5px 16px',
                        fontSize: 12,
                        fontFamily: 'MS Gothic, monospace',
                        color: item.disabled ? '#aaa' : '#222',
                        cursor: item.disabled ? 'default' : 'pointer',
                        backgroundColor: 'transparent',
                      }}
                      onMouseEnter={e => {
                        if (!item.disabled) (e.currentTarget as HTMLDivElement).style.backgroundColor = '#e8eef6';
                      }}
                      onMouseLeave={e => {
                        (e.currentTarget as HTMLDivElement).style.backgroundColor = 'transparent';
                      }}
                    >
                      <span>{item.label}</span>
                      {item.shortcut && (
                        <span style={{ color: '#888', fontSize: 11, marginLeft: 24 }}>{item.shortcut}</span>
                      )}
                    </div>
                  )
                )}
              </div>
            )}
          </div>
        ))}

        {saveStatus && (
          <span style={{ color: '#a8d4f5', fontSize: 11, marginLeft: 16, fontFamily: 'Meiryo, sans-serif' }}>
            {saveStatus}
          </span>
        )}

        {state.session && (
          <span style={{ color: '#a8d4f5', fontSize: 11, marginLeft: 16, fontFamily: 'Meiryo, sans-serif' }}>
            {UI.sessionParticipantsLabel(state.session.participants.length)}
          </span>
        )}
      </div>

      {showSaveAs && state.envConfig && state.schedule && (
        <SaveAsDialog
          envConfig={state.envConfig}
          schedule={state.schedule}
          defaultEnvName={defaultEnvName}
          defaultScheduleName={defaultScheduleName}
          onClose={() => setShowSaveAs(false)}
        />
      )}
    </>
  );
}
