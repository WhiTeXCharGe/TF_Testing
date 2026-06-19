import { useState, useEffect, useRef } from 'react';
import { useAppContext } from '../../context/AppContext';
import { downloadScheduleYaml } from '../../services/fileService';
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
  const { state, dispatch } = useAppContext();
  const [openMenu, setOpenMenu] = useState<string | null>(null);
  const barRef = useRef<HTMLDivElement>(null);

  const openFileDialog = () => {
    dispatch({ type: 'OPEN_FILE_DIALOG' });
    setOpenMenu(null);
  };
  const saveFile = () => {
    if (state.schedule) downloadScheduleYaml(state.schedule, state.currentSchedulePath ?? 'Schedule.yaml');
    setOpenMenu(null);
  };
  const saveFileAs = () => {
    if (state.schedule) downloadScheduleYaml(state.schedule, 'Schedule_new.yaml');
    setOpenMenu(null);
  };

  const menus: MenuDef[] = [
    {
      id: 'file',
      label: UI.fileMenu,
      items: [
        { label: UI.open, shortcut: 'Ctrl+O', action: openFileDialog },
        { separator: true },
        { label: UI.save, shortcut: 'Ctrl+S', action: saveFile, disabled: !state.schedule },
        { label: UI.saveAs, shortcut: 'Ctrl+Shift+S', action: saveFileAs, disabled: !state.schedule },
      ],
    },
    { id: 'edit', label: UI.editMenu, items: [] },
    { id: 'view', label: UI.viewMenu, items: [] },
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

  return (
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
      {/* App title */}
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

      {/* Menu items */}
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
    </div>
  );
}
