import { useState } from 'react';
import { downloadBothYamlFiles, saveYamlFilesAsElectron } from '../../services/fileService';
import { useAppContext } from '../../context/AppContext';
import { EnvConfig } from '../../types/envConfig';
import { ScheduleData } from '../../types/schedule';

interface Props {
  envConfig: EnvConfig;
  schedule: ScheduleData;
  defaultEnvName: string;
  defaultScheduleName: string;
  onClose: () => void;
}

export function SaveAsDialog({ envConfig, schedule, defaultEnvName, defaultScheduleName, onClose }: Props) {
  const { dispatch } = useAppContext();
  const [envName, setEnvName] = useState(defaultEnvName);
  const [scheduleName, setScheduleName] = useState(defaultScheduleName);

  const handleSave = async () => {
    if (window.electronAPI) {
      const saved = await saveYamlFilesAsElectron(envConfig, schedule, envName, scheduleName);
      if (saved) {
        dispatch({ type: 'SAVE_PATHS', payload: saved });
        onClose();
      }
      // saved === null means the user cancelled a native dialog — keep this modal open.
      return;
    }
    downloadBothYamlFiles(envConfig, schedule, envName, scheduleName);
    onClose();
  };

  return (
    <div
      style={{
        position: 'fixed', inset: 0, zIndex: 1200,
        background: 'rgba(0,0,0,0.4)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
      }}
      onClick={e => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div style={{
        background: '#fff', borderRadius: 6, padding: 24, width: 420,
        boxShadow: '0 8px 32px rgba(0,0,0,0.25)',
        fontFamily: 'Meiryo, sans-serif',
      }}>
        <div style={{ fontWeight: 700, fontSize: 15, marginBottom: 18, color: '#1c2b3a' }}>
          名前を付けて保存
        </div>

        <label style={{ display: 'block', marginBottom: 12 }}>
          <div style={{ fontSize: 12, color: '#455a6b', marginBottom: 4 }}>EnvConfig ファイル名</div>
          <input
            type="text"
            value={envName}
            onChange={e => setEnvName(e.target.value)}
            style={{
              width: '100%', padding: '6px 8px', border: '1px solid #c0d0e0',
              borderRadius: 4, fontSize: 13, boxSizing: 'border-box',
              fontFamily: 'Meiryo, sans-serif',
            }}
          />
        </label>

        <label style={{ display: 'block', marginBottom: 24 }}>
          <div style={{ fontSize: 12, color: '#455a6b', marginBottom: 4 }}>Schedule ファイル名</div>
          <input
            type="text"
            value={scheduleName}
            onChange={e => setScheduleName(e.target.value)}
            style={{
              width: '100%', padding: '6px 8px', border: '1px solid #c0d0e0',
              borderRadius: 4, fontSize: 13, boxSizing: 'border-box',
              fontFamily: 'Meiryo, sans-serif',
            }}
          />
        </label>

        <div style={{ fontSize: 11, color: '#888', marginBottom: 16 }}>
          {window.electronAPI
            ? '保存先はこの後のダイアログで選択します。'
            : '両ファイルはブラウザのダウンロードフォルダに保存されます。'}
        </div>

        <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
          <button
            type="button"
            onClick={onClose}
            style={{
              padding: '6px 18px', border: '1px solid #c0d0e0', borderRadius: 4,
              background: '#f5f8fc', cursor: 'pointer', fontSize: 13,
              fontFamily: 'Meiryo, sans-serif', color: '#333',
            }}
          >
            キャンセル
          </button>
          <button
            type="button"
            onClick={handleSave}
            disabled={!envName.trim() || !scheduleName.trim()}
            style={{
              padding: '6px 18px', border: 'none', borderRadius: 4,
              background: '#1c4f8a', color: '#fff', cursor: 'pointer', fontSize: 13,
              fontFamily: 'Meiryo, sans-serif',
              opacity: (!envName.trim() || !scheduleName.trim()) ? 0.5 : 1,
            }}
          >
            保存
          </button>
        </div>
      </div>
    </div>
  );
}
