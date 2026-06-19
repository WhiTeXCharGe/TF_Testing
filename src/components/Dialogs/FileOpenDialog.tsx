import { useState, useRef } from 'react';
import { useAppContext } from '../../context/AppContext';
import { parseEnvConfigYaml, parseScheduleYaml } from '../../services/yamlService';
import { UI } from '../../config/uiText';

export function FileOpenDialog() {
  const { state, dispatch } = useAppContext();
  const [envFile, setEnvFile] = useState<File | null>(null);
  const [schedFile, setSchedFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const envRef = useRef<HTMLInputElement>(null);
  const schedRef = useRef<HTMLInputElement>(null);

  if (!state.isFileOpenDialogOpen) return null;

  const handleCancel = () => {
    setEnvFile(null);
    setSchedFile(null);
    dispatch({ type: 'CLOSE_FILE_DIALOG' });
  };

  const handleOk = async () => {
    if (!envFile || !schedFile) return;
    setLoading(true);
    try {
      const [envText, schedText] = await Promise.all([
        readText(envFile),
        readText(schedFile),
      ]);
      const envConfig = parseEnvConfigYaml(envText);
      const schedule = parseScheduleYaml(schedText);
      dispatch({
        type: 'LOAD_FILES',
        payload: {
          envConfig,
          schedule,
          envPath: envFile.name,
          schedulePath: schedFile.name,
        },
      });
      setEnvFile(null);
      setSchedFile(null);
      dispatch({ type: 'CLOSE_FILE_DIALOG' });
    } catch (err) {
      dispatch({ type: 'SET_ERROR', payload: `ファイル読み込みエラー: ${(err as Error).message}` });
    } finally {
      setLoading(false);
    }
  };

  const canConfirm = !!envFile && !!schedFile && !loading;

  const overlay: React.CSSProperties = {
    position: 'fixed', inset: 0, backgroundColor: 'rgba(0,0,0,0.45)',
    display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 900,
  };
  const modal: React.CSSProperties = {
    backgroundColor: '#fff', borderRadius: 6, width: 420,
    boxShadow: '0 8px 24px rgba(0,0,0,0.35)', fontFamily: 'MS Gothic, monospace',
    overflow: 'hidden',
  };
  const titleBar: React.CSSProperties = {
    backgroundColor: '#1c2b3a', color: '#fff', padding: '10px 16px',
    fontSize: 13, fontWeight: 'bold',
  };
  const body: React.CSSProperties = { padding: '20px 20px 16px' };
  const fieldRow: React.CSSProperties = { marginBottom: 16 };
  const labelStyle: React.CSSProperties = { display: 'block', fontSize: 12, color: '#444', marginBottom: 6, fontWeight: 'bold' };
  const fileRow: React.CSSProperties = { display: 'flex', alignItems: 'center', gap: 10 };
  const chooseBtn: React.CSSProperties = {
    padding: '4px 12px', border: '1px solid #aaa', borderRadius: 3,
    cursor: 'pointer', fontSize: 12, backgroundColor: '#f5f5f5', whiteSpace: 'nowrap',
    fontFamily: 'MS Gothic, monospace',
  };
  const fileName: React.CSSProperties = {
    fontSize: 12, color: '#555', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
  };
  const footer: React.CSSProperties = {
    display: 'flex', justifyContent: 'flex-end', gap: 8, padding: '12px 20px',
    borderTop: '1px solid #e0e0e0', backgroundColor: '#fafafa',
  };
  const okBtn: React.CSSProperties = {
    padding: '6px 20px', backgroundColor: canConfirm ? '#1976d2' : '#aaa',
    color: '#fff', border: 'none', borderRadius: 4,
    cursor: canConfirm ? 'pointer' : 'default', fontSize: 12,
    fontFamily: 'MS Gothic, monospace',
  };
  const cancelBtn: React.CSSProperties = {
    padding: '6px 16px', border: '1px solid #aaa', borderRadius: 4,
    cursor: 'pointer', fontSize: 12, backgroundColor: '#fff', fontFamily: 'MS Gothic, monospace',
  };

  return (
    <div style={overlay} onClick={e => { if (e.target === e.currentTarget) handleCancel(); }}>
      <div style={modal}>
        <div style={titleBar}>{UI.fileOpenDialogTitle}</div>
        <div style={body}>
          {/* EnvConfig */}
          <div style={fieldRow}>
            <label style={labelStyle}>{UI.envConfigFileLabel}</label>
            <div style={fileRow}>
              <button style={chooseBtn} onClick={() => envRef.current?.click()}>
                {UI.chooseFile}
              </button>
              <span style={fileName}>{envFile ? envFile.name : UI.noFileChosen}</span>
              <input
                ref={envRef}
                type="file"
                accept=".yaml,.yml"
                style={{ display: 'none' }}
                onChange={e => setEnvFile(e.target.files?.[0] ?? null)}
              />
            </div>
          </div>
          {/* Schedule */}
          <div style={fieldRow}>
            <label style={labelStyle}>{UI.scheduleFileLabel}</label>
            <div style={fileRow}>
              <button style={chooseBtn} onClick={() => schedRef.current?.click()}>
                {UI.chooseFile}
              </button>
              <span style={fileName}>{schedFile ? schedFile.name : UI.noFileChosen}</span>
              <input
                ref={schedRef}
                type="file"
                accept=".yaml,.yml"
                style={{ display: 'none' }}
                onChange={e => setSchedFile(e.target.files?.[0] ?? null)}
              />
            </div>
          </div>
        </div>
        <div style={footer}>
          <button style={okBtn} onClick={handleOk} disabled={!canConfirm}>
            {loading ? '読み込み中...' : UI.dialogOk}
          </button>
          <button style={cancelBtn} onClick={handleCancel}>{UI.dialogCancel}</button>
        </div>
      </div>
    </div>
  );
}

function readText(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = () => reject(new Error(`読み込み失敗: ${file.name}`));
    reader.readAsText(file, 'utf-8');
  });
}
