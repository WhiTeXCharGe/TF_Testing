import React, { useState, useEffect } from 'react';
import { UI } from '@/config/uiConfig';
import { Icon } from '@/components/common/Icon';
import { Dialog } from '@/components/common/Dialog';
import { useRuns } from '@/hooks/useRuns';
import { nowLabel } from '@/utils/dateUtils';
import type { Run } from '@/types';

// ── Path tooltip popup ──────────────────────────────────────────────────────
interface PopupState {
  lines: { label: string; value: string }[];
  x: number;
  y: number;
}

function PathPopup({ state, onClose }: { state: PopupState | null; onClose: () => void }) {
  useEffect(() => {
    if (!state) return;
    const t = setTimeout(onClose, 4500);
    const onDocClick = (e: MouseEvent) => {
      if (!(e.target as HTMLElement).closest('.path-popup')) onClose();
    };
    document.addEventListener('click', onDocClick);
    return () => { clearTimeout(t); document.removeEventListener('click', onDocClick); };
  }, [state, onClose]);

  if (!state) return null;
  const left = Math.min(state.x, window.innerWidth - 440);
  return (
    <div className="path-popup" style={{ top: state.y, left }}>
      {state.lines.map((ln, i) => (
        <div key={i} style={{ marginBottom: i === state.lines.length - 1 ? 0 : 6 }}>
          <div style={{ fontSize: 9, opacity: .65, textTransform: 'uppercase', letterSpacing: .4 }}>
            {ln.label}
          </div>
          <div>{ln.value || UI.runLog.pathUnknown}</div>
        </div>
      ))}
    </div>
  );
}

// ── File chip ───────────────────────────────────────────────────────────────
function FileChip({ label, onClick, onHover }: {
  label: string;
  onClick: (e: React.MouseEvent) => void;
  onHover: (e: React.MouseEvent) => void;
}) {
  return (
    <span
      className="file-chip file-chip-clickable"
      onClick={onClick}
      onMouseEnter={onHover}
    >
      <Icon name="comment" size={11} />
      {label}
    </span>
  );
}

// ── Input Files cell ──────────────────────────────────────────────────────────
function InputCell({ run, onShowPaths, onShowCopy }: {
  run: Run;
  onShowPaths: (e: React.MouseEvent, which: 'env' | 'sched') => void;
  onShowCopy: () => void;
}) {
  return (
    <div className="input-cell">
      <div className="input-files-label">
        <FileChip
          label="EnvConfig"
          onClick={e => onShowPaths(e, 'env')}
          onHover={e => onShowPaths(e, 'env')}
        />
        <FileChip
          label="Schedule"
          onClick={e => onShowPaths(e, 'sched')}
          onHover={e => onShowPaths(e, 'sched')}
        />
      </div>
      <button className="btn btn-outline btn-xs" onClick={onShowCopy}>
        <Icon name="folder" size={12} />{UI.runLog.showCopyBtn}
      </button>
    </div>
  );
}

// ── Result cell ─────────────────────────────────────────────────────────────
function ResultCell({ run, busy, onShowResult }: {
  run: Run;
  busy: boolean;
  onShowResult: () => void;
}) {
  const outputPath = run.savedOutputPath ?? '';
  return (
    <div className="result-cell-v2">
      <button className="btn btn-primary btn-xs" onClick={onShowResult} disabled={busy}>
        <Icon name="download" size={12} />{busy ? UI.runLog.fetching : UI.runLog.showResultBtn}
      </button>
      <div className="output-path-box" title={outputPath || UI.runLog.outputEmpty}>
        {outputPath || <span className="output-path-empty">{UI.runLog.outputEmpty}</span>}
      </div>
    </div>
  );
}

// ── New Run modal ─────────────────────────────────────────────────────────────
function NewRunModal({ onClose, onSubmit }: {
  onClose: () => void;
  onSubmit: (payload: {
    envFile: File;
    schedFile: File;
    originalEnvPath: string;
    originalSchedPath: string;
  }) => Promise<void>;
}) {
  const [envFile,   setEnvFile]   = useState<File | null>(null);
  const [schedFile, setSchedFile] = useState<File | null>(null);
  const [originalEnvPath,   setOriginalEnvPath]   = useState('');
  const [originalSchedPath, setOriginalSchedPath] = useState('');
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit() {
    if (!envFile || !schedFile) return;
    setUploading(true);
    setError(null);
    try {
      await onSubmit({ envFile, schedFile, originalEnvPath, originalSchedPath });
    } catch (e) {
      setError(String((e as Error).message || e));
      setUploading(false);
    }
  }

  return (
    <div className="modal-overlay" onClick={e => { if (e.target === e.currentTarget && !uploading) onClose(); }}>
      <div className="modal" style={{ width: 520 }}>
        <div className="modal-hd">
          <div className="modal-title">{UI.runLog.modalTitle}</div>
          <button className="modal-close" onClick={onClose} disabled={uploading}>
            <Icon name="x" size={16} />
          </button>
        </div>
        <div className="modal-body">
          <p style={{ fontSize: 'var(--fs-sm)', color: 'var(--text-sec)', marginBottom: 16 }}>
            {UI.runLog.modalHint}
          </p>

          {/* ─── EnvConfig ─── */}
          <div className="form-group">
            <label className="form-label">{UI.runLog.envLabel}</label>
            <label className={'upzone' + (envFile ? ' has-file' : '')}>
              <div>{envFile ? `✓ ${envFile.name}` : `⬆ ${UI.runLog.selectFile} ${UI.runLog.envLabel}`}</div>
              <div style={{ fontSize: 10, marginTop: 2 }}>{envFile ? UI.runLog.fileSelected : UI.runLog.envHint}</div>
              <input type="file" accept=".yaml,.yml"
                     onChange={e => setEnvFile(e.target.files?.[0] ?? null)} />
            </label>
            <input
              className="form-input"
              style={{ marginTop: 6, fontSize: 11, fontFamily: 'var(--font-mono)' }}
              placeholder={UI.runLog.originalPathPlaceholder}
              value={originalEnvPath}
              onChange={e => setOriginalEnvPath(e.target.value)}
            />
            <div style={{ fontSize: 10, color: 'var(--text-sec)', marginTop: 2 }}>
              {UI.runLog.originalPathLabel}
            </div>
          </div>

          {/* ─── Schedule ─── */}
          <div className="form-group">
            <label className="form-label">{UI.runLog.schedLabel}</label>
            <label className={'upzone' + (schedFile ? ' has-file' : '')}>
              <div>{schedFile ? `✓ ${schedFile.name}` : `⬆ ${UI.runLog.selectFile} ${UI.runLog.schedLabel}`}</div>
              <div style={{ fontSize: 10, marginTop: 2 }}>{schedFile ? UI.runLog.fileSelected : UI.runLog.schedHint}</div>
              <input type="file" accept=".yaml,.yml"
                     onChange={e => setSchedFile(e.target.files?.[0] ?? null)} />
            </label>
            <input
              className="form-input"
              style={{ marginTop: 6, fontSize: 11, fontFamily: 'var(--font-mono)' }}
              placeholder={UI.runLog.originalPathPlaceholder}
              value={originalSchedPath}
              onChange={e => setOriginalSchedPath(e.target.value)}
            />
            <div style={{ fontSize: 10, color: 'var(--text-sec)', marginTop: 2 }}>
              {UI.runLog.originalPathLabel}
            </div>
          </div>

          {error && (
            <div style={{
              background: 'var(--red-lt)', color: 'var(--red)',
              border: '1px solid var(--red)', borderRadius: 5,
              padding: '8px 12px', fontSize: 12, marginTop: 8,
            }}>
              {UI.runLog.uploadError} {error}
            </div>
          )}
        </div>
        <div className="modal-footer">
          <button className="btn btn-secondary btn-sm" onClick={onClose} disabled={uploading}>
            {UI.runLog.cancel}
          </button>
          <button
            className="btn btn-primary btn-sm"
            disabled={!envFile || !schedFile || uploading}
            onClick={submit}
          >
            <Icon name="play" size={13} />
            {uploading ? UI.runLog.uploading : UI.runLog.submit}
          </button>
        </div>
      </div>
    </div>
  );
}

// ── Main page ───────────────────────────────────────────────────────────────
export function RunLogPage() {
  const { runs, submitNewRun, checkOutput, removeRun, refresh } = useRuns();
  const [popup, setPopup] = useState<PopupState | null>(null);
  const [modalOpen, setModalOpen] = useState(false);
  const [busy, setBusy] = useState<Set<string>>(new Set());

  // Dialog state — only one of these is open at a time.
  const [ganttDialog,  setGanttDialog]  = useState<{ run: Run; source: 'copy' | 'result' } | null>(null);
  const [notReady,     setNotReady]     = useState<Run | null>(null);
  const [confirmDel,   setConfirmDel]   = useState<Run | null>(null);

  function showPaths(e: React.MouseEvent, run: Run, which: 'env' | 'sched') {
    e.stopPropagation();
    const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
    const lines = which === 'env'
      ? [
          { label: UI.runLog.originalEnvLabel, value: run.originalEnvPath ?? '' },
          { label: UI.runLog.savedEnvLabel,    value: run.savedEnvPath    ?? '' },
        ]
      : [
          { label: UI.runLog.originalSchedLabel, value: run.originalSchedPath ?? '' },
          { label: UI.runLog.savedSchedLabel,    value: run.savedSchedPath    ?? '' },
        ];
    setPopup({ lines, x: rect.left, y: rect.bottom + 6 });
  }

  async function handleSubmit(p: {
    envFile: File; schedFile: File;
    originalEnvPath: string; originalSchedPath: string;
  }) {
    await submitNewRun(p);
    setModalOpen(false);
  }

  async function handleShowResult(run: Run) {
    setBusy(prev => new Set(prev).add(run.id));
    try {
      const out = await checkOutput(run.id);
      if (out.hasYaml) {
        // We have a local yaml — open the gantt editor placeholder.
        // (Also refresh the row so savedOutputPath is populated.)
        refresh();
        setGanttDialog({ run, source: 'result' });
      } else {
        // No local yaml. In future: try Azure Blob with the runId.
        // For now Azure is not connected → show "not ready" dialog.
        setNotReady(run);
      }
    } finally {
      setBusy(prev => { const n = new Set(prev); n.delete(run.id); return n; });
    }
  }

  async function handleConfirmDelete() {
    if (!confirmDel) return;
    await removeRun(confirmDel.id);
    setConfirmDel(null);
  }

  return (
    <div>
      <div className="page-header" style={{ display: 'flex', alignItems: 'flex-start' }}>
        <div style={{ flex: 1 }}>
          <h1 className="page-heading">{UI.runLog.heading}</h1>
          <p className="page-subheading">{UI.runLog.subheading}</p>
        </div>
        <button className="btn btn-primary" onClick={() => setModalOpen(true)}>
          <Icon name="plus" size={16} />{UI.runLog.newRun}
        </button>
      </div>

      <div className="card" style={{ padding: 0 }}>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th style={{ width: 160 }}>{UI.runLog.columnDate}</th>
                <th>{UI.runLog.columnInput}</th>
                <th style={{ width: 320 }}>{UI.runLog.columnResult}</th>
                <th style={{ width: 80 }}></th>
              </tr>
            </thead>
            <tbody>
              {runs.map(run => (
                <tr key={run.id}>
                  <td style={{ whiteSpace: 'nowrap' }}>
                    <div className="solve-date">{nowLabel(run.solveDate)}</div>
                    <div className="solve-folder">{run.folderPath}</div>
                  </td>
                  <td>
                    <InputCell
                      run={run}
                      onShowPaths={(e, which) => showPaths(e, run, which)}
                      onShowCopy={() => setGanttDialog({ run, source: 'copy' })}
                    />
                  </td>
                  <td>
                    <ResultCell
                      run={run}
                      busy={busy.has(run.id)}
                      onShowResult={() => handleShowResult(run)}
                    />
                  </td>
                  <td>
                    <button
                      className="btn btn-danger btn-xs"
                      onClick={() => setConfirmDel(run)}
                      title={UI.runLog.deleteBtn}
                    >
                      <Icon name="x" size={12} />{UI.runLog.deleteBtn}
                    </button>
                  </td>
                </tr>
              ))}
              {runs.length === 0 && (
                <tr>
                  <td colSpan={4} style={{ textAlign: 'center', padding: 24, color: 'var(--text-sec)' }}>
                    No runs yet. Click <strong>New Run</strong> to upload inputs.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      <PathPopup state={popup} onClose={() => setPopup(null)} />

      {modalOpen && <NewRunModal onClose={() => setModalOpen(false)} onSubmit={handleSubmit} />}

      {ganttDialog && (
        <Dialog
          title={UI.runLog.ganttDialogTitle}
          body={
            <>
              {UI.runLog.ganttDialogBody}
              <div style={{ marginTop: 10, fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text)' }}>
                Run id: {ganttDialog.run.id}<br />
                {ganttDialog.source === 'result'
                  ? <>Output: {ganttDialog.run.savedOutputPath ?? '(unknown)'}</>
                  : <>Input dir: {ganttDialog.run.inputDir}</>}
              </div>
            </>
          }
          buttons={[{ label: UI.runLog.ganttDialogClose, onClick: () => setGanttDialog(null), variant: 'primary' }]}
          onClose={() => setGanttDialog(null)}
        />
      )}

      {notReady && (
        <Dialog
          title={UI.runLog.notReadyTitle}
          body={
            <>
              {UI.runLog.notReadyBody}
              <div style={{ marginTop: 10, fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text)' }}>
                Run id: {notReady.id}
              </div>
            </>
          }
          buttons={[{ label: UI.runLog.notReadyClose, onClick: () => setNotReady(null), variant: 'primary' }]}
          onClose={() => setNotReady(null)}
        />
      )}

      {confirmDel && (
        <Dialog
          title={UI.runLog.deleteConfirmTitle}
          body={
            <>
              {UI.runLog.deleteConfirmBody}
              <div style={{ marginTop: 10, fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text)' }}>
                Run id: {confirmDel.id}
              </div>
            </>
          }
          buttons={[
            { label: UI.runLog.deleteConfirmNo,  onClick: () => setConfirmDel(null), variant: 'secondary' },
            { label: UI.runLog.deleteConfirmYes, onClick: handleConfirmDelete,       variant: 'danger' },
          ]}
          onClose={() => setConfirmDel(null)}
        />
      )}
    </div>
  );
}
