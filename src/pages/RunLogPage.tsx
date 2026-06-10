import React, { useState, useEffect, useRef } from 'react';
import { saveAs } from 'file-saver';
import { UI } from '@/config/uiConfig';
import { Icon } from '@/components/common/Icon';
import { Dialog } from '@/components/common/Dialog';
import { useRuns } from '@/hooks/useRuns';
import { nowLabel } from '@/utils/dateUtils';
import type { Run } from '@/types';
import type { RunStatus } from '@/services/runService';

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

  const canSubmit = !!envFile && !!schedFile && !uploading;

  async function submit() {
    if (!canSubmit || !envFile || !schedFile) return;
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
      <div className="modal" style={{ width: 540 }}>
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

          <DropZoneField
            file={envFile}
            onFile={setEnvFile}
            label={UI.runLog.envLabel}
            hint={UI.runLog.envHint}
            originalPath={originalEnvPath}
            onOriginalPath={setOriginalEnvPath}
            disabled={uploading}
          />

          <DropZoneField
            file={schedFile}
            onFile={setSchedFile}
            label={UI.runLog.schedLabel}
            hint={UI.runLog.schedHint}
            originalPath={originalSchedPath}
            onOriginalPath={setOriginalSchedPath}
            disabled={uploading}
          />

          {error && (
            <div style={{
              background: 'var(--red-lt)', color: 'var(--red)',
              border: '1px solid var(--red)', borderRadius: 5,
              padding: '8px 12px', fontSize: 12, marginTop: 8,
              fontFamily: 'var(--font-mono)',
            }}>
              {UI.runLog.uploadError} {error}
            </div>
          )}
        </div>
        <div className="modal-footer">
          <button className="btn btn-secondary btn-sm" onClick={onClose} disabled={uploading}>
            {UI.runLog.cancel}
          </button>
          <button className="btn btn-primary btn-sm" disabled={!canSubmit} onClick={submit}>
            <Icon name="play" size={13} />
            {uploading ? UI.runLog.uploading : UI.runLog.submit}
          </button>
        </div>
      </div>
    </div>
  );
}

// ── Drop-zone field: drag-drop OR click to pick + optional path text ────────
function DropZoneField({
  file, onFile, label, hint, originalPath, onOriginalPath, disabled,
}: {
  file: File | null;
  onFile: (f: File | null) => void;
  label: string;
  hint: string;
  originalPath: string;
  onOriginalPath: (s: string) => void;
  disabled: boolean;
}) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement | null>(null);

  function pickFirstYaml(list: FileList | null): File | null {
    if (!list || list.length === 0) return null;
    // Prefer a .yaml/.yml, but accept anything if that's what was dropped.
    for (let i = 0; i < list.length; i++) {
      const f = list.item(i);
      if (!f) continue;
      if (/\.ya?ml$/i.test(f.name)) return f;
    }
    return list.item(0);
  }

  function onDragOver(e: React.DragEvent) {
    if (disabled) return;
    e.preventDefault();
    e.stopPropagation();
    if (e.dataTransfer) e.dataTransfer.dropEffect = 'copy';
    if (!dragging) setDragging(true);
  }
  function onDragLeave(e: React.DragEvent) {
    e.preventDefault();
    e.stopPropagation();
    setDragging(false);
  }
  function onDrop(e: React.DragEvent) {
    if (disabled) return;
    e.preventDefault();
    e.stopPropagation();
    setDragging(false);
    const f = pickFirstYaml(e.dataTransfer?.files ?? null);
    if (f) onFile(f);
  }

  return (
    <div className="form-group">
      <label className="form-label">{label}</label>
      <div
        className={
          'dropzone' +
          (file ? ' has-file' : '') +
          (dragging ? ' is-dragging' : '') +
          (disabled ? ' is-disabled' : '')
        }
        onDragOver={onDragOver}
        onDragEnter={onDragOver}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
        onClick={() => !disabled && inputRef.current?.click()}
        role="button"
        tabIndex={0}
      >
        <div className="dropzone-line-1">
          {file
            ? <>✓ {file.name}</>
            : <>⬆ {dragging ? UI.runLog.dropHintActive : UI.runLog.dropHint}</>}
        </div>
        <div className="dropzone-line-2">
          {file
            ? `${Math.max(1, Math.round(file.size / 1024))} KB · ${UI.runLog.fileSelected}`
            : hint}
        </div>
        <input
          ref={inputRef}
          type="file"
          accept=".yaml,.yml"
          style={{ display: 'none' }}
          onChange={e => onFile(e.target.files?.[0] ?? null)}
        />
      </div>
      <input
        className="form-input"
        style={{ marginTop: 6, fontFamily: 'var(--font-mono)', fontSize: 11 }}
        placeholder={UI.runLog.originalPathPlaceholder}
        value={originalPath}
        onChange={e => onOriginalPath(e.target.value)}
        disabled={disabled}
        onClick={e => e.stopPropagation()}
      />
      <div style={{ fontSize: 10, color: 'var(--text-sec)', marginTop: 2 }}>
        {UI.runLog.originalPathLabel}
      </div>
    </div>
  );
}

// ── Solver status dialog state ────────────────────────────────────────────────
interface SolverStatusDialog {
  run:    Run;
  status: RunStatus;
}
interface SolverErrorDialog {
  run:     Run;
  message: string;
}

// ── Main page ───────────────────────────────────────────────────────────────
export function RunLogPage() {
  const {
    runs, loading, error,
    submitNewRun, checkOutput, removeRun,
    solverEnabled, submitToSolver, checkRunStatus, triggerDownload,
  } = useRuns();
  const [popup, setPopup] = useState<PopupState | null>(null);
  const [modalOpen, setModalOpen] = useState(false);
  const [busy, setBusy] = useState<Set<string>>(new Set());

  // Dialog state — only one of these is open at a time.
  const [ganttDialog,    setGanttDialog]    = useState<{ run: Run; source: 'copy' | 'result' } | null>(null);
  const [notReady,       setNotReady]       = useState<Run | null>(null);
  const [solverStatus,   setSolverStatus]   = useState<SolverStatusDialog | null>(null);
  const [solverError,    setSolverError]    = useState<SolverErrorDialog | null>(null);
  const [confirmDel,     setConfirmDel]     = useState<Run | null>(null);

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
    // 1. Save files locally so the run appears in the list.
    const run = await submitNewRun(p);

    // 2. If a backend is configured, also send the files to the solver.
    if (solverEnabled) {
      try {
        await submitToSolver(run.id, p.envFile, p.schedFile);
      } catch (err) {
        // Non-fatal: local row was already created; show a warning but close the modal.
        setSolverError({ run, message: String((err as Error).message || err) });
      }
    }

    setModalOpen(false);
  }

  async function handleShowResult(run: Run) {
    setBusy(prev => new Set(prev).add(run.id));
    try {
      if (solverEnabled) {
        // ── Solver API mode ─────────────────────────────────────────────────
        // GET /status/:runId → check if solve is done
        const status = await checkRunStatus(run.id);

        if (status.status === 'Completed') {
          // GET /download/:runId → receive YAML blob → save to disk
          const { blob, filename } = await triggerDownload(run.id);
          saveAs(blob, filename);
        } else {
          // Still running, submitted, or failed — show status dialog
          setSolverStatus({ run, status });
        }
      } else {
        // ── Local-only mode ─────────────────────────────────────────────────
        // Check whether the output YAML landed in public/local/<id>/output/
        const out = await checkOutput(run.id);
        if (out.hasYaml) {
          setGanttDialog({ run, source: 'result' });
        } else {
          setNotReady(run);
        }
      }
    } catch (err) {
      setSolverError({ run, message: String((err as Error).message || err) });
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
                    {(() => {
                      // If the output yaml exists, the run is finished → "Delete".
                      // If not, the run is either in-progress or never started → "Cancel".
                      // Both buttons do the same thing locally (remove row + folder).
                      // On the Azure side, Cancel will also terminate the running Batch task.
                      const finished = run.outputHasYaml;
                      const label = finished ? UI.runLog.deleteBtn : UI.runLog.cancelBtn;
                      return (
                        <button
                          className="btn btn-danger btn-xs"
                          onClick={() => setConfirmDel(run)}
                          title={label}
                        >
                          <Icon name="x" size={12} />{label}
                        </button>
                      );
                    })()}
                  </td>
                </tr>
              ))}
              {loading && runs.length === 0 && (
                <tr>
                  <td colSpan={4} style={{ textAlign: 'center', padding: 24, color: 'var(--text-sec)' }}>
                    Loading runs.json…
                  </td>
                </tr>
              )}
              {error && (
                <tr>
                  <td colSpan={4} style={{ textAlign: 'center', padding: 16, color: 'var(--red)' }}>
                    Failed to load runs.json — {error}
                  </td>
                </tr>
              )}
              {!loading && !error && runs.length === 0 && (
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

      {/* Solver status dialog — shown when status is Running or Submitted */}
      {solverStatus && (
        <Dialog
          title={
            solverStatus.status.status === 'Failed'
              ? UI.runLog.solverFailedTitle
              : UI.runLog.solverStatusTitle
          }
          body={
            <>
              {solverStatus.status.status === 'Failed' ? (
                <>
                  <div>{UI.runLog.solverFailedLabel}</div>
                  <div style={{ marginTop: 6, fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--red, #c0392b)' }}>
                    {solverStatus.status.error ?? UI.runLog.solverFailedUnknown}
                  </div>
                </>
              ) : (
                <div>
                  {solverStatus.status.status === 'Submitted'
                    ? UI.runLog.solverStatusSubmitted
                    : UI.runLog.solverStatusRunning}
                  {solverStatus.status.stage != null && (
                    <span style={{ marginLeft: 6, opacity: 0.7 }}>
                      (Stage {solverStatus.status.stage}, {Math.round((solverStatus.status.progress ?? 0) * 100)}%)
                    </span>
                  )}
                </div>
              )}
              <div style={{ marginTop: 10, fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text)' }}>
                Run id: {solverStatus.run.id}
              </div>
            </>
          }
          buttons={[{ label: UI.runLog.solverStatusClose, onClick: () => setSolverStatus(null), variant: 'primary' }]}
          onClose={() => setSolverStatus(null)}
        />
      )}

      {/* Solver error dialog — shown when the network call itself fails */}
      {solverError && (
        <Dialog
          title={UI.runLog.solverErrorTitle}
          body={
            <>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--red, #c0392b)', wordBreak: 'break-word' }}>
                {solverError.message}
              </div>
              <div style={{ marginTop: 10, fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text)' }}>
                Run id: {solverError.run.id}
              </div>
            </>
          }
          buttons={[{ label: UI.runLog.solverStatusClose, onClick: () => setSolverError(null), variant: 'primary' }]}
          onClose={() => setSolverError(null)}
        />
      )}

      {confirmDel && (() => {
        // Choose Delete vs Cancel wording based on whether the run has finished.
        const finished = confirmDel.outputHasYaml;
        const title = finished ? UI.runLog.deleteConfirmTitle : UI.runLog.cancelConfirmTitle;
        const body  = finished ? UI.runLog.deleteConfirmBody  : UI.runLog.cancelConfirmBody;
        const yes   = finished ? UI.runLog.deleteConfirmYes   : UI.runLog.cancelConfirmYes;
        const no    = finished ? UI.runLog.deleteConfirmNo    : UI.runLog.cancelConfirmNo;
        return (
          <Dialog
            title={title}
            body={
              <>
                {body}
                <div style={{ marginTop: 10, fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text)' }}>
                  Run id: {confirmDel.id}
                </div>
              </>
            }
            buttons={[
              { label: no,  onClick: () => setConfirmDel(null), variant: 'secondary' },
              { label: yes, onClick: handleConfirmDelete,       variant: 'danger' },
            ]}
            onClose={() => setConfirmDel(null)}
          />
        );
      })()}
    </div>
  );
}
