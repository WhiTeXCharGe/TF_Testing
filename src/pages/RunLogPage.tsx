import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { UI } from '@/config/uiConfig';
import { Icon } from '@/components/common/Icon';
import { useRuns } from '@/hooks/useRuns';
import { nowLabel } from '@/utils/dateUtils';
import type { Run } from '@/types';

// ── Path tooltip popup ──────────────────────────────────────────────────────
interface PopupState { text: string; x: number; y: number; }

function PathPopup({ state, onClose }: { state: PopupState | null; onClose: () => void }) {
  useEffect(() => {
    if (!state) return;
    const t = setTimeout(onClose, 3000);
    const onDocClick = (e: MouseEvent) => {
      if (!(e.target as HTMLElement).closest('.path-popup')) onClose();
    };
    document.addEventListener('click', onDocClick);
    return () => { clearTimeout(t); document.removeEventListener('click', onDocClick); };
  }, [state, onClose]);

  if (!state) return null;
  const left = Math.min(state.x, window.innerWidth - 430);
  return (
    <div className="path-popup" style={{ top: state.y, left }}>
      {`Local path:\n${state.text}`}
    </div>
  );
}

// ── File chip ───────────────────────────────────────────────────────────────
function FileChip({ label }: { label: string }) {
  return (
    <span className="file-chip">
      <Icon name="comment" size={11} />
      {label}
    </span>
  );
}

// ── Input Files cell ──────────────────────────────────────────────────────────
function InputCell({ run, onPath, onGantt }: {
  run: Run;
  onPath: (e: React.MouseEvent, text: string) => void;
  onGantt: () => void;
}) {
  const paths = `${run.folderPath}input/${run.inputEnvName}\n${run.folderPath}input/${run.inputSchedName}`;
  return (
    <div className="input-cell">
      <div className="input-files-label">
        <FileChip label="EnvConfig" />
        <FileChip label="Schedule" />
      </div>
      <button className="btn btn-ghost btn-xs" onClick={e => onPath(e, paths)}>
        <Icon name="folder" size={12} />{UI.runLog.pathBtn}
      </button>
      <button className="btn btn-outline btn-xs" onClick={onGantt}>
        <Icon name="gantt" size={12} />{UI.runLog.ganttBtn}
      </button>
    </div>
  );
}

// ── Result cell ───────────────────────────────────────────────────────────────
function ResultCell({ run, fetching, onFetch, onPath, onGantt }: {
  run: Run;
  fetching: boolean;
  onFetch: () => void;
  onPath: (e: React.MouseEvent, text: string) => void;
  onGantt: () => void;
}) {
  if (run.output === 'ready') {
    return (
      <div className="result-cell">
        <button className="btn btn-ghost btn-xs" onClick={e => onPath(e, `${run.folderPath}output/`)}>
          <Icon name="folder" size={12} />{UI.runLog.outputBtn}
        </button>
        <button className="btn btn-outline btn-xs" onClick={onGantt} disabled={!run.outputHasYaml}>
          <Icon name="gantt" size={12} />{UI.runLog.ganttBtn}
        </button>
        <span className="ready-tag"><Icon name="check" size={11} />{UI.runLog.filesReady}</span>
      </div>
    );
  }
  // output === 'none' (or fetching)
  return (
    <div className="result-cell">
      <button className="btn btn-primary btn-xs" onClick={onFetch} disabled={fetching}>
        <Icon name="download" size={12} />{fetching ? UI.runLog.fetching : UI.runLog.fetchBtn}
      </button>
      <button className="btn btn-ghost btn-xs" disabled>
        <Icon name="gantt" size={12} />{UI.runLog.ganttBtn}
      </button>
    </div>
  );
}

// ── New Run modal ─────────────────────────────────────────────────────────────
function NewRunModal({ onClose, onSubmit }: {
  onClose: () => void;
  onSubmit: (envName: string, schedName: string) => void;
}) {
  const [envName, setEnvName]     = useState<string | null>(null);
  const [schedName, setSchedName] = useState<string | null>(null);

  return (
    <div className="modal-overlay" onClick={e => { if (e.target === e.currentTarget) onClose(); }}>
      <div className="modal">
        <div className="modal-hd">
          <div className="modal-title">{UI.runLog.modalTitle}</div>
          <button className="modal-close" onClick={onClose}><Icon name="x" size={16} /></button>
        </div>
        <div className="modal-body">
          <p style={{ fontSize: 'var(--fs-sm)', color: 'var(--text-sec)', marginBottom: 16 }}>
            {UI.runLog.modalHint}
          </p>

          <div className="form-group">
            <label className="form-label">{UI.runLog.envLabel}</label>
            <label className={'upzone' + (envName ? ' has-file' : '')}>
              <div>{envName ? `✓ ${envName}` : `⬆ ${UI.runLog.selectFile} ${UI.runLog.envLabel}`}</div>
              <div style={{ fontSize: 10, marginTop: 2 }}>{envName ? UI.runLog.fileSelected : UI.runLog.envHint}</div>
              <input type="file" accept=".yaml,.yml"
                     onChange={e => setEnvName(e.target.files?.[0]?.name ?? 'EnvConfig.yaml')} />
            </label>
          </div>

          <div className="form-group">
            <label className="form-label">{UI.runLog.schedLabel}</label>
            <label className={'upzone' + (schedName ? ' has-file' : '')}>
              <div>{schedName ? `✓ ${schedName}` : `⬆ ${UI.runLog.selectFile} ${UI.runLog.schedLabel}`}</div>
              <div style={{ fontSize: 10, marginTop: 2 }}>{schedName ? UI.runLog.fileSelected : UI.runLog.schedHint}</div>
              <input type="file" accept=".yaml,.yml"
                     onChange={e => setSchedName(e.target.files?.[0]?.name ?? 'Schedule.yaml')} />
            </label>
          </div>
        </div>
        <div className="modal-footer">
          <button className="btn btn-secondary btn-sm" onClick={onClose}>{UI.runLog.cancel}</button>
          <button
            className="btn btn-primary btn-sm"
            disabled={!envName || !schedName}
            onClick={() => onSubmit(envName ?? 'EnvConfig.yaml', schedName ?? 'Schedule.yaml')}
          >
            <Icon name="play" size={13} />{UI.runLog.submit}
          </button>
        </div>
      </div>
    </div>
  );
}

// ── Main page ───────────────────────────────────────────────────────────────
export function RunLogPage() {
  const navigate = useNavigate();
  const { runs, addRun, fetchOutput } = useRuns();
  const [popup, setPopup] = useState<PopupState | null>(null);
  const [modalOpen, setModalOpen] = useState(false);
  const [fetching, setFetching] = useState<Set<string>>(new Set());

  function showPath(e: React.MouseEvent, text: string) {
    e.stopPropagation();
    const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
    setPopup({ text, x: rect.left, y: rect.bottom + 6 });
  }

  async function handleFetch(id: string) {
    setFetching(prev => new Set(prev).add(id));
    await fetchOutput(id);
    setFetching(prev => { const n = new Set(prev); n.delete(id); return n; });
  }

  function handleSubmit(envName: string, schedName: string) {
    addRun({ envName, schedName });
    setModalOpen(false);
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
                <th style={{ width: 230 }}>{UI.runLog.columnResult}</th>
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
                      onPath={showPath}
                      onGantt={() => navigate(`/gantt/${run.id}`)}
                    />
                  </td>
                  <td>
                    <ResultCell
                      run={run}
                      fetching={fetching.has(run.id)}
                      onFetch={() => handleFetch(run.id)}
                      onPath={showPath}
                      onGantt={() => navigate(`/gantt/${run.id}/result`)}
                    />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <PathPopup state={popup} onClose={() => setPopup(null)} />
      {modalOpen && <NewRunModal onClose={() => setModalOpen(false)} onSubmit={handleSubmit} />}
    </div>
  );
}
