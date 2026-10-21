/**
 * "Choose a solver" dialog — shown after the user clicks Submit Run in the New Run modal.
 * Radio cards for Timefold / Sigma-I D-Wave+SCIP / Sigma-I SCIP+SCIP. Sigma-I options
 * also expose the Stage 1 / Stage 2 time limits (seconds).
 */
import { useState } from 'react';
import { UI } from '@/config/uiConfig';
import { Icon } from './Icon';
import {
  SOLVER_OPTIONS, DEFAULT_SOLVER_OPTION, MIN_TIMEOUT_SEC, MAX_TIMEOUT_SEC,
  defaultSelection, getSolverOption, isValidTimeout,
  type SolverOptionId, type SolverSelection,
} from '@/services/solverOptions';

interface Props {
  busy: boolean;
  onBack: () => void;
  onConfirm: (selection: SolverSelection) => void;
}

export function SolverPickerDialog({ busy, onBack, onConfirm }: Props) {
  const [selected, setSelected] = useState<SolverOptionId>(DEFAULT_SOLVER_OPTION);
  // Kept as strings so the user can clear/retype a field without it snapping back.
  const [stage1, setStage1] = useState('');
  const [stage2, setStage2] = useState('');

  const def = getSolverOption(selected);
  const isSigmai = def.engine === 'sigmai';
  const t1 = Number(stage1);
  const t2 = Number(stage2);
  const timeoutsOk = !isSigmai || (isValidTimeout(t1) && isValidTimeout(t2));

  function choose(id: SolverOptionId) {
    setSelected(id);
    const d = defaultSelection(id);
    setStage1(d.stage1TimeoutSec != null ? String(d.stage1TimeoutSec) : '');
    setStage2(d.stage2TimeoutSec != null ? String(d.stage2TimeoutSec) : '');
  }

  function confirm() {
    if (!timeoutsOk || busy) return;
    onConfirm(isSigmai
      ? { option: selected, stage1TimeoutSec: t1, stage2TimeoutSec: t2 }
      : { option: selected });
  }

  const P = UI.solverPicker;

  return (
    <div className="modal-overlay">
      <div className="modal" style={{ width: 520 }} role="dialog" aria-label={P.title}>
        <div className="modal-hd">
          <div className="modal-title">{P.title}</div>
        </div>
        <div className="modal-body">
          <p style={{ fontSize: 'var(--fs-sm)', color: 'var(--text-sec)', marginBottom: 12 }}>{P.hint}</p>

          <div role="radiogroup" style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {SOLVER_OPTIONS.map(o => {
              const active = o.id === selected;
              return (
                <label
                  key={o.id}
                  style={{
                    display: 'flex', alignItems: 'flex-start', gap: 10, cursor: busy ? 'default' : 'pointer',
                    padding: '10px 12px', borderRadius: 6,
                    border: `1px solid ${active ? 'var(--blue)' : 'var(--bdr)'}`,
                    background: active ? 'var(--blue-lt)' : 'var(--surface)',
                  }}
                >
                  <input
                    type="radio" name="solver-option" checked={active} disabled={busy}
                    onChange={() => choose(o.id)} style={{ marginTop: 3 }}
                  />
                  <span>
                    <span style={{ display: 'block', fontWeight: 600, color: 'var(--text)' }}>
                      {P.options[o.id].name}
                    </span>
                    <span style={{ display: 'block', fontSize: 'var(--fs-sm)', color: 'var(--text-sec)' }}>
                      {P.options[o.id].desc}
                    </span>
                  </span>
                </label>
              );
            })}
          </div>

          {isSigmai && (
            <div style={{ marginTop: 14 }}>
              <div style={{ fontSize: 'var(--fs-sm)', fontWeight: 600, marginBottom: 6 }}>{P.timeoutsTitle}</div>
              <div style={{ display: 'flex', gap: 12 }}>
                <TimeoutField label={P.stage1Timeout} value={stage1} onChange={setStage1} disabled={busy} />
                <TimeoutField label={P.stage2Timeout} value={stage2} onChange={setStage2} disabled={busy} />
              </div>
              {!timeoutsOk && (
                <div style={{ color: 'var(--red)', fontSize: 12, marginTop: 6 }}>
                  {P.timeoutInvalid(MIN_TIMEOUT_SEC, MAX_TIMEOUT_SEC)}
                </div>
              )}
            </div>
          )}
        </div>
        <div className="modal-footer">
          <button className="btn btn-secondary btn-sm" onClick={onBack} disabled={busy}>{P.back}</button>
          <button className="btn btn-primary btn-sm" onClick={confirm} disabled={!timeoutsOk || busy}>
            <Icon name="play" size={13} />{busy ? UI.runLog.uploading : P.start}
          </button>
        </div>
      </div>
    </div>
  );
}

function TimeoutField({ label, value, onChange, disabled }: {
  label: string; value: string; onChange: (v: string) => void; disabled: boolean;
}) {
  return (
    <label style={{ flex: 1, fontSize: 'var(--fs-sm)', color: 'var(--text-sec)' }}>
      {label}
      <input
        type="number" inputMode="numeric" min={MIN_TIMEOUT_SEC} max={MAX_TIMEOUT_SEC} step={1}
        value={value} disabled={disabled} onChange={e => onChange(e.target.value)}
        style={{
          display: 'block', width: '100%', marginTop: 4, padding: '6px 8px',
          border: '1px solid var(--bdr)', borderRadius: 5, fontFamily: 'var(--font-mono)',
        }}
      />
    </label>
  );
}
