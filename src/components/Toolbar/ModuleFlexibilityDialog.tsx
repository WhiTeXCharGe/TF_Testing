import { useMemo, useState } from 'react';
import { useAppContext } from '../../context/AppContext';
import { UI } from '../../config/uiText';
import { PlanFlexibility } from '../../types/schedule';

// 'mixed' = the bars under this node don't share one value; null = nothing
// is assigned under it yet, so there is no bar whose flexibility could change.
type Shown = PlanFlexibility | 'mixed' | null;

const FLEX_OPTIONS: { value: PlanFlexibility; label: string }[] = [
  { value: 'Flexible', label: UI.flexibleDesc },
  { value: 'Reluctant', label: UI.reluctantDesc },
  { value: 'Fixed', label: UI.fixedDesc },
];

function combine(values: Shown[]): Shown {
  const real = values.filter((v): v is PlanFlexibility | 'mixed' => v !== null);
  if (real.length === 0) return null;
  const first = real[0];
  return real.every(v => v === first) ? first : 'mixed';
}

interface OpNode { id: string; name: string }
interface PhaseNode { id: string; name: string; ops: OpNode[] }
interface ModuleNode { id: string; name: string; phases: PhaseNode[] }

const overlay: React.CSSProperties = {
  position: 'fixed', inset: 0, backgroundColor: 'rgba(0,0,0,0.4)',
  display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 800,
};
const modal: React.CSSProperties = {
  backgroundColor: '#fff', borderRadius: 6, width: 560, maxHeight: '86vh',
  display: 'flex', flexDirection: 'column',
  boxShadow: '0 8px 24px rgba(0,0,0,0.3)', fontFamily: 'MS Gothic, monospace', overflow: 'hidden',
};
const titleBar: React.CSSProperties = {
  backgroundColor: '#1c2b3a', color: '#fff', padding: '10px 16px', fontSize: 13, fontWeight: 'bold',
};
const body: React.CSSProperties = { padding: '14px 20px', overflowY: 'auto', flex: 1 };
const sectionLabel: React.CSSProperties = { fontSize: 11, color: '#666', marginBottom: 6, fontWeight: 'bold' };
const footer: React.CSSProperties = {
  display: 'flex', justifyContent: 'flex-end', gap: 8,
  padding: '12px 20px', borderTop: '1px solid #e0e0e0', backgroundColor: '#fafafa',
};
const okBtn: React.CSSProperties = {
  padding: '6px 20px', backgroundColor: '#1976d2', color: '#fff', border: 'none',
  borderRadius: 4, cursor: 'pointer', fontSize: 12, fontFamily: 'MS Gothic, monospace',
};
const cancelBtn: React.CSSProperties = {
  padding: '6px 16px', border: '1px solid #aaa', borderRadius: 4, cursor: 'pointer',
  fontSize: 12, backgroundColor: '#fff', fontFamily: 'MS Gothic, monospace',
};
const smallBtn: React.CSSProperties = {
  padding: '2px 8px', border: '1px solid #b8c6d5', borderRadius: 3, cursor: 'pointer',
  fontSize: 11, backgroundColor: '#fff', fontFamily: 'MS Gothic, monospace',
};
const inputStyle: React.CSSProperties = {
  padding: '4px 6px', border: '1px solid #b8c6d5', borderRadius: 3, fontSize: 12,
  fontFamily: 'MS Gothic, monospace',
};

function FlexSelect({ value, onChange, label, placeholder }: {
  value: Shown;
  onChange: (v: PlanFlexibility) => void;
  label: string;
  placeholder?: string;
}) {
  return (
    <select
      aria-label={label}
      value={value === 'mixed' || value === null ? '' : value}
      onChange={e => onChange(e.target.value as PlanFlexibility)}
      style={{ ...inputStyle, cursor: 'pointer' }}
    >
      {value === 'mixed' && <option value="" disabled>{UI.moduleFlexMixed}</option>}
      {value === null && <option value="" disabled>{placeholder ?? UI.moduleFlexNoAssignment}</option>}
      {FLEX_OPTIONS.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
    </select>
  );
}

export function ModuleFlexibilityDialog() {
  const { state, dispatch } = useAppContext();
  const { schedule, envConfig } = state;
  const [isOpen, setIsOpen] = useState(false);
  const [added, setAdded] = useState<string[]>([]);
  const [edits, setEdits] = useState<Record<string, PlanFlexibility>>({});
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [search, setSearch] = useState('');

  const modules = useMemo<ModuleNode[]>(() => {
    if (!schedule) return [];
    const opName = new Map<string, string>();
    for (const wf of envConfig?.workflowList ?? []) {
      for (const ph of wf.phaseList) for (const op of ph.operationList) opName.set(op.id, op.name ?? op.id);
    }
    return schedule.workflowTaskList
      .filter(wt => wt.phaseTaskList.length > 0)
      .map(wt => ({
        id: wt.id,
        name: wt.name ?? wt.id,
        phases: wt.phaseTaskList.map(pt => ({
          id: pt.id,
          name: pt.name ?? pt.phase ?? pt.id,
          ops: pt.operationTaskList.map(ot => ({ id: ot.id, name: ot.name ?? opName.get(ot.operation) ?? ot.operation ?? ot.id })),
        })),
      }));
  }, [schedule, envConfig]);

  // What each 作業's bars are right now: one value, 'mixed', or null (no bars).
  const currentByOp = useMemo(() => {
    const byOp = new Map<string, Shown>();
    for (const a of schedule?.assignmentList ?? []) {
      const prev = byOp.get(a.operationTask);
      byOp.set(a.operationTask, prev === undefined ? a.planFlexibility : prev === a.planFlexibility ? prev : 'mixed');
    }
    return byOp;
  }, [schedule]);

  if (!schedule) return null;

  const moduleById = new Map(modules.map(m => [m.id, m]));
  const addedModules = added.map(id => moduleById.get(id)).filter((m): m is ModuleNode => !!m);

  const opShown = (opId: string): Shown => {
    const current = currentByOp.get(opId) ?? null;
    return current === null ? null : (edits[opId] ?? current);
  };
  const phaseShown = (p: PhaseNode): Shown => combine(p.ops.map(o => opShown(o.id)));
  const moduleShown = (m: ModuleNode): Shown => combine(m.phases.map(phaseShown));

  const setOps = (opIds: string[], flex: PlanFlexibility) =>
    setEdits(prev => {
      const next = { ...prev };
      for (const id of opIds) next[id] = flex;
      return next;
    });
  const phaseOpIds = (p: PhaseNode) => p.ops.map(o => o.id);
  const moduleOpIds = (m: ModuleNode) => m.phases.flatMap(phaseOpIds);

  const needle = search.trim().toLowerCase();
  const candidates = modules.filter(m =>
    !added.includes(m.id) && (!needle || m.name.toLowerCase().includes(needle) || m.id.toLowerCase().includes(needle)),
  );

  const toggleExpanded = (key: string) =>
    setExpanded(prev => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key); else next.add(key);
      return next;
    });

  const removeModule = (m: ModuleNode) => {
    setAdded(prev => prev.filter(id => id !== m.id));
    // drop its edits so they can't be applied after the module is taken off the list
    const ids = new Set(moduleOpIds(m));
    setEdits(prev => Object.fromEntries(Object.entries(prev).filter(([id]) => !ids.has(id))));
  };

  // Only 作業 the user actually changed, and only ones that have bars.
  const changes = addedModules
    .flatMap(moduleOpIds)
    .filter(id => edits[id] !== undefined && (currentByOp.get(id) ?? null) !== null)
    .map(id => ({ operationTaskId: id, flexibility: edits[id] }));

  const open = () => {
    setAdded([]);
    setEdits({});
    setExpanded(new Set());
    setSearch('');
    setIsOpen(true);
  };

  const apply = () => {
    if (changes.length > 0) dispatch({ type: 'BULK_UPDATE_FLEXIBILITY_BY_TASK', payload: { changes } });
    setIsOpen(false);
  };

  const allOpIds = addedModules.flatMap(moduleOpIds);

  return (
    <>
      <button
        style={{ ...smallBtn, padding: '4px 10px', fontSize: 12 }}
        onClick={open}
      >
        {UI.moduleFlexBtn}
      </button>

      {isOpen && (
        <div style={overlay} onClick={e => { if (e.target === e.currentTarget) setIsOpen(false); }}>
          <div style={modal} role="dialog" aria-label={UI.moduleFlexDialogTitle}>
            <div style={titleBar}>{UI.moduleFlexDialogTitle}</div>
            <div style={body}>

              {/* 製番を追加 (searchable) */}
              <div style={{ marginBottom: 14 }}>
                <div style={sectionLabel}>{UI.moduleFlexAddLabel}</div>
                <div style={{ display: 'flex', gap: 6, marginBottom: 6 }}>
                  <input
                    type="text"
                    value={search}
                    onChange={e => setSearch(e.target.value)}
                    placeholder={UI.moduleFlexSearchPlaceholder}
                    style={{ ...inputStyle, flex: 1 }}
                  />
                  <button
                    style={{ ...smallBtn, opacity: candidates.length ? 1 : 0.4 }}
                    disabled={candidates.length === 0}
                    onClick={() => setAdded(prev => [...prev, ...candidates.map(m => m.id)])}
                  >
                    {UI.moduleFlexAddAll}
                  </button>
                </div>
                <div style={{ maxHeight: 120, overflowY: 'auto', border: '1px solid #dde5ef', borderRadius: 3 }}>
                  {candidates.length === 0 && (
                    <div style={{ padding: '6px 8px', fontSize: 11, color: '#888' }}>
                      {modules.length > 0 && added.length === modules.length && !needle ? UI.moduleFlexAllAdded : UI.moduleFlexNoMatch}
                    </div>
                  )}
                  {candidates.map(m => (
                    <div
                      key={m.id}
                      role="button"
                      tabIndex={0}
                      onClick={() => setAdded(prev => [...prev, m.id])}
                      onKeyDown={e => { if (e.key === 'Enter') setAdded(prev => [...prev, m.id]); }}
                      style={{ padding: '4px 8px', fontSize: 12, cursor: 'pointer', borderBottom: '1px solid #edf2f8' }}
                    >
                      + {m.name}
                    </div>
                  ))}
                </div>
              </div>

              {/* 一括設定 for every added module */}
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
                <span style={{ fontSize: 12 }}>{UI.moduleFlexSetAllLabel}</span>
                <FlexSelect
                  label={UI.moduleFlexSetAllLabel}
                  value={addedModules.length === 0 ? null : combine(addedModules.map(moduleShown))}
                  placeholder={UI.moduleFlexSetAllPlaceholder}
                  onChange={flex => setOps(allOpIds, flex)}
                />
              </div>

              {/* Added modules */}
              {addedModules.length === 0 && (
                <div style={{ fontSize: 12, color: '#888', padding: '8px 0' }}>{UI.moduleFlexNothingAdded}</div>
              )}
              {addedModules.map(m => {
                const mOpen = expanded.has(m.id);
                return (
                  <div key={m.id} data-testid={`flex-module-${m.id}`} style={{ border: '1px solid #dde5ef', borderRadius: 4, marginBottom: 8 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '6px 8px', backgroundColor: '#f3f7fb' }}>
                      <button style={smallBtn} aria-label={mOpen ? UI.moduleFlexCollapse : UI.moduleFlexExpand} onClick={() => toggleExpanded(m.id)}>
                        {mOpen ? '▼' : '▶'}
                      </button>
                      <span style={{ flex: 1, fontSize: 12, fontWeight: 'bold' }}>{m.name}</span>
                      <FlexSelect
                        label={`${m.name} ${UI.moduleFlexModuleLevel}`}
                        value={moduleShown(m)}
                        onChange={flex => setOps(moduleOpIds(m), flex)}
                      />
                      <button style={smallBtn} onClick={() => removeModule(m)}>{UI.moduleFlexRemove}</button>
                    </div>

                    {mOpen && m.phases.map(p => {
                      const pKey = `${m.id}/${p.id}`;
                      const pOpen = expanded.has(pKey);
                      return (
                        <div key={p.id} style={{ borderTop: '1px solid #edf2f8' }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '4px 8px 4px 24px' }}>
                            <button style={smallBtn} aria-label={pOpen ? UI.moduleFlexCollapse : UI.moduleFlexExpand} onClick={() => toggleExpanded(pKey)}>
                              {pOpen ? '▼' : '▶'}
                            </button>
                            <span style={{ flex: 1, fontSize: 12 }}>{p.name}</span>
                            <FlexSelect
                              label={`${m.name} ${p.name}`}
                              value={phaseShown(p)}
                              onChange={flex => setOps(phaseOpIds(p), flex)}
                            />
                          </div>
                          {pOpen && p.ops.map(o => (
                            <div key={o.id} style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '3px 8px 3px 56px' }}>
                              <span style={{ flex: 1, fontSize: 11, color: '#444' }}>{o.name}</span>
                              <FlexSelect
                                label={`${m.name} ${p.name} ${o.name}`}
                                value={opShown(o.id)}
                                onChange={flex => setOps([o.id], flex)}
                              />
                            </div>
                          ))}
                        </div>
                      );
                    })}
                  </div>
                );
              })}
            </div>
            <div style={footer}>
              <button style={{ ...okBtn, opacity: changes.length ? 1 : 0.5 }} disabled={changes.length === 0} onClick={apply}>{UI.bulkApply}</button>
              <button style={cancelBtn} onClick={() => setIsOpen(false)}>{UI.dialogCancel}</button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
