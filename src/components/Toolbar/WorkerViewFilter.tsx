import { useMemo } from 'react';
import { useAppContext } from '../../context/AppContext';
import { FilterChip } from '../common/FilterChip';
import { SelectOption } from '../common/SearchableSelect';
import { filterBarStyles as S } from '../../styles/toolbar';
import { UI } from '../../config/uiText';
import { DEFAULT_WORKER_VIEW_FILTER } from '../../types/appState';

export function WorkerViewFilter() {
  const { state, dispatch } = useAppContext();
  const { workerViewFilter: f, schedule, envConfig } = state;

  const set = (patch: Partial<typeof f>) =>
    dispatch({ type: 'SET_WORKER_VIEW_FILTER', payload: patch });

  // ── Option lists ─────────────────────────────────────────────────────────

  const moduleOptions = useMemo<SelectOption[]>(() => {
    if (!schedule) return [];
    return schedule.workflowTaskList
      .filter(wt => wt.phaseTaskList && wt.phaseTaskList.length > 0)
      .map(wt => ({ value: wt.id, label: wt.name ?? wt.id }));
  }, [schedule]);

  const phaseOptions = useMemo<SelectOption[]>(() => {
    if (!envConfig) return [];
    const seen = new Map<string, string>();
    for (const wf of envConfig.workflowList) {
      for (const ph of wf.phaseList) {
        if (!seen.has(ph.id)) seen.set(ph.id, ph.name ?? ph.id);
      }
    }
    return [...seen.entries()].map(([value, label]) => ({ value, label }));
  }, [envConfig]);

  const fabOptions = useMemo<SelectOption[]>(() => {
    if (!envConfig) return [];
    return envConfig.fabList.map(f => ({ value: f.id, label: f.name ?? f.id }));
  }, [envConfig]);

  const regionOptions = useMemo<SelectOption[]>(() => {
    if (!envConfig) return [];
    return envConfig.regionList.map(r => ({ value: r.id, label: r.name ?? r.id }));
  }, [envConfig]);

  // ── Dirty check ──────────────────────────────────────────────────────────

  const hasAny =
    !!f.barName ||
    f.moduleIds.length > 0 ||
    f.phaseIds.length > 0 ||
    f.fabIds.length > 0 ||
    f.regionIds.length > 0 ||
    !!f.startDate ||
    !!f.endDate;

  const clearAll = () => dispatch({ type: 'SET_WORKER_VIEW_FILTER', payload: { ...DEFAULT_WORKER_VIEW_FILTER } });

  return (
    <div style={S.root}>
      {/* Bar name text search */}
      <input
        style={{ ...S.textInput, width: 160 }}
        type="text"
        value={f.barName}
        placeholder={UI.wvFilterBarNamePlaceholder}
        onChange={e => set({ barName: e.target.value })}
      />

      <FilterChip label={UI.wvFilterModule} options={moduleOptions} selected={f.moduleIds} onChange={v => set({ moduleIds: v })} />
      <FilterChip label={UI.wvFilterPhase}  options={phaseOptions}  selected={f.phaseIds}  onChange={v => set({ phaseIds: v })} />
      <FilterChip label={UI.wvFilterFab}    options={fabOptions}    selected={f.fabIds}    onChange={v => set({ fabIds: v })} />
      <FilterChip label={UI.wvFilterRegion} options={regionOptions} selected={f.regionIds} onChange={v => set({ regionIds: v })} />

      {/* Date range */}
      <div style={S.dateGroup}>
        <span style={S.dateLabel}>{UI.filterStartDate}</span>
        <input
          style={S.dateInput}
          type="date"
          value={f.startDate ?? ''}
          onChange={e => set({ startDate: e.target.value || null })}
        />
        <span style={S.dateSep}>{UI.filterDateSep}</span>
        <span style={S.dateLabel}>{UI.filterEndDate}</span>
        <input
          style={S.dateInput}
          type="date"
          value={f.endDate ?? ''}
          onChange={e => set({ endDate: e.target.value || null })}
        />
      </div>

      {hasAny && (
        <button style={S.clearAll} onClick={clearAll}>{UI.filterClear}</button>
      )}
    </div>
  );
}
