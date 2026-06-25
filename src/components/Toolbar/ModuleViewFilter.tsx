import { useMemo } from 'react';
import { useAppContext } from '../../context/AppContext';
import { FilterChip } from '../common/FilterChip';
import { SelectOption } from '../common/SearchableSelect';
import { filterBarStyles as S } from '../../styles/toolbar';
import { UI } from '../../config/uiText';
import { DEFAULT_MODULE_VIEW_FILTER } from '../../types/appState';

export function ModuleViewFilter() {
  const { state, dispatch } = useAppContext();
  const { moduleViewFilter: f, schedule, envConfig } = state;

  const set = (patch: Partial<typeof f>) =>
    dispatch({ type: 'SET_MODULE_VIEW_FILTER', payload: patch });

  // ── Option lists ─────────────────────────────────────────────────────────

  // Workers who have at least one assignment
  const workerOptions = useMemo<SelectOption[]>(() => {
    if (!schedule || !envConfig) return [];
    const usedIds = new Set(schedule.assignmentList.map(a => a.worker));
    return envConfig.workerList
      .filter(w => usedIds.has(w.id))
      .map(w => {
        const co = envConfig.workerCompanyList.find(c => c.id === w.workerCompany);
        return { value: w.id, label: w.name ?? w.id, sub: co?.name };
      });
  }, [schedule, envConfig]);

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
    f.workerIds.length > 0 ||
    f.fabIds.length > 0 ||
    f.regionIds.length > 0 ||
    !!f.startDate ||
    !!f.endDate;

  const clearAll = () => dispatch({ type: 'SET_MODULE_VIEW_FILTER', payload: { ...DEFAULT_MODULE_VIEW_FILTER } });

  return (
    <div style={S.root}>
      <FilterChip label={UI.mvFilterWorker} options={workerOptions} selected={f.workerIds} onChange={v => set({ workerIds: v })} />
      <FilterChip label={UI.mvFilterFab}    options={fabOptions}    selected={f.fabIds}    onChange={v => set({ fabIds: v })} />
      <FilterChip label={UI.mvFilterRegion} options={regionOptions} selected={f.regionIds} onChange={v => set({ regionIds: v })} />

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
