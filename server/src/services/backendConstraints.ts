import type { EnvConfig, ScheduleData, Violation } from '../types.js';

export function runBackendConstraints(envConfig: EnvConfig, schedule: ScheduleData): Violation[] {
  return [
    ...checkPhaseSequenceOrder(schedule),
    ...checkRequiredWorkload(envConfig, schedule),
    ...checkResponsibleWorker(envConfig, schedule),
    ...checkTravelDays(envConfig, schedule),
  ];
}

// ── 工程間順序: phase N+1 must not start before all assignments in phase N finish ──
function checkPhaseSequenceOrder(schedule: ScheduleData): Violation[] {
  const violations: Violation[] = [];

  for (const wt of schedule.workflowTaskList) {
    const phases = wt.phaseTaskList;
    if (phases.length < 2) continue;

    // Collect last work date for each phase
    const phaseLastDate: Map<string, string> = new Map();
    for (const pt of phases) {
      let last = '';
      for (const ot of pt.operationTaskList) {
        for (const a of schedule.assignmentList) {
          if (a.operationTask !== ot.id) continue;
          for (const wd of a.workDateList) {
            if (wd.hour > 0 && wd.date > last) last = wd.date;
          }
        }
      }
      if (last) phaseLastDate.set(pt.id, last);
    }

    // Check each consecutive pair
    for (let i = 0; i < phases.length - 1; i++) {
      const current = phases[i];
      const next = phases[i + 1];
      if (!current || !next) continue;

      const currentLastDate = phaseLastDate.get(current.id);
      if (!currentLastDate) continue;

      // Find assignments in the next phase that start before current phase ends
      for (const ot of next.operationTaskList) {
        for (let idx = 0; idx < schedule.assignmentList.length; idx++) {
          const a = schedule.assignmentList[idx]!;
          if (a.operationTask !== ot.id) continue;
          const firstWorkDate = a.workDateList
            .filter(wd => wd.hour > 0)
            .map(wd => wd.date)
            .sort()[0];
          if (firstWorkDate && firstWorkDate <= currentLastDate) {
            violations.push({
              type: 'PHASE_SEQUENCE',
              assignmentIndices: [idx],
              date: firstWorkDate,
              severity: 'error',
              message: `工程順序違反: ${wt.name ?? wt.id} — ${next.name ?? next.id} が ${current.name ?? current.id} の完了前(${currentLastDate})に開始しています (${firstWorkDate})`,
            });
          }
        }
      }
    }
  }

  return violations;
}

// ── 必要作業量: total assigned hours must meet workloadHours for each operationTask ──
function checkRequiredWorkload(_envConfig: EnvConfig, schedule: ScheduleData): Violation[] {
  const violations: Violation[] = [];

  // Build operationTask → workloadHours map
  const workloadRequired = new Map<string, number>();
  const opTaskName = new Map<string, string>();
  for (const wt of schedule.workflowTaskList) {
    for (const pt of wt.phaseTaskList) {
      for (const ot of pt.operationTaskList) {
        if (ot.workloadHours > 0) workloadRequired.set(ot.id, ot.workloadHours);
        opTaskName.set(ot.id, ot.name ?? ot.id);
      }
    }
  }

  // Sum assigned hours per operationTask
  const assignedHours = new Map<string, { total: number; indices: number[] }>();
  schedule.assignmentList.forEach((a, idx) => {
    const hours = a.workDateList.reduce((sum, wd) => sum + (wd.hour > 0 ? wd.hour : 0), 0);
    const existing = assignedHours.get(a.operationTask) ?? { total: 0, indices: [] };
    assignedHours.set(a.operationTask, {
      total: existing.total + hours,
      indices: [...existing.indices, idx],
    });
  });

  for (const [opTaskId, required] of workloadRequired.entries()) {
    const assigned = assignedHours.get(opTaskId);
    const total = assigned?.total ?? 0;
    if (total < required) {
      violations.push({
        type: 'WORKLOAD_TOTAL',
        assignmentIndices: assigned?.indices ?? [],
        severity: 'warning',
        message: `必要作業量不足: ${opTaskName.get(opTaskId) ?? opTaskId} — 必要 ${required}h / 割当 ${total}h (不足 ${required - total}h)`,
      });
    }
    if (total > required) {
      violations.push({
        type: 'WORKLOAD_TOTAL',
        assignmentIndices: assigned?.indices ?? [],
        severity: 'warning',
        message: `作業量超過: ${opTaskName.get(opTaskId) ?? opTaskId} — 必要 ${required}h / 割当 ${total}h (超過 ${total - required}h)`,
      });
    }
  }

  return violations;
}

// ── 作業責任者: each operationTask must have at least one manager-role worker ──
function checkResponsibleWorker(envConfig: EnvConfig, schedule: ScheduleData): Violation[] {
  const violations: Violation[] = [];
  const managerIds = new Set(envConfig.workerList.filter(w => w.isManager).map(w => w.id));
  if (managerIds.size === 0) return [];

  // Group assignments by operationTask
  const byOpTask = new Map<string, { indices: number[]; workers: Set<string> }>();
  schedule.assignmentList.forEach((a, idx) => {
    const entry = byOpTask.get(a.operationTask) ?? { indices: [], workers: new Set() };
    entry.indices.push(idx);
    entry.workers.add(a.worker);
    byOpTask.set(a.operationTask, entry);
  });

  // Build operationTask name map
  const opTaskName = new Map<string, string>();
  for (const wt of schedule.workflowTaskList) {
    for (const pt of wt.phaseTaskList) {
      for (const ot of pt.operationTaskList) {
        opTaskName.set(ot.id, ot.name ?? ot.id);
      }
    }
  }

  for (const [opTaskId, { indices, workers }] of byOpTask.entries()) {
    const hasManager = [...workers].some(wId => managerIds.has(wId));
    if (!hasManager) {
      violations.push({
        type: 'RESPONSIBLE_WORKER',
        assignmentIndices: indices,
        severity: 'error',
        message: `作業責任者なし: ${opTaskName.get(opTaskId) ?? opTaskId} — 責任者（isManager）が割り当てられていません`,
      });
    }
  }

  return violations;
}

// ── 移動日制: worker moving between regions needs transit days in between ──
function checkTravelDays(envConfig: EnvConfig, schedule: ScheduleData): Violation[] {
  const violations: Violation[] = [];
  if (!envConfig.transiteDayMap || envConfig.transiteDayMap.length === 0) return [];

  // Build fab → region map
  const fabRegion = new Map(envConfig.fabList.map(f => [f.id, f.region ?? '']));

  // Build operationTask → fab map
  const opTaskFab = new Map<string, string>();
  for (const wt of schedule.workflowTaskList) {
    const fab = wt.fab ?? '';
    for (const pt of wt.phaseTaskList) {
      for (const ot of pt.operationTaskList) {
        opTaskFab.set(ot.id, fab);
      }
    }
  }

  // Build transit lookup: "fromRegion→toRegion" → days
  const transitMap = new Map<string, number>();
  for (const t of envConfig.transiteDayMap) {
    transitMap.set(`${t.from}→${t.to}`, t.days);
  }

  // Group assignments by worker, sorted by first work date
  const byWorker = new Map<string, Array<{ idx: number; opTaskId: string; workDates: string[] }>>();
  schedule.assignmentList.forEach((a, idx) => {
    const dates = a.workDateList.filter(wd => wd.hour > 0).map(wd => wd.date).sort();
    if (dates.length === 0) return;
    const list = byWorker.get(a.worker) ?? [];
    list.push({ idx, opTaskId: a.operationTask, workDates: dates });
    byWorker.set(a.worker, list);
  });

  for (const [, assignments] of byWorker.entries()) {
    const sorted = assignments.sort((a, b) => (a.workDates[0] ?? '') < (b.workDates[0] ?? '') ? -1 : 1);

    for (let i = 0; i < sorted.length - 1; i++) {
      const curr = sorted[i];
      const next = sorted[i + 1];
      if (!curr || !next) continue;

      const currFab = opTaskFab.get(curr.opTaskId) ?? '';
      const nextFab = opTaskFab.get(next.opTaskId) ?? '';
      const currRegion = fabRegion.get(currFab) ?? '';
      const nextRegion = fabRegion.get(nextFab) ?? '';

      if (!currRegion || !nextRegion || currRegion === nextRegion) continue;

      const requiredDays = transitMap.get(`${currRegion}→${nextRegion}`) ?? transitMap.get(`${nextRegion}→${currRegion}`);
      if (!requiredDays) continue;

      const currLastDate = curr.workDates[curr.workDates.length - 1] ?? '';
      const nextFirstDate = next.workDates[0] ?? '';

      // Calculate gap in days
      const gapMs = new Date(nextFirstDate).getTime() - new Date(currLastDate).getTime();
      const gapDays = Math.floor(gapMs / 86400000);

      if (gapDays < requiredDays) {
        violations.push({
          type: 'TRAVEL_DAYS',
          assignmentIndices: [curr.idx, next.idx],
          date: nextFirstDate,
          severity: 'error',
          message: `移動日不足: ${currRegion} → ${nextRegion} — 必要 ${requiredDays}日 / 実際 ${gapDays}日 (${currLastDate} → ${nextFirstDate})`,
        });
      }
    }
  }

  return violations;
}
