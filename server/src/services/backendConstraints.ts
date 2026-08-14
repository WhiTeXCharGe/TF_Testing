import type { EnvConfig, ScheduleData, Violation } from '../types.js';

export function runBackendConstraints(envConfig: EnvConfig, schedule: ScheduleData): Violation[] {
  return [
    ...checkBarOverlaps(schedule),
    ...checkTaskWorkerCount(envConfig, schedule),
    ...checkPhaseSequenceOrder(schedule),
    ...checkRequiredWorkload(envConfig, schedule),
    ...checkResponsibleWorker(envConfig, schedule),
    ...checkTravelDays(envConfig, schedule),
    ...checkOvertimeLimits(envConfig, schedule),
  ];
}

// ── 残業時間制約: monthly/annual overtime per worker must stay within their company's limit ──
// Overtime for a calendar day = hours worked beyond the standard 8h/day, summed
// across ALL of that worker's assignments on that day (a worker can be on more
// than one task the same day). Aggregated per worker per month (YYYY-MM) and
// per year (YYYY), then compared against worker_company_list's
// monthly_overtime_limit / annual_overtime_limit. Heavy (scans every work date
// for every assignment) — this is why it only runs on-demand via 制約チェック
// rather than live on every edit.
const STANDARD_DAY_HOURS = 8;

function checkOvertimeLimits(envConfig: EnvConfig, schedule: ScheduleData): Violation[] {
  const violations: Violation[] = [];
  const workerCompanyMap = new Map(envConfig.workerCompanyList.map(wc => [wc.id, wc]));
  const workerById = new Map(envConfig.workerList.map(w => [w.id, w]));

  // worker → date → total hours worked that day (across all their assignments)
  const dailyHoursByWorker = new Map<string, Map<string, number>>();
  // worker → set of assignment indices (for reporting)
  const workerIndices = new Map<string, number[]>();
  schedule.assignmentList.forEach((a, idx) => {
    let dateMap = dailyHoursByWorker.get(a.worker);
    if (!dateMap) { dateMap = new Map(); dailyHoursByWorker.set(a.worker, dateMap); }
    let touchedThisAssignment = false;
    for (const wd of a.workDateList) {
      if (wd.hour <= 0) continue;
      dateMap.set(wd.date, (dateMap.get(wd.date) ?? 0) + wd.hour);
      touchedThisAssignment = true;
    }
    if (touchedThisAssignment) {
      const list = workerIndices.get(a.worker) ?? [];
      list.push(idx);
      workerIndices.set(a.worker, list);
    }
  });

  for (const [workerId, dateMap] of dailyHoursByWorker.entries()) {
    const worker = workerById.get(workerId);
    if (!worker) continue;
    const company = worker.workerCompany ? workerCompanyMap.get(worker.workerCompany) : undefined;
    if (!company) continue;

    const monthlyOvertime = new Map<string, number>();
    const annualOvertime = new Map<string, number>();
    for (const [date, hours] of dateMap.entries()) {
      const overtime = Math.max(0, hours - STANDARD_DAY_HOURS);
      if (overtime <= 0) continue;
      const [year, month] = date.split('-');
      if (!year || !month) continue;
      const monthKey = `${year}-${month}`;
      monthlyOvertime.set(monthKey, (monthlyOvertime.get(monthKey) ?? 0) + overtime);
      annualOvertime.set(year, (annualOvertime.get(year) ?? 0) + overtime);
    }

    const indices = workerIndices.get(workerId) ?? [];
    const workerLabel = worker.name ?? workerId;

    if (company.monthlyOvertimeLimit != null && company.monthlyOvertimeLimit > 0) {
      for (const [monthKey, hours] of monthlyOvertime.entries()) {
        if (hours > company.monthlyOvertimeLimit) {
          violations.push({
            type: 'OVERTIME',
            assignmentIndices: indices,
            severity: 'error',
            message: `残業時間超過(月): worker=${workerLabel} ${monthKey} 残業=${hours}h 上限=${company.monthlyOvertimeLimit}h`,
          });
        }
      }
    }

    if (company.annualOvertimeLimit != null && company.annualOvertimeLimit > 0) {
      for (const [year, hours] of annualOvertime.entries()) {
        if (hours > company.annualOvertimeLimit) {
          violations.push({
            type: 'OVERTIME',
            assignmentIndices: indices,
            severity: 'error',
            message: `残業時間超過(年): worker=${workerLabel} ${year} 残業=${hours}h 上限=${company.annualOvertimeLimit}h`,
          });
        }
      }
    }
  }

  return violations;
}

// ── 同一日作業重複禁止: same worker cannot have two assignments on the same day ──
// (moved from the frontend live-checker for performance with large datasets — was
// re-run on every bar move; now only runs on-demand via 制約チェック.)
function checkBarOverlaps(schedule: ScheduleData): Violation[] {
  const violations: Violation[] = [];
  const byWorker = new Map<string, number[]>();
  schedule.assignmentList.forEach((a, index) => {
    const list = byWorker.get(a.worker) ?? [];
    list.push(index);
    byWorker.set(a.worker, list);
  });

  for (const indices of byWorker.values()) {
    const dayToIndices = new Map<string, number[]>();
    for (const index of indices) {
      const assignment = schedule.assignmentList[index]!;
      for (const wd of assignment.workDateList) {
        if (wd.hour <= 0) continue;
        const list = dayToIndices.get(wd.date) ?? [];
        list.push(index);
        dayToIndices.set(wd.date, list);
      }
    }

    for (const [date, dayIndices] of dayToIndices) {
      if (dayIndices.length > 1) {
        const [a, b] = dayIndices;
        if (a !== undefined && b !== undefined) {
          violations.push({
            type: 'OVERLAP',
            assignmentIndices: [a, b],
            date,
            severity: 'error',
            message: `同一日作業重複禁止: ${date} に同一作業者へ複数割当`,
          });
        }
      }
    }
  }
  return violations;
}

// ── 最小・最大作業者数: unique worker count per operationTask must stay within range ──
function checkTaskWorkerCount(envConfig: EnvConfig, schedule: ScheduleData): Violation[] {
  const violations: Violation[] = [];

  // operationTask → { operationId, recommendsWorkerMin, recommendsWorkerMax }
  const opTaskInfo = new Map<string, { operationId: string; min?: number; max?: number }>();
  for (const wt of schedule.workflowTaskList) {
    for (const pt of wt.phaseTaskList) {
      for (const ot of pt.operationTaskList) {
        opTaskInfo.set(ot.id, {
          operationId: ot.operation,
          min: ot.recommendsWorkerMin,
          max: ot.recommendsWorkerMax,
        });
      }
    }
  }

  // operation → { minWorkerNum, maxWorkerNum } fallback when the task itself doesn't override it
  const operationRange = new Map<string, { min?: number; max?: number }>();
  for (const wf of envConfig.workflowList) {
    for (const ph of wf.phaseList) {
      for (const op of ph.operationList) {
        operationRange.set(op.id, { min: op.minWorkerNum, max: op.maxWorkerNum });
      }
    }
  }

  // Count unique workers per operationTask across all dates (not per-day)
  const opTaskWorkers = new Map<string, Set<string>>();
  schedule.assignmentList.forEach(a => {
    if (!a.workDateList.some(wd => wd.hour > 0)) return;
    const set = opTaskWorkers.get(a.operationTask) ?? new Set<string>();
    set.add(a.worker);
    opTaskWorkers.set(a.operationTask, set);
  });

  for (const [operationTaskId, workerSet] of opTaskWorkers.entries()) {
    const info = opTaskInfo.get(operationTaskId);
    if (!info) continue;
    const opRange = operationRange.get(info.operationId);

    const min = info.min ?? opRange?.min;
    const max = info.max ?? opRange?.max;

    const relatedIndices = schedule.assignmentList
      .map((a, i) => ({ a, i }))
      .filter(({ a }) => a.operationTask === operationTaskId && a.workDateList.some(wd => wd.hour > 0))
      .map(x => x.i);

    if (min != null && workerSet.size < min) {
      violations.push({
        type: 'TASK_WORKER_COUNT',
        assignmentIndices: relatedIndices,
        severity: 'error',
        message: `最小作業者数違反: ${operationTaskId} count=${workerSet.size} min=${min}`,
      });
    }

    if (max != null && workerSet.size > max) {
      violations.push({
        type: 'TASK_WORKER_COUNT',
        assignmentIndices: relatedIndices,
        severity: 'error',
        message: `最大作業者数違反: ${operationTaskId} count=${workerSet.size} max=${max}`,
      });
    }
  }

  return violations;
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

// ── 必要作業量: total assigned hours for an operationTask must land in
// [required, required + X*T], where:
//   required = ot.workloadHours (the declared target for the task)
//   X        = number of unique workers actually assigned work on the task
//   T        = the operation's max declared hours/day (op.workHours), i.e.
//              one worker's biggest possible single-day contribution
//
// The ceiling is "required + one extra worker-day of slack" — formally
// required=X·T·D and ceiling=X·T·(D+1), and since (D+1)−D is always 1,
// ceiling−required always reduces to exactly X·T regardless of D. That
// cancellation is deliberate: D (how many distinct days the task took) is
// not well-defined when different workers on the same task work different,
// non-overlapping date ranges. (Formerly duplicated on the frontend, which
// re-ran this scan on every edit — moved here entirely for performance.)
function checkRequiredWorkload(envConfig: EnvConfig, schedule: ScheduleData): Violation[] {
  const violations: Violation[] = [];

  // operation → declared max hours/day
  const operationWorkHours = new Map<string, number[]>();
  for (const wf of envConfig.workflowList) {
    for (const ph of wf.phaseList) {
      for (const op of ph.operationList) {
        if (op.workHours && op.workHours.length > 0) operationWorkHours.set(op.id, op.workHours);
      }
    }
  }

  // operationTask → { workloadHours, operationId, label }
  const opTaskMeta = new Map<string, { workloadHours: number; operationId: string; label: string }>();
  for (const wt of schedule.workflowTaskList) {
    for (const pt of wt.phaseTaskList) {
      for (const ot of pt.operationTaskList) {
        if (ot.workloadHours > 0) {
          opTaskMeta.set(ot.id, { workloadHours: ot.workloadHours, operationId: ot.operation, label: ot.name ?? ot.id });
        }
      }
    }
  }
  if (opTaskMeta.size === 0) return violations;

  // Sum actual hours, unique workers, and the biggest single-day hour value
  // actually observed, per operationTask.
  const actualMap = new Map<string, { total: number; workers: Set<string>; maxObservedDayHours: number; indices: number[] }>();
  schedule.assignmentList.forEach((a, idx) => {
    if (!opTaskMeta.has(a.operationTask)) return;
    const sum = a.workDateList.reduce((acc, wd) => acc + (wd.hour > 0 ? wd.hour : 0), 0);
    if (sum === 0) return;
    const entry = actualMap.get(a.operationTask) ?? { total: 0, workers: new Set<string>(), maxObservedDayHours: 0, indices: [] };
    entry.total += sum;
    entry.workers.add(a.worker);
    for (const wd of a.workDateList) {
      if (wd.hour > entry.maxObservedDayHours) entry.maxObservedDayHours = wd.hour;
    }
    entry.indices.push(idx);
    actualMap.set(a.operationTask, entry);
  });

  for (const [opTaskId, { total, workers, maxObservedDayHours, indices }] of actualMap.entries()) {
    const meta = opTaskMeta.get(opTaskId)!;
    const declared = operationWorkHours.get(meta.operationId);
    const T = declared && declared.length > 0 ? Math.max(...declared) : (maxObservedDayHours || 8);
    const X = workers.size;
    const ceiling = meta.workloadHours + X * T;

    if (total < meta.workloadHours) {
      violations.push({
        type: 'WORKLOAD_TOTAL',
        assignmentIndices: indices,
        severity: 'warning',
        message: `必要作業量不足: ${meta.label} 実績=${total}h 必要=${meta.workloadHours}h (不足 ${meta.workloadHours - total}h)`,
      });
    } else if (total > ceiling) {
      violations.push({
        type: 'WORKLOAD_TOTAL',
        assignmentIndices: indices,
        severity: 'warning',
        message: `過剰作業量: ${meta.label} 実績=${total}h 必要=${meta.workloadHours}h 上限=${ceiling}h`,
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
