import { ScheduleData } from '../types/schedule';
import { EnvConfig } from '../types/envConfig';
import { Violation } from '../types/appState';
import { normalizeDate } from '../utils/dateUtils';

export function checkConstraints(envConfig: EnvConfig, schedule: ScheduleData): Violation[] {
  return [
    ...checkDailyWorkHourRange(envConfig, schedule),
    ...checkSkillMapCompatibility(envConfig, schedule),
    ...checkBarOverlaps(schedule),
    ...checkTaskWorkerCount(envConfig, schedule),
    ...checkWorkerUnavailableDays(envConfig, schedule),
    ...checkPhaseDateOverrun(schedule),
  ];
}

function buildOperationLookups(envConfig: EnvConfig) {
  const byOperationId = new Map<string, { name?: string; minWorkerNum?: number; maxWorkerNum?: number; workHours?: number[]; requiredSkillLevel?: number }>();
  for (const wf of envConfig.workflowList) {
    for (const ph of wf.phaseList) {
      for (const op of ph.operationList) {
        const requiredSkillLevel = Number((op as unknown as Record<string, unknown>).required_skill_level ?? 0);
        byOperationId.set(op.id, {
          name: op.name,
          minWorkerNum: op.minWorkerNum,
          maxWorkerNum: op.maxWorkerNum,
          workHours: op.workHours,
          requiredSkillLevel: Number.isFinite(requiredSkillLevel) ? requiredSkillLevel : 0,
        });
      }
    }
  }
  return byOperationId;
}

function buildOperationTaskLookup(schedule: ScheduleData) {
  const byOperationTaskId = new Map<string, {
    operationId: string;
    phaseId: string;
    phaseStart: string;
    phaseEnd: string;
    recommendsWorkerMin?: number;
    recommendsWorkerMax?: number;
  }>();

  for (const wt of schedule.workflowTaskList) {
    for (const pt of wt.phaseTaskList) {
      for (const ot of pt.operationTaskList) {
        byOperationTaskId.set(ot.id, {
          operationId: ot.operation,
          phaseId: pt.id,
          phaseStart: pt.startDate,
          phaseEnd: pt.endDate,
          recommendsWorkerMin: ot.recommendsWorkerMin,
          recommendsWorkerMax: ot.recommendsWorkerMax,
        });
      }
    }
  }

  return byOperationTaskId;
}

function checkDailyWorkHourRange(envConfig: EnvConfig, schedule: ScheduleData): Violation[] {
  const violations: Violation[] = [];
  const operationLookup = buildOperationLookups(envConfig);
  const operationTaskLookup = buildOperationTaskLookup(schedule);

  schedule.assignmentList.forEach((assignment, assignmentIndex) => {
    const opTask = operationTaskLookup.get(assignment.operationTask);
    const allowed = opTask ? operationLookup.get(opTask.operationId)?.workHours : undefined;

    for (const wd of assignment.workDateList) {
      if (wd.hour <= 0) continue;

      if (allowed && allowed.length > 0 && !allowed.includes(wd.hour)) {
        violations.push({
          type: 'WORK_HOUR_RANGE',
          assignmentIndices: [assignmentIndex],
          date: normalizeDate(wd.date),
          severity: 'error',
          message: `作業時間違反: ${assignment.operationTask} ${normalizeDate(wd.date)} ${wd.hour}h (許容: ${allowed.join(',')})`,
        });
      }

      if (wd.hour > 24) {
        violations.push({
          type: 'WORK_HOUR_RANGE',
          assignmentIndices: [assignmentIndex],
          date: normalizeDate(wd.date),
          severity: 'error',
          message: `作業時間違反: ${assignment.operationTask} ${normalizeDate(wd.date)} ${wd.hour}h (>24h)`,
        });
      }
    }
  });

  return violations;
}

function checkSkillMapCompatibility(envConfig: EnvConfig, schedule: ScheduleData): Violation[] {
  const violations: Violation[] = [];
  const workerById = new Map(envConfig.workerList.map(w => [w.id, w]));
  const operationLookup = buildOperationLookups(envConfig);
  const operationTaskLookup = buildOperationTaskLookup(schedule);

  schedule.assignmentList.forEach((assignment, assignmentIndex) => {
    const worker = workerById.get(assignment.worker);
    const opTask = operationTaskLookup.get(assignment.operationTask);
    if (!worker || !opTask) return;

    const op = operationLookup.get(opTask.operationId);
    const required = op?.requiredSkillLevel ?? 0;
    if (required <= 0) return;

    const workerSkill = Number(worker.skillMap?.[opTask.operationId] ?? 0);
    if (workerSkill >= required) return;

    const date = assignment.workDateList.find(wd => wd.hour > 0)?.date;
    violations.push({
      type: 'SKILL_MISMATCH',
      assignmentIndices: [assignmentIndex],
      date: date ? normalizeDate(date) : undefined,
      severity: 'error',
      message: `スキル不足: worker=${worker.name ?? worker.id} operation=${opTask.operationId} required=${required} actual=${workerSkill}`,
    });
  });

  return violations;
}

// Two assignments for the same worker overlap in time
function checkBarOverlaps(schedule: ScheduleData): Violation[] {
  const violations: Violation[] = [];
  const byWorker: Record<string, Array<{ assignment: ScheduleData['assignmentList'][0]; index: number }>> = {};
  schedule.assignmentList.forEach((assignment, index) => {
    (byWorker[assignment.worker] ??= []).push({ assignment, index });
  });

  for (const assignments of Object.values(byWorker)) {
    const dayToIndices = new Map<string, number[]>();

    for (const { assignment, index } of assignments) {
      for (const wd of assignment.workDateList) {
        if (wd.hour <= 0) continue;
        const d = normalizeDate(wd.date);
        const list = dayToIndices.get(d) ?? [];
        list.push(index);
        dayToIndices.set(d, list);
      }
    }

    for (const [date, indices] of dayToIndices) {
      if (indices.length > 1) {
        const [a, b] = indices;
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

function checkTaskWorkerCount(envConfig: EnvConfig, schedule: ScheduleData): Violation[] {
  const violations: Violation[] = [];
  const operationLookup = buildOperationLookups(envConfig);
  const operationTaskLookup = buildOperationTaskLookup(schedule);

  const keyToWorkers = new Map<string, Set<string>>();

  schedule.assignmentList.forEach(assignment => {
    for (const wd of assignment.workDateList) {
      if (wd.hour <= 0) continue;
      const d = normalizeDate(wd.date);
      const key = `${assignment.operationTask}@@${d}`;
      const set = keyToWorkers.get(key) ?? new Set<string>();
      set.add(assignment.worker);
      keyToWorkers.set(key, set);
    }
  });

  for (const [key, workerSet] of keyToWorkers.entries()) {
    const [operationTaskId, date] = key.split('@@');
    const opTask = operationTaskLookup.get(operationTaskId);
    if (!opTask) continue;
    const op = operationLookup.get(opTask.operationId);

    const min = opTask.recommendsWorkerMin ?? op?.minWorkerNum;
    const max = opTask.recommendsWorkerMax ?? op?.maxWorkerNum;

    if (min != null && workerSet.size < min) {
      const relatedIndices = schedule.assignmentList
        .map((a, i) => ({ a, i }))
        .filter(({ a }) => a.operationTask === operationTaskId && a.workDateList.some(wd => normalizeDate(wd.date) === date && wd.hour > 0))
        .map(x => x.i);
      violations.push({
        type: 'TASK_WORKER_COUNT',
        assignmentIndices: relatedIndices,
        date,
        severity: 'error',
        message: `最小作業者数違反: ${operationTaskId} ${date} count=${workerSet.size} min=${min}`,
      });
    }

    if (max != null && workerSet.size > max) {
      const relatedIndices = schedule.assignmentList
        .map((a, i) => ({ a, i }))
        .filter(({ a }) => a.operationTask === operationTaskId && a.workDateList.some(wd => normalizeDate(wd.date) === date && wd.hour > 0))
        .map(x => x.i);
      violations.push({
        type: 'TASK_WORKER_COUNT',
        assignmentIndices: relatedIndices,
        date,
        severity: 'error',
        message: `最大作業者数違反: ${operationTaskId} ${date} count=${workerSet.size} max=${max}`,
      });
    }
  }

  return violations;
}

// An assignment falls on a worker's unavailable day
function checkWorkerUnavailableDays(envConfig: EnvConfig, schedule: ScheduleData): Violation[] {
  const violations: Violation[] = [];
  const workerMap = new Map(envConfig.workerList.map(w => [w.id, w]));

  schedule.assignmentList.forEach((assignment, index) => {
    const worker = workerMap.get(assignment.worker);
    if (!worker) return;

    const unavailableDays = collectUnavailableDays(worker.unavailableDates, schedule.planRange.startDate, schedule.planRange.endDate);
    const workDates = assignment.workDateList.filter(w => w.hour > 0).map(w => normalizeDate(w.date));

    for (const date of workDates) {
      if (unavailableDays.has(date)) {
        violations.push({
          type: 'WORKER_UNAVAILABLE',
          assignmentIndices: [index],
          date,
          severity: 'error',
          message: `Worker ${worker.name ?? worker.id}: 利用不可日に割り当て (${date})`,
        });
        break; // one violation per assignment is enough
      }
    }
  });
  return violations;
}

// An assignment's end date exceeds its phase task's end date
function checkPhaseDateOverrun(schedule: ScheduleData): Violation[] {
  const violations: Violation[] = [];
  const opTaskToPhase = new Map<string, { startDate: string; endDate: string }>();

  for (const wt of schedule.workflowTaskList) {
    for (const pt of wt.phaseTaskList) {
      for (const ot of pt.operationTaskList) {
        opTaskToPhase.set(ot.id, { startDate: pt.startDate, endDate: pt.endDate });
      }
    }
  }

  schedule.assignmentList.forEach((assignment, index) => {
    const phase = opTaskToPhase.get(assignment.operationTask);
    if (!phase) return;

    let violatedDate: string | undefined;
    for (const wd of assignment.workDateList) {
      if (wd.hour <= 0) continue;
      const d = normalizeDate(wd.date);
      if (d < phase.startDate || d > phase.endDate) {
        violatedDate = d;
        break;
      }
    }

    if (violatedDate) {
      violations.push({
        type: 'PHASE_OVERRUN',
        assignmentIndices: [index],
        date: violatedDate,
        severity: 'error',
        message: `工程開始日・終了日違反: ${assignment.operationTask} (${phase.startDate}..${phase.endDate})`,
      });
    }
  });
  return violations;
}

// Build a set of unavailable calendar dates from UnavailableDateEntry[]
function collectUnavailableDays(entries: EnvConfig['workerList'][0]['unavailableDates'], planStart: string, planEnd: string): Set<string> {
  const days = new Set<string>();
  const WEEKDAY_MAP: Record<string, number> = {
    sun: 0, mon: 1, tue: 2, wed: 3, thu: 4, fri: 5, sat: 6,
  };

  for (const entry of entries) {
    if (entry.single?.days) {
      for (const d of entry.single.days) days.add(normalizeDate(d));
    }
    if (entry.weekly?.weekdays) {
      for (const wd of entry.weekly.weekdays) {
        const dow = WEEKDAY_MAP[wd.toLowerCase()];
        if (dow !== undefined) {
          const cursor = new Date(`${planStart}T00:00:00`);
          const end = new Date(`${planEnd}T00:00:00`);
          while (cursor <= end) {
            if (cursor.getDay() === dow) {
              const y = cursor.getFullYear();
              const m = String(cursor.getMonth() + 1).padStart(2, '0');
              const d = String(cursor.getDate()).padStart(2, '0');
              days.add(`${y}-${m}-${d}`);
            }
            cursor.setDate(cursor.getDate() + 1);
          }
        }
      }
    }
  }
  return days;
}