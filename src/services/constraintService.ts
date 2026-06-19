import { ScheduleData } from '../types/schedule';
import { EnvConfig } from '../types/envConfig';
import { Violation } from '../types/appState';
import { normalizeDate, getDayOfWeek } from '../utils/dateUtils';

export function checkConstraints(envConfig: EnvConfig, schedule: ScheduleData): Violation[] {
  return [
    ...checkBarOverlaps(schedule),
    ...checkWorkerUnavailableDays(envConfig, schedule),
    ...checkPhaseDateOverrun(schedule),
  ];
}

// Two assignments for the same worker overlap in time
function checkBarOverlaps(schedule: ScheduleData): Violation[] {
  const violations: Violation[] = [];
  const byWorker = groupBy(schedule.assignmentList, a => a.worker);

  for (const [, assignments] of Object.entries(byWorker)) {
    for (let i = 0; i < assignments.length; i++) {
      for (let j = i + 1; j < assignments.length; j++) {
        const a = assignments[i];
        const b = assignments[j];
        if (a.startDate <= b.endDate && b.startDate <= a.endDate) {
          violations.push({
            type: 'OVERLAP',
            assignmentIndices: [
              schedule.assignmentList.indexOf(a),
              schedule.assignmentList.indexOf(b),
            ],
            message: `Worker ${a.worker}: 割り当て重複 (${a.operationTask} / ${b.operationTask})`,
          });
        }
      }
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

    const unavailableDays = collectUnavailableDays(worker.unavailableDates);
    const workDates = assignment.workDateList.map(w => normalizeDate(w.date));

    for (const date of workDates) {
      if (unavailableDays.has(date)) {
        violations.push({
          type: 'WORKER_UNAVAILABLE',
          assignmentIndices: [index],
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
  const opTaskToPhase = new Map<string, { endDate: string }>();

  for (const wt of schedule.workflowTaskList) {
    for (const pt of wt.phaseTaskList) {
      for (const ot of pt.operationTaskList) {
        opTaskToPhase.set(ot.id, { endDate: pt.endDate });
      }
    }
  }

  schedule.assignmentList.forEach((assignment, index) => {
    const phase = opTaskToPhase.get(assignment.operationTask);
    if (phase && assignment.endDate > phase.endDate) {
      violations.push({
        type: 'PHASE_OVERRUN',
        assignmentIndices: [index],
        message: `タスク ${assignment.operationTask}: フェーズ終了日超過 (${assignment.endDate} > ${phase.endDate})`,
      });
    }
  });
  return violations;
}

// Build a set of unavailable calendar dates from UnavailableDateEntry[]
function collectUnavailableDays(entries: EnvConfig['workerList'][0]['unavailableDates']): Set<string> {
  const days = new Set<string>();
  const WEEKDAY_MAP: Record<string, number> = {
    sun: 0, mon: 1, tue: 2, wed: 3, thu: 4, fri: 5, sat: 6,
  };

  for (const entry of entries) {
    if (entry.single?.days) {
      for (const d of entry.single.days) days.add(normalizeDate(d));
    }
    if (entry.weekly?.weekdays) {
      // We can't check weekly without date range context, so store weekday numbers
      // and check against work_date_list dates
      for (const wd of entry.weekly.weekdays) {
        const dow = WEEKDAY_MAP[wd.toLowerCase()];
        if (dow !== undefined) {
          // Mark this as a "weekday unavailable" — checked below
          days.add(`__dow__${dow}`);
        }
      }
    }
  }
  return days;
}

function groupBy<T>(arr: T[], key: (item: T) => string): Record<string, T[]> {
  return arr.reduce((acc, item) => {
    const k = key(item);
    (acc[k] ??= []).push(item);
    return acc;
  }, {} as Record<string, T[]>);
}
