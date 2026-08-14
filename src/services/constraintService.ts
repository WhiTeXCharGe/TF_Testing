import { ScheduleData } from '../types/schedule';
import { EnvConfig } from '../types/envConfig';
import { Violation } from '../types/appState';
import { normalizeDate } from '../utils/dateUtils';
import { UI } from '../config/uiText';

export function checkConstraints(envConfig: EnvConfig, schedule: ScheduleData): Violation[] {
  return [
    ...checkDailyWorkHourRange(envConfig, schedule),
    ...checkSkillMapCompatibility(envConfig, schedule),
    ...checkWorkerUnavailableDays(envConfig, schedule),
    ...checkPhaseDateOverrun(schedule),
    ...checkRegionSuitability(envConfig, schedule),
    ...checkCompanySuitability(envConfig, schedule),
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
          message: UI.workHourRangeViolation(assignment.operationTask, normalizeDate(wd.date), wd.hour, allowed.join(',')),
        });
      }

      if (wd.hour > 24) {
        violations.push({
          type: 'WORK_HOUR_RANGE',
          assignmentIndices: [assignmentIndex],
          date: normalizeDate(wd.date),
          severity: 'error',
          message: UI.workHourOver24Violation(assignment.operationTask, normalizeDate(wd.date), wd.hour),
        });
      }
    }
  });

  return violations;
}

function checkSkillMapCompatibility(envConfig: EnvConfig, schedule: ScheduleData): Violation[] {
  const violations: Violation[] = [];
  const miscTaskIds = buildMiscTaskIds(schedule);
  const workerById = new Map(envConfig.workerList.map(w => [w.id, w]));
  const operationLookup = buildOperationLookups(envConfig);
  const operationTaskLookup = buildOperationTaskLookup(schedule);

  schedule.assignmentList.forEach((assignment, assignmentIndex) => {
    if (miscTaskIds.has(assignment.operationTask)) return;
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
      message: UI.skillMismatchViolation(worker.name ?? worker.id, opTask.operationId, required, workerSkill),
    });
  });

  return violations;
}

function buildMiscTaskIds(schedule: ScheduleData): Set<string> {
  return new Set(
    schedule.workflowTaskList.filter(wt => wt.phaseTaskList.length === 0).map(wt => wt.id),
  );
}

// Resolve operationTask → regionId, mirroring workerViewModel.ts's opTaskRegionMap
// (misc tasks — empty phaseTaskList — use wt.region directly since they have no fab).
function buildOpTaskRegionMap(envConfig: EnvConfig, schedule: ScheduleData): Map<string, string | null> {
  const fabRegionMap = new Map(envConfig.fabList.map(f => [f.id, f.region ?? null]));
  const opTaskRegionMap = new Map<string, string | null>();
  for (const wt of schedule.workflowTaskList) {
    if (wt.phaseTaskList.length === 0) {
      opTaskRegionMap.set(wt.id, wt.region ?? (wt.fab ? fabRegionMap.get(wt.fab) ?? null : null));
      continue;
    }
    const regionId = wt.fab ? fabRegionMap.get(wt.fab) ?? null : null;
    for (const pt of wt.phaseTaskList) {
      for (const ot of pt.operationTaskList) {
        opTaskRegionMap.set(ot.id, regionId);
      }
    }
  }
  return opTaskRegionMap;
}

// Same idea as buildOpTaskRegionMap but for Fab.customerCompany. Misc tasks have
// no fab reference, so they have no resolvable company and are skipped.
function buildOpTaskCompanyMap(envConfig: EnvConfig, schedule: ScheduleData): Map<string, string | null> {
  const fabCompanyMap = new Map(envConfig.fabList.map(f => [f.id, f.customerCompany ?? null]));
  const opTaskCompanyMap = new Map<string, string | null>();
  for (const wt of schedule.workflowTaskList) {
    if (wt.phaseTaskList.length === 0) {
      opTaskCompanyMap.set(wt.id, wt.fab ? fabCompanyMap.get(wt.fab) ?? null : null);
      continue;
    }
    const companyId = wt.fab ? fabCompanyMap.get(wt.fab) ?? null : null;
    for (const pt of wt.phaseTaskList) {
      for (const ot of pt.operationTaskList) {
        opTaskCompanyMap.set(ot.id, companyId);
      }
    }
  }
  return opTaskCompanyMap;
}

// 地域適性: a worker assigned to a region where their fab_suitability_map (kind:
// 'region') score is exactly 0 is unsuitable for that region. Workers/regions with
// no explicit score entry are treated as unrestricted (no violation).
function checkRegionSuitability(envConfig: EnvConfig, schedule: ScheduleData): Violation[] {
  const violations: Violation[] = [];
  const workerById = new Map(envConfig.workerList.map(w => [w.id, w]));
  const opTaskRegionMap = buildOpTaskRegionMap(envConfig, schedule);

  schedule.assignmentList.forEach((assignment, index) => {
    const worker = workerById.get(assignment.worker);
    if (!worker) return;
    const regionId = opTaskRegionMap.get(assignment.operationTask);
    if (!regionId) return;
    const regionEntry = worker.fabSuitabilityMap?.find(e => e.kind === 'region');
    const score = regionEntry?.suitability[regionId];
    if (score !== 0) return;

    const date = assignment.workDateList.find(wd => wd.hour > 0)?.date;
    violations.push({
      type: 'REGION_SUITABILITY',
      assignmentIndices: [index],
      date: date ? normalizeDate(date) : undefined,
      severity: 'error',
      message: UI.regionSuitabilityViolation(worker.name ?? worker.id, regionId),
    });
  });

  return violations;
}

// 企業適性: same idea as region suitability, but keyed on fab_suitability_map's
// 'customer_company' entry and Fab.customerCompany.
function checkCompanySuitability(envConfig: EnvConfig, schedule: ScheduleData): Violation[] {
  const violations: Violation[] = [];
  const workerById = new Map(envConfig.workerList.map(w => [w.id, w]));
  const opTaskCompanyMap = buildOpTaskCompanyMap(envConfig, schedule);

  schedule.assignmentList.forEach((assignment, index) => {
    const worker = workerById.get(assignment.worker);
    if (!worker) return;
    const companyId = opTaskCompanyMap.get(assignment.operationTask);
    if (!companyId) return;
    const companyEntry = worker.fabSuitabilityMap?.find(e => e.kind === 'customer_company');
    const score = companyEntry?.suitability[companyId];
    if (score !== 0) return;

    const date = assignment.workDateList.find(wd => wd.hour > 0)?.date;
    violations.push({
      type: 'COMPANY_SUITABILITY',
      assignmentIndices: [index],
      date: date ? normalizeDate(date) : undefined,
      severity: 'error',
      message: UI.companySuitabilityViolation(worker.name ?? worker.id, companyId),
    });
  });

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
          message: UI.workerUnavailableViolation(worker.name ?? worker.id, date),
        });
        break; // one violation per assignment is enough
      }
    }
  });
  return violations;
}

// An assignment's end date exceeds its phase task's end date (skips misc tasks)
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
    // misc task assignments have no phase entry — skip
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
        message: UI.phaseOverrunViolation(assignment.operationTask, phase.startDate, phase.endDate),
      });
    }
  });
  return violations;
}

// 必要作業量 (workload total) moved entirely to the backend
// (server/src/services/backendConstraints.ts) — it's a heavy per-assignment
// scan across every operationTask, too slow to re-run live on every edit.

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