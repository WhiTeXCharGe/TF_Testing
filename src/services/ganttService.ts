/**
 * Gantt data builder — faithful TypeScript port of yaml_to_suother_like_excel.py.
 *
 * Reads the REAL EnvConfig.yaml + Schedule.yaml structures:
 *   EnvConfig: environment.worker_company_list[], environment.worker_list[]
 *   Schedule:  schedule.plan_range{start_date,end_date},
 *              schedule.workflow_task_list[] (→ phase_task_list → operation_task_list),
 *              schedule.assignment_list[] (worker, operation_task, work_date_list[])
 *
 * Output: GanttData (employees × dates grid of GanttCell).
 *
 * Mirrors the Python rules:
 *   - Cell shows the MODULE NAME (e.g. "SU 1001A"); first assignment per day wins.
 *   - Module color is stable by FIRST-OCCURRENCE ORDER in workflow_task_list,
 *     keyed by normalized base code (text before first "_").
 *   - RED is reserved ONLY for a worker's unavailable_dates.
 *   - Workers included = those with ≥1 assignment OR ≥1 unavailable day in range
 *     (INCLUDE_ALL_WORKERS = False), sorted by company name then worker name.
 */
import type {
  RawEnvConfig, RawSchedule, RawScheduleBody, RawEnvironment,
  RawWorker, RawWorkflowTask, RawWorkDate,
  GanttData, GanttEmployee, GanttCell, GanttModule,
} from '@/types';
import { parseDate, dateRange, toKey } from '@/utils/dateUtils';
import {
  normalizeModuleCode, assignModuleColorsByOrder, companyColor, MODULE_UNKNOWN_COLOR,
} from '@/utils/colorUtils';

// ── Unavailable date parsing (port of parse_unavailable_dates) ──────────────
// Supports: { date: "..." } | plain "YYYY/MM/DD" | { single: { days: [...] } }
//           | { weekly: { weekdays: ["sat","sun",...] } }
const WEEKDAY_TO_INT: Record<string, number> = {
  mon: 0, monday: 0, tue: 1, tues: 1, tuesday: 1, wed: 2, weds: 2, wednesday: 2,
  thu: 3, thur: 3, thurs: 3, thursday: 3, fri: 4, friday: 4,
  sat: 5, saturday: 5, sun: 6, sunday: 6,
};

/** JS getDay() (0=Sun..6=Sat) → Python weekday() (0=Mon..6=Sun). */
function pyWeekday(d: Date): number {
  return (d.getDay() + 6) % 7;
}

function parseUnavailDates(raw: unknown, planStart: Date, planEnd: Date): Set<string> {
  const off = new Set<string>();
  if (raw == null) return off;

  const addSingle = (val: unknown) => {
    try {
      const dd = parseDate(String(val));
      if (dd >= planStart && dd <= planEnd) off.add(toKey(dd));
    } catch { /* ignore unparseable */ }
  };

  const items = Array.isArray(raw) ? raw : [raw];
  const weekly = new Set<number>();

  for (const item of items) {
    if (item == null) continue;

    if (typeof item === 'object') {
      const obj = item as Record<string, unknown>;

      // Case A: { date: "..." }
      if ('date' in obj) { addSingle(obj['date']); continue; }

      // Case C: { single: { days: [...] } }
      const single = obj['single'];
      if (single && typeof single === 'object') {
        const days = (single as Record<string, unknown>)['days'];
        if (Array.isArray(days)) days.forEach(addSingle);
      }
      // Case C: { weekly: { weekdays: [...] } }
      const weeklyObj = obj['weekly'];
      if (weeklyObj && typeof weeklyObj === 'object') {
        const wds = (weeklyObj as Record<string, unknown>)['weekdays'];
        if (Array.isArray(wds)) {
          for (const w of wds) {
            const wd = WEEKDAY_TO_INT[String(w).trim().toLowerCase()];
            if (wd != null) weekly.add(wd);
          }
        }
      }
      continue;
    }

    // Case B: plain scalar date
    addSingle(item);
  }

  if (weekly.size > 0) {
    for (const d of dateRange(planStart, planEnd)) {
      if (weekly.has(pyWeekday(d))) off.add(toKey(d));
    }
  }
  return off;
}

// ── operation_task.id → workflow_task(module).id  (port of build_op_task_index)
function buildOpTaskIndex(modules: RawWorkflowTask[]): Map<string, string> {
  const idx = new Map<string, string>();
  for (const m of modules) {
    for (const ph of m.phase_task_list ?? []) {
      for (const ot of ph.operation_task_list ?? []) {
        idx.set(ot.id, m.id);
      }
    }
  }
  return idx;
}

// ── Root unwrapping (env.get("environment", env) / sched.get("schedule", sched))
function envRoot(env: RawEnvConfig): RawEnvironment {
  return (env.environment ?? env) as RawEnvironment;
}
function schedRoot(sched: RawSchedule): RawScheduleBody {
  return (sched.schedule ?? sched) as RawScheduleBody;
}

// ── Main builder ──────────────────────────────────────────────────────────
export function buildGanttData(env: RawEnvConfig, sched: RawSchedule): GanttData {
  const e = envRoot(env);
  const s = schedRoot(sched);

  // Plan range (full range from plan_range)
  const planStart = parseDate(s.plan_range.start_date);
  const planEnd   = parseDate(s.plan_range.end_date);
  const dates     = dateRange(planStart, planEnd);

  // Lookups
  const workerCompanies = new Map<string, string>();   // companyId → name
  for (const c of e.worker_company_list ?? []) workerCompanies.set(c.id, c.name ?? c.id);

  const workers = new Map<string, RawWorker>();
  for (const w of e.worker_list ?? []) workers.set(w.id, w);

  const modules = s.workflow_task_list ?? [];
  const moduleNameById = new Map<string, string>();
  for (const m of modules) moduleNameById.set(m.id, m.name ?? m.id);

  // Module colors: stable by first-occurrence order of normalized base code.
  const baseToColor = assignModuleColorsByOrder(
    modules.map(m => normalizeModuleCode(m.name ?? '')),
  );
  const moduleColorById = new Map<string, string>();
  for (const m of modules) {
    const base = normalizeModuleCode(m.name ?? '');
    moduleColorById.set(m.id, baseToColor.get(base) ?? MODULE_UNKNOWN_COLOR);
  }

  const opTaskToModule = buildOpTaskIndex(modules);

  // Expand assignments to (workerId, dateKey) → first module id.
  const cellModule = new Map<string, string>();   // key: `${wid}__${dateKey}`
  const assignedWorkers = new Set<string>();
  for (const a of s.assignment_list ?? []) {
    const wid = a.worker;
    const mid = opTaskToModule.get(a.operation_task) ?? `UNKNOWN::${a.operation_task}`;
    const wdList: RawWorkDate[] = a.work_date_list ?? a.work_date_lsit ?? [];
    for (const wd of wdList) {
      if (!wd?.date) continue;
      const dd = parseDate(wd.date);
      if (dd < planStart || dd > planEnd) continue;
      const key = `${wid}__${toKey(dd)}`;
      if (!cellModule.has(key)) cellModule.set(key, mid);   // first only
      assignedWorkers.add(wid);
    }
  }

  // Worker unavailable map + which workers have any day off in range.
  const workerOff = new Map<string, Set<string>>();
  const workersWithOff = new Set<string>();
  for (const [wid, w] of workers) {
    const off = parseUnavailDates(w.unavailable_dates, planStart, planEnd);
    workerOff.set(wid, off);
    if (off.size > 0) workersWithOff.add(wid);
  }

  // Included workers (INCLUDE_ALL_WORKERS = False): assigned OR with days off.
  const includedIds = [...new Set([...assignedWorkers, ...workersWithOff])];

  const companyNameOf = (wid: string) => {
    const w = workers.get(wid);
    return (w && workerCompanies.get(w.worker_company ?? '')) ?? w?.worker_company ?? '';
  };
  const workerNameOf = (wid: string) => workers.get(wid)?.name ?? wid;

  // Sort by company name, then worker name.
  includedIds.sort((a, b) => {
    const ca = companyNameOf(a), cb = companyNameOf(b);
    if (ca !== cb) return ca.localeCompare(cb);
    return workerNameOf(a).localeCompare(workerNameOf(b));
  });

  // Employee rows.
  const employees: GanttEmployee[] = includedIds.map(wid => {
    const w = workers.get(wid)!;
    const company = companyNameOf(wid);
    return {
      id: wid,
      name: workerNameOf(wid),
      company,
      companyColor: companyColor(company),
      role: String(w.role ?? ''),
      isManager: Boolean(w.is_manager),
    };
  });

  // Modules present in the grid (for the legend), in workflow order.
  const usedModuleIds = new Set<string>();
  for (const mid of cellModule.values()) if (!mid.startsWith('UNKNOWN::')) usedModuleIds.add(mid);
  const ganttModules: GanttModule[] = modules
    .filter(m => usedModuleIds.has(m.id))
    .map(m => {
      const base = normalizeModuleCode(m.name ?? '');
      return { code: m.name ?? m.id, baseCode: base, color: baseToColor.get(base) ?? MODULE_UNKNOWN_COLOR };
    });

  const todayDate = new Date();
  todayDate.setHours(0, 0, 0, 0);
  const todayKey = toKey(todayDate);

  // Cell grid.
  const cells: GanttCell[][] = employees.map(emp => {
    const off = workerOff.get(emp.id) ?? new Set<string>();
    return dates.map(d => {
      const dk = toKey(d);
      const isToday = dk === todayKey;

      // Unavailable → red (takes precedence).
      if (off.has(dk)) return { type: 'unavailable', isToday } as GanttCell;

      const mid = cellModule.get(`${emp.id}__${dk}`);
      if (mid) {
        const name = mid.startsWith('UNKNOWN::')
          ? mid.replace('UNKNOWN::', '')
          : (moduleNameById.get(mid) ?? mid);
        const color = mid.startsWith('UNKNOWN::')
          ? MODULE_UNKNOWN_COLOR
          : (moduleColorById.get(mid) ?? MODULE_UNKNOWN_COLOR);
        return { type: 'work', moduleCode: name, moduleColor: color, operationId: '', isToday } as GanttCell;
      }
      return { type: 'empty', isToday } as GanttCell;
    });
  });

  return {
    employees, dates,
    cutoffDate: null,         // real Schedule has no cut-off; Python ignores it
    todayDate, cells,
    modules: ganttModules,
    planStart, planEnd,
  };
}
