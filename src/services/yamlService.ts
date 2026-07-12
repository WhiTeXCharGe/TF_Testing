import * as yaml from 'js-yaml';
import { ScheduleData, Assignment, WorkflowTask, PhaseTask, OperationTask, WorkDate, PlanFlexibility } from '../types/schedule';
import { EnvConfig, Worker, WorkerCompany, Fab, Region, CustomerCompany, Workflow, Phase, Operation, UnavailableDateEntry, TransiteDayMap } from '../types/envConfig';
import { HOURS_PER_DAY } from '../config/appConfig';
import { normalizeDate } from '../utils/dateUtils';

// ── Schedule YAML ────────────────────────────────────────────────────────────

export function parseScheduleYaml(raw: string): ScheduleData {
  const parsed = yaml.load(raw) as Record<string, unknown>;
  // Timefold wraps output under "schedule:" key; handle both forms
  const data = (parsed.schedule ?? parsed) as Record<string, unknown>;
  return normalizeSchedule(data);
}

function normalizeSchedule(data: Record<string, unknown>): ScheduleData {
  const planRange = data.plan_range as Record<string, string> ?? {};
  const workflowTaskList = ((data.workflow_task_list ?? []) as unknown[]).map(parseWorkflowTask);
  // misc_task_list items are converted to WorkflowTask with empty phaseTaskList
  const miscAsWorkflow = ((data.misc_task_list ?? []) as unknown[]).map(parseMiscTask);
  const assignmentList = ((data.assignment_list ?? []) as unknown[]).map(parseAssignment);

  return {
    planRange: {
      startDate: normalizeDate(planRange.start_date ?? ''),
      endDate: normalizeDate(planRange.end_date ?? ''),
    },
    workflowTaskList: [...workflowTaskList, ...miscAsWorkflow],
    assignmentList,
  };
}

function parseMiscTask(raw: unknown): WorkflowTask {
  const r = raw as Record<string, unknown>;
  return {
    id: String(r.id ?? ''),
    name: r.name as string | undefined,
    description: r.description as string | undefined,
    workflow: String(r.workflow ?? ''),
    fab: undefined,
    region: r.region as string | undefined,
    colorCode: r.color_code ? String(r.color_code) : undefined,
    phaseTaskList: [], // empty = marker for misc
  };
}

function parseWorkflowTask(raw: unknown): WorkflowTask {
  const r = raw as Record<string, unknown>;
  return {
    id: String(r.id ?? ''),
    name: r.name as string | undefined,
    description: r.description as string | undefined,
    workflow: String(r.workflow ?? ''),
    fab: r.fab ? String(r.fab) : undefined,
    region: r.region as string | undefined,
    colorCode: r.color_code ? String(r.color_code) : undefined,
    phaseTaskList: ((r.phase_task_list ?? []) as unknown[]).map(parsePhaseTask),
  };
}

function parsePhaseTask(raw: unknown): PhaseTask {
  const r = raw as Record<string, unknown>;
  return {
    id: String(r.id ?? ''),
    name: r.name as string | undefined,
    description: r.description as string | undefined,
    phase: String(r.phase ?? ''),
    startDate: normalizeDate(r.start_date as string ?? ''),
    endDate: normalizeDate(r.end_date as string ?? ''),
    operationTaskList: ((r.operation_task_list ?? []) as unknown[]).map(parseOperationTask),
  };
}

function parseOperationTask(raw: unknown): OperationTask {
  const r = raw as Record<string, unknown>;
  // Support workload_hours (Timefold output) OR workload_days * H (older YAMLs)
  const workloadHours =
    r.workload_hours != null
      ? Number(r.workload_hours)
      : (Number(r.workload_days ?? 0) * HOURS_PER_DAY);

  return {
    id: String(r.id ?? ''),
    name: r.name as string | undefined,
    description: r.description as string | undefined,
    operation: String(r.operation ?? ''),
    workloadHours,
    recommendsWorkerMin:
      r.recommends_worker_min != null ? Number(r.recommends_worker_min)
      : r.recommendsWorkerMin != null ? Number(r.recommendsWorkerMin)
      : undefined,
    recommendsWorkerMax:
      r.recommends_worker_max != null ? Number(r.recommends_worker_max)
      : r.recommendsWorkerMax != null ? Number(r.recommendsWorkerMax)
      : undefined,
    colorCode: r.color_code ? String(r.color_code) : undefined,
  };
}

function parseAssignment(raw: unknown): Assignment {
  const r = raw as Record<string, unknown>;
  // Tolerate both "work_date_list" and "work_date_lsit" (webapp typo)
  const rawList = (r.work_date_list ?? r.work_date_lsit ?? []) as unknown[];
  const workDateList: WorkDate[] = rawList.map((w) => {
    const wd = w as Record<string, unknown>;
    return { date: normalizeDate(wd.date as string ?? ''), hour: Number(wd.hour ?? 0) };
  });

  return {
    worker: String(r.worker ?? ''),
    operationTask: String(r.operation_task ?? ''),
    startDate: normalizeDate(r.start_date as string ?? ''),
    endDate: normalizeDate(r.end_date as string ?? ''),
    workDateList,
    planFlexibility: (r.plan_flexibility as PlanFlexibility) ?? 'Flexible',
    description: r.description as string | undefined,
  };
}

// ── YAML format helpers ──────────────────────────────────────────────────────

// Convert internal date 2025-09-01 → YAML date 2025/09/01
function toYD(d: string): string { return d.replace(/-/g, '/'); }

// Inline flow array: [a, b, c]
function flowArr(items: (string | number)[]): string {
  return '[' + items.join(', ') + ']';
}

// Inline flow map with numeric values: {k: v, k2: v2}
function flowNumMap(obj: Record<string, number>): string {
  return '{' + Object.entries(obj).map(([k, v]) => `${k}: ${v}`).join(', ') + '}';
}

// YAML scalar string. Returns '' for undefined/null (YAML null), '""' for empty string, or the value.
function ys(v: string | undefined | null): string {
  if (v === undefined || v === null) return '';
  if (v === '') return '""';
  // Quote if value is a YAML boolean/null keyword or starts with special chars
  if (/^(true|false|null|yes|no|on|off)$/i.test(v)) return `"${v}"`;
  if (/^[\[{&*!|>'"%@`#:]/.test(v) || v.includes(': ') || v.includes('\n')) {
    return '"' + v.replace(/\\/g, '\\\\').replace(/"/g, '\\"') + '"';
  }
  return v;
}

function emitUnavailDates(lines: string[], dates: UnavailableDateEntry[], indent: string): void {
  if (!dates || dates.length === 0) {
    lines.push(`${indent}unavailable_dates: []`);
    return;
  }
  lines.push(`${indent}unavailable_dates:`);
  for (const ud of dates) {
    if (ud.weekly) {
      lines.push(`${indent}- weekly:`);
      lines.push(`${indent}    weekdays: ${flowArr(ud.weekly.weekdays)}`);
    }
    if (ud.single) {
      lines.push(`${indent}- single:`);
      lines.push(`${indent}    days:`);
      for (const d of ud.single.days) {
        lines.push(`${indent}    - ${toYD(d)}`);
      }
    }
  }
}

// ── Schedule YAML stringify ──────────────────────────────────────────────────

export function stringifyScheduleYaml(data: ScheduleData): string {
  const normalTasks = data.workflowTaskList.filter(wt => wt.phaseTaskList.length > 0);
  const miscTasks   = data.workflowTaskList.filter(wt => wt.phaseTaskList.length === 0);
  const L: string[] = [];
  const p = (s: string) => L.push(s);

  p('schedule:');
  p('  plan_range:');
  p(`    start_date: ${toYD(data.planRange.startDate)}`);
  p(`    end_date: ${toYD(data.planRange.endDate)}`);

  p('  workflow_task_list:');
  for (const wt of normalTasks) {
    p(`  - id: ${wt.id}`);
    if (wt.name !== undefined) p(`    name: ${ys(wt.name)}`);
    p(`    description: ${ys(wt.description)}`);
    p(`    workflow: ${wt.workflow}`);
    if (wt.fab !== undefined) p(`    fab: ${wt.fab}`);
    if (wt.region !== undefined) p(`    region: ${wt.region}`);
    if (wt.colorCode !== undefined) p(`    color_code: ${wt.colorCode}`);
    p(`    phase_task_list:`);
    for (const pt of wt.phaseTaskList) {
      p(`    - id: ${pt.id}`);
      if (pt.name !== undefined) p(`      name: ${ys(pt.name)}`);
      p(`      description: ${ys(pt.description)}`);
      p(`      phase: ${pt.phase}`);
      p(`      start_date: ${toYD(pt.startDate)}`);
      p(`      end_date: ${toYD(pt.endDate)}`);
      p(`      operation_task_list:`);
      for (const ot of pt.operationTaskList) {
        p(`      - id: ${ot.id}`);
        if (ot.name !== undefined) p(`        name: ${ys(ot.name)}`);
        p(`        description: ${ys(ot.description)}`);
        p(`        operation: ${ot.operation}`);
        p(`        workload_hours: ${ot.workloadHours}`);
        p(`        color_code: ${ot.colorCode ?? ''}`);
      }
    }
  }

  p('  misc_task_list:');
  for (const wt of miscTasks) {
    p(`  - id: ${wt.id}`);
    if (wt.name !== undefined) p(`    name: ${ys(wt.name)}`);
    p(`    description: ${ys(wt.description)}`);
    p(`    workflow: ${wt.workflow}`);
    if (wt.region !== undefined) p(`    region: ${wt.region}`);
    if (wt.colorCode !== undefined) p(`    color_code: ${wt.colorCode}`);
  }

  p('  assignment_list:');
  for (const a of data.assignmentList) {
    p(`  - worker: ${a.worker}`);
    p(`    operation_task: ${a.operationTask}`);
    p(`    start_date: ${toYD(a.startDate)}`);
    p(`    end_date: ${toYD(a.endDate)}`);
    p(`    work_date_list:`);
    for (const w of a.workDateList) {
      p(`    - date: ${toYD(w.date)}`);
      p(`      hour: ${w.hour}`);
    }
    p(`    plan_flexibility: ${a.planFlexibility}`);
    p(`    description: ${ys(a.description)}`);
  }

  return L.join('\n') + '\n';
}

// ── EnvConfig YAML ───────────────────────────────────────────────────────────

export function parseEnvConfigYaml(raw: string): EnvConfig {
  const parsed = yaml.load(raw) as Record<string, unknown>;
  // Support "environment:" wrapper or bare
  const data = (parsed.environment ?? parsed) as Record<string, unknown>;
  return normalizeEnvConfig(data);
}

function normalizeEnvConfig(data: Record<string, unknown>): EnvConfig {
  return {
    workflowList: ((data.workflow_list ?? []) as unknown[]).map(parseWorkflow),
    fabList: ((data.fab_list ?? []) as unknown[]).map(parseFab),
    regionList: ((data.region_list ?? []) as unknown[]).map(parseRegion),
    customerCompanyList: ((data.customer_company_list ?? []) as unknown[]).map(parseCustomerCompany),
    workerCompanyList: ((data.worker_company_list ?? []) as unknown[]).map(parseWorkerCompany),
    workerList: ((data.worker_list ?? []) as unknown[]).map(parseWorker),
    transiteDayMap: ((data.transite_day_map ?? []) as unknown[]).map(parseTransiteDay),
  };
}

function parseUnavailableDates(raw: unknown[]): UnavailableDateEntry[] {
  return raw.map(entry => {
    const e = entry as Record<string, unknown>;
    const result: UnavailableDateEntry = {};
    if (e.weekly) {
      const w = e.weekly as Record<string, unknown>;
      result.weekly = { weekdays: (w.weekdays as string[]) ?? [] };
    }
    if (e.single) {
      const s = e.single as Record<string, unknown>;
      result.single = { days: (s.days as string[]) ?? [] };
    }
    return result;
  });
}

function parseWorker(raw: unknown): Worker {
  const r = raw as Record<string, unknown>;
  // Support both structured description object and legacy definition string
  let description: Worker['description'];
  if (r.description && typeof r.description === 'object') {
    description = r.description as Worker['description'];
  } else if (typeof r.definition === 'string' && r.definition) {
    description = { '備考': r.definition };
  }
  return {
    id: String(r.id ?? ''),
    name: r.name as string | undefined,
    description,
    workerCompany: r.worker_company as string | undefined,
    isManager: Boolean(r.is_manager ?? false),
    skillMap: (r.skill_map as Record<string, number>) ?? {},
    workerTypeByOperation: r.worker_type_by_operation as Record<string, string> | undefined,
    fabSuitabilityMap: r.fab_suitability_map as Worker['fabSuitabilityMap'],
    affinity: r.affinity as string[] | undefined,
    unavailableDates: parseUnavailableDates((r.unavailable_dates as unknown[]) ?? []),
  };
}

function parseWorkerCompany(raw: unknown): WorkerCompany {
  const r = raw as Record<string, unknown>;
  return {
    id: String(r.id ?? ''),
    name: r.name as string | undefined,
    annualOvertimeLimit: Number(r.annual_overtime_limit ?? 0),
    monthlyOvertimeLimit: Number(r.monthly_overtime_limit ?? 0),
    unavailableDates: parseUnavailableDates((r.unavailable_dates as unknown[]) ?? []),
  };
}

function parseFab(raw: unknown): Fab {
  const r = raw as Record<string, unknown>;
  return {
    id: String(r.id ?? ''),
    name: r.name as string | undefined,
    region: r.region as string | undefined,
    customerCompany: r.customer_company as string | undefined,
    unavailableDates: parseUnavailableDates((r.unavailable_dates as unknown[]) ?? []),
  };
}

function parseRegion(raw: unknown): Region {
  const r = raw as Record<string, unknown>;
  return {
    id: String(r.id ?? ''),
    name: r.name as string | undefined,
    maxStayOn: Number(r.max_stay_on ?? 0),
    maxAnnualStay: Number(r.max_annual_stay ?? 0),
    stayOffInterval: Number(r.stay_off_interval ?? 0),
    unavailableDates: parseUnavailableDates((r.unavailable_dates as unknown[]) ?? []),
  };
}

function parseCustomerCompany(raw: unknown): CustomerCompany {
  const r = raw as Record<string, unknown>;
  return {
    id: String(r.id ?? ''),
    name: r.name as string | undefined,
    unavailableDates: parseUnavailableDates((r.unavailable_dates as unknown[]) ?? []),
  };
}

function parseWorkflow(raw: unknown): Workflow {
  const r = raw as Record<string, unknown>;
  return {
    id: String(r.id ?? ''),
    name: r.name as string | undefined,
    phaseList: ((r.phase_list ?? []) as unknown[]).map(parsePhase),
  };
}

function parsePhase(raw: unknown): Phase {
  const r = raw as Record<string, unknown>;
  return {
    id: String(r.id ?? ''),
    name: r.name as string | undefined,
    operationList: ((r.operation_list ?? []) as unknown[]).map(parseOperation),
  };
}

function parseOperation(raw: unknown): Operation {
  const r = raw as Record<string, unknown>;
  return {
    id: String(r.id ?? ''),
    name: r.name as string | undefined,
    workHours: (r.work_hours as number[]) ?? [],
    workloadHours: r.workload_hours != null ? Number(r.workload_hours) : undefined,
    minWorkerNum: Number(r.min_worker_num ?? 0),
    maxWorkerNum: Number(r.max_worker_num ?? 0),
  };
}

function parseTransiteDay(raw: unknown): TransiteDayMap {
  const r = raw as Record<string, unknown>;
  return {
    from: String(r.from ?? ''),
    to: String(r.to ?? ''),
    days: Number(r.days ?? 0),
  };
}

// ── EnvConfig YAML stringify ─────────────────────────────────────────────────

export function stringifyEnvConfigYaml(config: EnvConfig): string {
  const L: string[] = [];
  const p = (s: string) => L.push(s);

  p('environment:');

  // workflow_list
  p('  workflow_list:');
  for (const wf of config.workflowList) {
    p(`  - id: ${wf.id}`);
    p(`    name: ${ys(wf.name)}`);
    p(`    phase_list:`);
    for (const ph of wf.phaseList) {
      p(`    - id: ${ph.id}`);
      p(`      name: ${ys(ph.name)}`);
      p(`      operation_list:`);
      for (const op of ph.operationList) {
        p(`      - id: ${op.id}`);
        p(`        name: ${ys(op.name)}`);
        p(`        work_hours: ${flowArr(op.workHours)}`);
        p(`        workload_hours: ${op.workloadHours ?? 0}`);
        p(`        min_worker_num: ${op.minWorkerNum}`);
        p(`        max_worker_num: ${op.maxWorkerNum}`);
      }
    }
  }

  // fab_list
  p('  fab_list:');
  for (const f of config.fabList) {
    p(`  - id: ${f.id}`);
    p(`    name: ${ys(f.name)}`);
    p(`    region: ${f.region}`);
    p(`    customer_company: ${f.customerCompany}`);
  }

  // region_list
  p('  region_list:');
  for (const r of config.regionList) {
    p(`  - id: ${r.id}`);
    p(`    name: ${ys(r.name)}`);
  }

  // customer_company_list
  p('  customer_company_list:');
  for (const c of config.customerCompanyList) {
    p(`  - id: ${c.id}`);
    p(`    name: ${ys(c.name)}`);
  }

  // worker_company_list
  p('  worker_company_list:');
  for (const wc of config.workerCompanyList) {
    p(`  - id: ${wc.id}`);
    p(`    name: ${ys(wc.name)}`);
    p(`    annual_overtime_limit: ${wc.annualOvertimeLimit}`);
    p(`    monthly_overtime_limit: ${wc.monthlyOvertimeLimit}`);
  }

  // transite_day_map
  p('  transite_day_map:');
  for (const t of config.transiteDayMap) {
    p(`  - from: ${t.from}`);
    p(`    to: ${t.to}`);
    p(`    days: ${t.days}`);
  }

  // worker_list
  p('  worker_list:');
  for (const w of config.workerList) {
    p(`  - id: ${w.id}`);
    p(`    name: ${ys(w.name)}`);
    p(`    worker_company: ${w.workerCompany}`);
    p(`    is_manager: ${w.isManager}`);
    // skill_map inline
    if (w.skillMap && Object.keys(w.skillMap).length > 0) {
      p(`    skill_map: ${flowNumMap(w.skillMap as Record<string, number>)}`);
    } else {
      p(`    skill_map: {}`);
    }
    if (w.workerTypeByOperation !== undefined) {
      if (Object.keys(w.workerTypeByOperation).length > 0) {
        const wto = w.workerTypeByOperation;
        p(`    worker_type_by_operation: {${Object.entries(wto).map(([k, v]) => `${k}: ${ys(v)}`).join(', ')}}`);
      } else {
        p(`    worker_type_by_operation: {}`);
      }
    }
    if (w.fabSuitabilityMap !== undefined) {
      if (w.fabSuitabilityMap.length > 0) {
        p(`    fab_suitability_map:`);
        for (const entry of w.fabSuitabilityMap) {
          p(`    - kind: ${entry.kind}`);
          p(`      suitability: ${flowNumMap(entry.suitability)}`);
        }
      } else {
        p(`    fab_suitability_map: []`);
      }
    }
    emitUnavailDates(L, w.unavailableDates, '    ');
    if (w.affinity !== undefined) {
      if (w.affinity.length > 0) {
        p(`    affinity: ${flowArr(w.affinity)}`);
      } else {
        p(`    affinity: []`);
      }
    }
    if (w.description !== undefined) {
      const d = w.description as Record<string, string | undefined>;
      p(`    description:`);
      p(`      業務形態: ${ys(d['業務形態'])}`);
      p(`      VISA: ${ys(d['VISA'])}`);
      p(`      海外運転: ${ys(d['海外運転'])}`);
      if (d['備考'] !== undefined) p(`      備考: ${ys(d['備考'])}`);
    }
  }

  return L.join('\n') + '\n';
}