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
  const assignmentList = ((data.assignment_list ?? []) as unknown[]).map(parseAssignment);

  return {
    planRange: {
      startDate: normalizeDate(planRange.start_date ?? ''),
      endDate: normalizeDate(planRange.end_date ?? ''),
    },
    workflowTaskList,
    assignmentList,
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
  };
}

// ── Schedule YAML stringify ──────────────────────────────────────────────────

export function stringifyScheduleYaml(data: ScheduleData): string {
  const out = {
    schedule: {
      plan_range: {
        start_date: data.planRange.startDate,
        end_date: data.planRange.endDate,
      },
      workflow_task_list: data.workflowTaskList.map(wt => {
        if (wt.phaseTaskList.length === 0) {
          return {
            id: wt.id,
            name: wt.name,
            workflow: wt.workflow,
            ...(wt.region !== undefined ? { region: wt.region } : {}),
            ...(wt.colorCode !== undefined ? { color_code: wt.colorCode } : {}),
          };
        }
        return {
          id: wt.id,
          name: wt.name,
          description: wt.description,
          workflow: wt.workflow,
          fab: wt.fab,
          phase_task_list: wt.phaseTaskList.map(pt => ({
            id: pt.id,
            name: pt.name,
            description: pt.description,
            phase: pt.phase,
            start_date: pt.startDate,
            end_date: pt.endDate,
            operation_task_list: pt.operationTaskList.map(ot => ({
              id: ot.id,
              name: ot.name,
              description: ot.description,
              operation: ot.operation,
              workload_hours: ot.workloadHours,
              color_code: ot.colorCode ?? '',
            })),
          })),
        };
      }),
      assignment_list: data.assignmentList.map(a => ({
        worker: a.worker,
        operation_task: a.operationTask,
        start_date: a.startDate,
        end_date: a.endDate,
        work_date_list: a.workDateList.map(w => ({ date: w.date, hour: w.hour })),
        plan_flexibility: a.planFlexibility,
      })),
    },
  };
  return yaml.dump(out, { lineWidth: -1 });
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
  return {
    id: String(r.id ?? ''),
    name: r.name as string | undefined,
    description: r.description as string | undefined,
    workerCompany: r.worker_company as string | undefined,
    isManager: Boolean(r.is_manager ?? false),
    skillMap: (r.skill_map as Record<string, number>) ?? {},
    unavailableDates: parseUnavailableDates((r.unavailable_dates as unknown[]) ?? []),
    definition: r.definition as string | undefined,
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
  const out = {
    environment: {
      workflow_list: config.workflowList.map(wf => ({
        id: wf.id,
        name: wf.name,
        phase_list: wf.phaseList.map(ph => ({
          id: ph.id,
          name: ph.name,
          operation_list: ph.operationList.map(op => ({
            id: op.id,
            name: op.name,
            work_hours: op.workHours,
            min_worker_num: op.minWorkerNum,
            max_worker_num: op.maxWorkerNum,
          })),
        })),
      })),
      fab_list: config.fabList.map(f => ({
        id: f.id,
        name: f.name,
        region: f.region,
        customer_company: f.customerCompany,
      })),
      region_list: config.regionList.map(r => ({
        id: r.id,
        name: r.name,
      })),
      customer_company_list: config.customerCompanyList.map(c => ({
        id: c.id,
        name: c.name,
      })),
      worker_company_list: config.workerCompanyList.map(wc => ({
        id: wc.id,
        name: wc.name,
        annual_overtime_limit: wc.annualOvertimeLimit,
        monthly_overtime_limit: wc.monthlyOvertimeLimit,
      })),
      worker_list: config.workerList.map(w => ({
        id: w.id,
        name: w.name,
        worker_company: w.workerCompany,
        is_manager: w.isManager,
        skill_map: w.skillMap,
        unavailable_dates: w.unavailableDates.map(entry => {
          const out: Record<string, unknown> = {};
          if (entry.weekly) out.weekly = { weekdays: entry.weekly.weekdays };
          if (entry.single) out.single = { days: entry.single.days };
          return out;
        }),
        ...(w.definition !== undefined ? { definition: w.definition } : {}),
      })),
      transite_day_map: config.transiteDayMap.map(t => ({
        from: t.from,
        to: t.to,
        days: t.days,
      })),
    },
  };
  return yaml.dump(out, { lineWidth: -1 });
}