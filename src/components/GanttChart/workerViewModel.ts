import { EnvConfig, Worker } from '../../types/envConfig';
import { Assignment, PlanFlexibility, ScheduleData } from '../../types/schedule';
import { generateDateRange } from '../../utils/dateUtils';
import { UI } from '../../config/uiText';

export interface WorkerMetaInfo {
  id: string;
  company: string;
  name: string;
  manager: string;
  remarks: string;
  workType: string;
  assignedDuties: string;
  visa: string;
  overseasDriving: string;
}

export interface WorkerDayCell {
  kind: 'work' | 'unavailable' | 'empty';
  moduleName?: string;
  taskName?: string;
  color?: string;
  textColor?: string;
  planFlexibility?: PlanFlexibility;
}

export interface WorkerSegment {
  kind: 'work' | 'unavailable';
  startIndex: number;
  endIndex: number;
  label: string;
  color: string;
  textColor: string;
  assignmentIndex?: number;
  isMisc?: boolean;
  planFlexibility?: PlanFlexibility;
}

export interface FlightStint {
  regionId: string;
  firstWorkDate: string;
  lastWorkDate: string;
  flightInDate: string;   // firstWorkDate - 1 day
  flightOutDate: string;  // lastWorkDate + 1 day
}

export interface WorkerTimelineRow {
  workerId: string;
  meta: WorkerMetaInfo;
  dayCells: WorkerDayCell[];
  segments: WorkerSegment[];
  flightStints: FlightStint[];
}

export interface HeaderMonthGroup {
  label: string;
  startIndex: number;
  span: number;
}

export interface WorkerTimelineModel {
  rows: WorkerTimelineRow[];
  monthGroups: HeaderMonthGroup[];
  dateWorkOptions: Record<string, string[]>;
  regionColorMap: Map<string, string>;
  regionNameMap: Map<string, string>;
}

type UnavailableDateEntryLike = {
  date?: string;
  single?: { days?: string[] };
  weekly?: { weekdays?: string[] };
};

const PB_WORKFLOW_ID = 'wf_personal_business';
const PB_COLOR = '#898989';
const PB_TEXT_COLOR = '#ffffff';
const UNAVAILABLE_COLOR = '#ff0000';
const UNAVAILABLE_TEXT_COLOR = '#000000';

const REGION_PALETTE = [
  '#4FC3F7', '#81C784', '#FFB74D', '#F06292', '#BA68C8',
  '#4DB6AC', '#FF8A65', '#90A4AE', '#A1887F', '#DCE775',
  '#7986CB', '#4CAF50', '#FF7043', '#26C6DA', '#AB47BC',
];

const MODULE_PALETTE = [
  '#FFE599', '#FFD966', '#F9CB9C', '#F6B26B', '#FCE5CD',
  '#B6D7A8', '#93C47D', '#6AA84F', '#D9EAD3', '#E2EFDA',
  '#9FC5E8', '#6FA8DC', '#3D85C6', '#A4C2F4', '#CFE2F3',
  '#B4A7D6', '#8E7CC3', '#674EA7', '#D9D2E9', '#EAD1DC',
  '#A2C4C9', '#76A5AF', '#45818E', '#D0E0E3', '#DDEBF7',
  '#C27BA0', '#D5A6BD', '#E6B8AF', '#F8CBAD', '#FFF2CC',
];

const UNKNOWN_COLOR = '#FFF2CC';
const BLACK = '#000000';

function normalizeModuleCode(name: string): string {
  if (!name) return '';
  return name.trim().split('_', 1)[0] ?? '';
}

function addDays(dateStr: string, n: number): string {
  const d = new Date(`${dateStr}T00:00:00`);
  d.setDate(d.getDate() + n);
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, '0');
  const dd = String(d.getDate()).padStart(2, '0');
  return `${y}-${m}-${dd}`;
}

function daysBetween(a: string, b: string): number {
  return Math.round((new Date(`${b}T00:00:00`).getTime() - new Date(`${a}T00:00:00`).getTime()) / 86400000);
}

function computeFlightStintsMultiRegion(
  assignments: Array<{ startDate: string; endDate: string; regionId: string }>,
  regionStayOff: Map<string, number>,
): FlightStint[] {
  if (assignments.length === 0) return [];
  const sorted = [...assignments].sort((a, b) => a.startDate.localeCompare(b.startDate));

  const stints: FlightStint[] = [];
  let { regionId: curRegion, startDate: curFirst, endDate: curLast } = sorted[0];

  for (let i = 1; i < sorted.length; i++) {
    const { startDate, endDate, regionId } = sorted[i];
    if (regionId === curRegion) {
      const gap = daysBetween(curLast, startDate) - 1;
      const interval = regionStayOff.get(curRegion) ?? 0;
      if (gap <= interval) {
        if (endDate > curLast) curLast = endDate;
        continue;
      }
    }
    stints.push({
      regionId: curRegion,
      firstWorkDate: curFirst,
      lastWorkDate: curLast,
      flightInDate: addDays(curFirst, -1),
      flightOutDate: addDays(curLast, 1),
    });
    curRegion = regionId;
    curFirst = startDate;
    curLast = endDate;
  }
  stints.push({
    regionId: curRegion,
    firstWorkDate: curFirst,
    lastWorkDate: curLast,
    flightInDate: addDays(curFirst, -1),
    flightOutDate: addDays(curLast, 1),
  });
  return stints;
}

function weekdayToInt(dayText: string): number | null {
  const key = dayText.trim().toLowerCase();
  const map: Record<string, number> = {
    mon: 1, monday: 1,
    tue: 2, tues: 2, tuesday: 2,
    wed: 3, weds: 3, wednesday: 3,
    thu: 4, thur: 4, thurs: 4, thursday: 4,
    fri: 5, friday: 5,
    sat: 6, saturday: 6,
    sun: 0, sunday: 0,
  };
  return map[key] ?? null;
}

function parseUnavailableDates(
  raw: unknown,
  planStart: string,
  planEnd: string,
): Set<string> {
  const result = new Set<string>();
  if (!raw) return result;

  const addDate = (dateText: unknown) => {
    if (typeof dateText !== 'string') return;
    const normalized = dateText.replace(/\//g, '-');
    if (normalized >= planStart && normalized <= planEnd) {
      result.add(normalized);
    }
  };

  const list: unknown[] = Array.isArray(raw) ? raw : [raw];
  const weekly = new Set<number>();

  for (const item of list) {
    if (!item) continue;
    if (typeof item === 'string') { addDate(item); continue; }
    if (typeof item !== 'object') continue;
    const entry = item as UnavailableDateEntryLike;
    if (entry.date) addDate(entry.date);
    if (entry.single?.days) entry.single.days.forEach(addDate);
    if (entry.weekly?.weekdays) {
      for (const wd of entry.weekly.weekdays) {
        const weekday = weekdayToInt(wd);
        if (weekday !== null) weekly.add(weekday);
      }
    }
  }

  if (weekly.size > 0) {
    const d = new Date(`${planStart}T00:00:00`);
    const end = new Date(`${planEnd}T00:00:00`);
    while (d <= end) {
      if (weekly.has(d.getDay())) {
        const y = d.getFullYear();
        const m = String(d.getMonth() + 1).padStart(2, '0');
        const dd = String(d.getDate()).padStart(2, '0');
        result.add(`${y}-${m}-${dd}`);
      }
      d.setDate(d.getDate() + 1);
    }
  }

  return result;
}

function buildOpTaskToModuleMap(schedule: ScheduleData): Map<string, string> {
  const map = new Map<string, string>();
  for (const workflowTask of schedule.workflowTaskList) {
    if (workflowTask.phaseTaskList.length === 0) {
      // misc task: operationTask id IS the workflowTask id
      map.set(workflowTask.id, workflowTask.id);
      continue;
    }
    for (const phaseTask of workflowTask.phaseTaskList) {
      for (const operationTask of phaseTask.operationTaskList) {
        map.set(operationTask.id, workflowTask.id);
      }
    }
  }
  return map;
}

// operationTask id → task name. Misc tasks have no OperationTask, so they're absent here.
function buildOpTaskNameMap(schedule: ScheduleData): Map<string, string> {
  const map = new Map<string, string>();
  for (const workflowTask of schedule.workflowTaskList) {
    for (const phaseTask of workflowTask.phaseTaskList) {
      for (const operationTask of phaseTask.operationTaskList) {
        map.set(operationTask.id, operationTask.name ?? operationTask.operation ?? operationTask.id);
      }
    }
  }
  return map;
}

export function buildOpTaskColorMap(schedule: ScheduleData): Map<string, string> {
  const result = new Map<string, string>();
  const byBaseCode = new Map<string, string>();
  let colorIdx = 0;

  for (const workflowTask of schedule.workflowTaskList) {
    if (workflowTask.phaseTaskList.length === 0) {
      const color = workflowTask.colorCode
        ? `#${workflowTask.colorCode}`
        : MODULE_PALETTE[colorIdx++ % MODULE_PALETTE.length];
      result.set(workflowTask.id, color);
      continue;
    }
    const baseCode = normalizeModuleCode(workflowTask.name ?? '');
    for (const phaseTask of workflowTask.phaseTaskList) {
      for (const operationTask of phaseTask.operationTaskList) {
        if (operationTask.colorCode) {
          result.set(operationTask.id, `#${operationTask.colorCode}`);
        } else {
          if (!byBaseCode.has(baseCode)) {
            byBaseCode.set(baseCode, MODULE_PALETTE[colorIdx++ % MODULE_PALETTE.length]);
          }
          result.set(operationTask.id, byBaseCode.get(baseCode) ?? UNKNOWN_COLOR);
        }
      }
    }
  }

  return result;
}

function buildMonthGroups(dates: string[]): HeaderMonthGroup[] {
  const groups: HeaderMonthGroup[] = [];
  let i = 0;
  while (i < dates.length) {
    const [year, month] = dates[i].split('-');
    let j = i;
    while (j + 1 < dates.length) {
      const [nextYear, nextMonth] = dates[j + 1].split('-');
      if (nextYear !== year || nextMonth !== month) break;
      j += 1;
    }
    groups.push({ label: UI.monthLabel(Number(month)), startIndex: i, span: j - i + 1 });
    i = j + 1;
  }
  return groups;
}

function getWorkerCompanyName(worker: Worker, envConfig: EnvConfig): string {
  const companyId = worker.workerCompany ?? '';
  const company = envConfig.workerCompanyList.find(c => c.id === companyId);
  return company?.name ?? companyId;
}

function collectDayAssignments(assignment: Assignment): string[] {
  if (assignment.startDate && assignment.endDate && assignment.startDate <= assignment.endDate) {
    return generateDateRange(assignment.startDate, assignment.endDate);
  }
  return assignment.workDateList.map(wd => wd.date);
}

export function buildWorkerTimelineModel(
  envConfig: EnvConfig,
  schedule: ScheduleData,
  dates: string[],
  today: string,
): WorkerTimelineModel {
  const monthGroups = buildMonthGroups(dates);
  const opTaskToModule = buildOpTaskToModuleMap(schedule);
  const opTaskNameMap = buildOpTaskNameMap(schedule);
  const opTaskColorMap = buildOpTaskColorMap(schedule);
  const moduleById = new Map(schedule.workflowTaskList.map(w => [w.id, w]));

  // Build region lookup maps
  const fabRegionMap = new Map(envConfig.fabList.map(f => [f.id, f.region ?? null]));
  const regionColorMap = new Map(
    envConfig.regionList.map((r, i) => [r.id, REGION_PALETTE[i % REGION_PALETTE.length]]),
  );
  const regionNameMap = new Map(envConfig.regionList.map(r => [r.id, r.name ?? r.id]));
  const regionStayOff = new Map(envConfig.regionList.map(r => [r.id, r.stayOffInterval ?? 0]));

  // opTask id → regionId (null if no region)
  const opTaskRegionMap = new Map<string, string | null>();
  for (const wt of schedule.workflowTaskList) {
    if (wt.phaseTaskList.length === 0) {
      // misc task: region from workflowTask.region directly
      opTaskRegionMap.set(wt.id, wt.region ?? null);
    } else {
      const regionId = wt.fab ? (fabRegionMap.get(wt.fab) ?? null) : null;
      for (const pt of wt.phaseTaskList) {
        for (const ot of pt.operationTaskList) {
          opTaskRegionMap.set(ot.id, regionId);
        }
      }
    }
  }
  const miscTaskIds = new Set(
    schedule.workflowTaskList.filter(wt => wt.phaseTaskList.length === 0).map(wt => wt.id),
  );

  const dateIndex = new Map(dates.map((d, i) => [d, i]));

  const workerById = new Map(envConfig.workerList.map(worker => [worker.id, worker]));
  const workerDayAssignments = new Map<string, Map<string, { moduleName: string; taskName: string; color: string; textColor: string; assignmentIndex: number; isMisc: boolean; planFlexibility: PlanFlexibility }>>();
  const assignedWorkerIds = new Set<string>();

  // Per-worker region assignments for flight stint computation
  const workerRegionAssignments = new Map<string, Array<{ startDate: string; endDate: string; regionId: string }>>();

  for (const [assignmentIndex, assignment] of schedule.assignmentList.entries()) {
    const workerId = assignment.worker;
    const moduleId = opTaskToModule.get(assignment.operationTask) ?? `UNKNOWN::${assignment.operationTask}`;
    const moduleInfo = moduleById.get(moduleId);
    const isMisc = miscTaskIds.has(moduleId);

    const regionId = opTaskRegionMap.get(assignment.operationTask) ?? null;
    if (regionId) {
      if (!workerRegionAssignments.has(workerId)) workerRegionAssignments.set(workerId, []);
      workerRegionAssignments.get(workerId)!.push({
        startDate: assignment.startDate,
        endDate: assignment.endDate,
        regionId,
      });
    }

    let color = opTaskColorMap.get(assignment.operationTask) ?? UNKNOWN_COLOR;
    let textColor = BLACK;
    const moduleName = moduleInfo?.name ?? moduleId.replace('UNKNOWN::', '');
    const taskName = isMisc ? '' : (opTaskNameMap.get(assignment.operationTask) ?? '');

    if (moduleInfo?.workflow === PB_WORKFLOW_ID) {
      color = PB_COLOR;
      textColor = PB_TEXT_COLOR;
    }

    const days = collectDayAssignments(assignment);
    for (const day of days) {
      if (!dateIndex.has(day)) continue;
      if (!workerDayAssignments.has(workerId)) {
        workerDayAssignments.set(workerId, new Map());
      }
      const dayMap = workerDayAssignments.get(workerId);
      if (!dayMap) continue;
      if (!dayMap.has(day)) {
        dayMap.set(day, { moduleName, taskName, color, textColor, assignmentIndex, isMisc, planFlexibility: assignment.planFlexibility });
      }
      assignedWorkerIds.add(workerId);
    }
  }

  const workerOffDates = new Map<string, Set<string>>();
  const workersWithOff = new Set<string>();

  for (const worker of envConfig.workerList) {
    const off = parseUnavailableDates(
      worker.unavailableDates as unknown,
      schedule.planRange.startDate,
      schedule.planRange.endDate,
    );
    workerOffDates.set(worker.id, off);
    if (off.size > 0) workersWithOff.add(worker.id);
  }

  const includeIds = new Set<string>([...assignedWorkerIds, ...workersWithOff]);
  const sortedWorkerIds = [...includeIds].sort((a, b) => {
    const wa = workerById.get(a);
    const wb = workerById.get(b);
    const companyA = wa ? getWorkerCompanyName(wa, envConfig) : '';
    const companyB = wb ? getWorkerCompanyName(wb, envConfig) : '';
    if (companyA !== companyB) return companyA.localeCompare(companyB, 'ja');
    return (wa?.name ?? a).localeCompare(wb?.name ?? b, 'ja');
  });

  const dateWorkOptionSet: Record<string, Set<string>> = Object.fromEntries(
    dates.map(d => [d, new Set<string>()]),
  );

  const rows: WorkerTimelineRow[] = sortedWorkerIds.map(workerId => {
    const worker = workerById.get(workerId);
    const workerName = worker?.name ?? workerId;
    const company = worker ? getWorkerCompanyName(worker, envConfig) : '';
    const remarks = worker?.description?.['備考'] ?? '';
    const manager = worker?.isManager ? UI.workerGridManagerYes : '';
    const workType = worker?.description?.['業務形態'] ?? '';
    const visa = worker?.description?.['VISA'] ?? '';
    const overseasDriving = worker?.description?.['海外運転'] ?? '';
    const assignedDuties = worker?.skillMap
      ? (() => {
          // Find skill names from envConfig operation list
          const opNames: string[] = [];
          for (const wf of envConfig.workflowList) {
            for (const ph of wf.phaseList) {
              for (const op of ph.operationList) {
                if (worker.skillMap![op.id] !== undefined) {
                  opNames.push(op.name ?? op.id);
                }
              }
            }
          }
          return [...new Set(opNames)].join(', ');
        })()
      : '';

    const dayMap = workerDayAssignments.get(workerId) ?? new Map();
    const offSet = workerOffDates.get(workerId) ?? new Set<string>();
    const dayCells: WorkerDayCell[] = [];
    const dayAssignmentIndices: Array<{ index: number; isMisc: boolean } | undefined> = [];

    for (const day of dates) {
      if (offSet.has(day)) {
        dayCells.push({ kind: 'unavailable' });
        dayAssignmentIndices.push(undefined);
        continue;
      }
      const work = dayMap.get(day);
      if (!work) {
        dayCells.push({ kind: 'empty' });
        dayAssignmentIndices.push(undefined);
        continue;
      }
      dayCells.push({ kind: 'work', moduleName: work.moduleName, taskName: work.taskName, color: work.color, textColor: work.textColor, planFlexibility: work.planFlexibility });
      dayAssignmentIndices.push({ index: work.assignmentIndex, isMisc: work.isMisc });
      dateWorkOptionSet[day]?.add(work.moduleName);
    }

    const segments: WorkerSegment[] = [];
    let start = -1;
    let prev: WorkerDayCell | null = null;
    for (let i = 0; i <= dayCells.length; i += 1) {
      const current = i < dayCells.length ? dayCells[i] : null;
      const prevAssignIdx = i > 0 ? dayAssignmentIndices[i - 1]?.index : undefined;
      const currAssignIdx = i < dayCells.length ? dayAssignmentIndices[i]?.index : undefined;
      const sameAsPrev = prev && current && prev.kind === current.kind && prev.moduleName === current.moduleName
        && currAssignIdx === prevAssignIdx;

      if (prev && !sameAsPrev) {
        if (prev.kind === 'work') {
          const aInfo = dayAssignmentIndices[start];
          const label = prev.taskName ? `${prev.moduleName ?? ''} ${prev.taskName}` : (prev.moduleName ?? '');
          segments.push({
            kind: 'work',
            startIndex: start,
            endIndex: i - 1,
            label,
            color: prev.color ?? UNKNOWN_COLOR,
            textColor: prev.textColor ?? BLACK,
            assignmentIndex: aInfo?.index,
            isMisc: aInfo?.isMisc,
            planFlexibility: prev.planFlexibility,
          });
        } else if (prev.kind === 'unavailable') {
          segments.push({
            kind: 'unavailable',
            startIndex: start,
            endIndex: i - 1,
            label: '',
            color: UNAVAILABLE_COLOR,
            textColor: UNAVAILABLE_TEXT_COLOR,
          });
        }
      }

      if (!prev || !sameAsPrev) start = i;
      prev = current;
    }

    // Compute flight stints for this worker
    const regionAssignments = workerRegionAssignments.get(workerId) ?? [];
    const flightStints = computeFlightStintsMultiRegion(regionAssignments, regionStayOff);

    return {
      workerId,
      meta: { id: workerId, company, name: workerName, manager, remarks, workType, visa, overseasDriving, assignedDuties },
      dayCells,
      segments,
      flightStints,
    };
  });

  const dateWorkOptions: Record<string, string[]> = {};
  for (const d of dates) {
    dateWorkOptions[d] = [...(dateWorkOptionSet[d] ?? new Set<string>())].sort((a, b) => a.localeCompare(b, 'ja'));
  }

  return { rows, monthGroups, dateWorkOptions, regionColorMap, regionNameMap };
}
