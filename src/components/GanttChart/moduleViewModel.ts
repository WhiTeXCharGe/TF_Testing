import { EnvConfig, Operation, Worker } from '../../types/envConfig';
import { ScheduleData } from '../../types/schedule';
import { getColorForPhaseIndex } from '../../utils/colorUtils';

export interface HeaderMonthGroup {
  label: string;
  startIndex: number;
  span: number;
}

export interface ModuleWorkerSlot {
  assignmentIndex: number;
  workerId: string;
  workerName: string;
  companyName: string;
  startDate: string;
  endDate: string;
}

export interface ModuleTask {
  moduleId: string;
  phaseId: string;
  taskId: string;          // operationTask.id
  taskName: string;
  operationId: string;     // operationTask.operation
  minWorker: number;
  maxWorker: number;
  workloadHours: number;   // from operationTask.workloadHours or envConfig op.workHours[0]
  slots: ModuleWorkerSlot[];
  startDate: string | null; // earliest assigned worker start
  endDate: string | null;   // latest assigned worker end
  color: string;
  description?: string;
}

export interface ModulePhase {
  moduleId: string;
  phaseId: string;
  phaseName: string;
  planStartDate: string;    // 作業開始可能日 (phaseTask.startDate)
  planEndDate: string;      // 終了希望日   (phaseTask.endDate)
  barStartDate: string | null; // earliest worker start across all tasks
  barEndDate: string | null;   // latest   worker end   across all tasks
  workerCount: number;      // distinct assigned workers in phase
  tasks: ModuleTask[];
  color: string;
  description?: string;
}

export interface ModuleNode {
  moduleId: string;
  moduleName: string;       // 製番
  workflowName: string;     // 属性 — EnvConfig workflow name looked up via WorkflowTask.workflow
  fab: string | null;
  region: string | null;
  phases: ModulePhase[];
}

export interface ModuleViewModel {
  modules: ModuleNode[];
  monthGroups: HeaderMonthGroup[];
}

function minStr(a: string, b: string): string { return a < b ? a : b; }
function maxStr(a: string, b: string): string { return a > b ? a : b; }

function buildMonthGroups(dates: string[]): HeaderMonthGroup[] {
  const groups: HeaderMonthGroup[] = [];
  let i = 0;
  while (i < dates.length) {
    const [year, month] = dates[i].split('-');
    let j = i;
    while (j + 1 < dates.length) {
      const [ny, nm] = dates[j + 1].split('-');
      if (ny !== year || nm !== month) break;
      j += 1;
    }
    groups.push({ label: `${Number(month)}月`, startIndex: i, span: j - i + 1 });
    i = j + 1;
  }
  return groups;
}

function buildOperationMap(envConfig: EnvConfig): Map<string, Operation> {
  const map = new Map<string, Operation>();
  for (const wf of envConfig.workflowList) {
    for (const ph of wf.phaseList) {
      for (const op of ph.operationList) {
        map.set(op.id, op);
      }
    }
  }
  return map;
}

function getWorkerName(workerById: Map<string, Worker>, workerId: string): string {
  return workerById.get(workerId)?.name ?? workerId;
}

function getCompanyName(
  workerById: Map<string, Worker>,
  companyById: Map<string, string>,
  workerId: string,
): string {
  const companyId = workerById.get(workerId)?.workerCompany ?? '';
  return companyById.get(companyId) ?? companyId;
}

/**
 * Normalize Schedule.yaml + EnvConfig.yaml into a Module → Phase → Task tree.
 * Min/Max worker rules: Schedule.yaml first, EnvConfig.yaml as fallback.
 */
export function buildModuleViewModel(
  envConfig: EnvConfig,
  schedule: ScheduleData,
  dates: string[],
): ModuleViewModel {
  const monthGroups = buildMonthGroups(dates);
  const operationMap = buildOperationMap(envConfig);
  const workerById = new Map(envConfig.workerList.map(w => [w.id, w]));
  const companyById = new Map(envConfig.workerCompanyList.map(c => [c.id, c.name ?? c.id]));
  const workflowNameById = new Map(envConfig.workflowList.map(w => [w.id, w.name ?? w.id]));

  // assignments grouped by operationTask id
  const assignmentsByOpTask = new Map<string, Array<{ index: number; worker: string; startDate: string; endDate: string }>>();
  schedule.assignmentList.forEach((a, index) => {
    const list = assignmentsByOpTask.get(a.operationTask) ?? [];
    list.push({ index, worker: a.worker, startDate: a.startDate, endDate: a.endDate });
    assignmentsByOpTask.set(a.operationTask, list);
  });

  const modules: ModuleNode[] = schedule.workflowTaskList.filter(wt => wt.phaseTaskList && wt.phaseTaskList.length > 0).map(wt => {
    const moduleId = wt.id;
    const moduleName = wt.name ?? wt.id;

    const phases: ModulePhase[] = wt.phaseTaskList.map((pt, phaseIdx) => {
      const color = getColorForPhaseIndex(phaseIdx);
      const phaseWorkers = new Set<string>();
      let phaseBarStart: string | null = null;
      let phaseBarEnd: string | null = null;

      const tasks: ModuleTask[] = pt.operationTaskList.map(ot => {
        const op = operationMap.get(ot.operation);
        const minWorker = ot.recommendsWorkerMin ?? op?.minWorkerNum ?? 1;
        const workloadHours = ot.workloadHours ?? op?.workHours?.[0] ?? 8;

        const rawAssignments = assignmentsByOpTask.get(ot.id) ?? [];
        const slots: ModuleWorkerSlot[] = rawAssignments.map(ra => ({
          assignmentIndex: ra.index,
          workerId: ra.worker,
          workerName: getWorkerName(workerById, ra.worker),
          companyName: getCompanyName(workerById, companyById, ra.worker),
          startDate: ra.startDate,
          endDate: ra.endDate,
        }));

        const maxWorker =
          ot.recommendsWorkerMax ??
          op?.maxWorkerNum ??
          Math.max(minWorker, slots.length, 1);

        let taskStart: string | null = null;
        let taskEnd: string | null = null;
        for (const s of slots) {
          taskStart = taskStart === null ? s.startDate : minStr(taskStart, s.startDate);
          taskEnd = taskEnd === null ? s.endDate : maxStr(taskEnd, s.endDate);
          phaseWorkers.add(s.workerId);
        }
        if (taskStart) phaseBarStart = phaseBarStart === null ? taskStart : minStr(phaseBarStart, taskStart);
        if (taskEnd) phaseBarEnd = phaseBarEnd === null ? taskEnd : maxStr(phaseBarEnd, taskEnd);

        return {
          moduleId,
          phaseId: pt.id,
          taskId: ot.id,
          taskName: ot.name ?? op?.name ?? ot.operation ?? ot.id,
          operationId: ot.operation,
          minWorker,
          maxWorker,
          workloadHours,
          slots,
          startDate: taskStart,
          endDate: taskEnd,
          color,
          description: ot.description,
        };
      });

      return {
        moduleId,
        phaseId: pt.id,
        phaseName: pt.name ?? pt.phase ?? pt.id,
        planStartDate: pt.startDate,
        planEndDate: pt.endDate,
        barStartDate: phaseBarStart,
        barEndDate: phaseBarEnd,
        workerCount: phaseWorkers.size,
        tasks,
        color,
        description: pt.description,
      };
    });

    const workflowName = workflowNameById.get(wt.workflow) ?? wt.workflow ?? '';
    return { moduleId, moduleName, workflowName, fab: wt.fab ?? null, region: wt.region ?? null, phases };
  });

  return { modules, monthGroups };
}