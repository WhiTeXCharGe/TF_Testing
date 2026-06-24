export type PlanFlexibility = 'Flexible' | 'Reluctant' | 'Fixed';

export interface PlanRange {
  startDate: string; // YYYY-MM-DD
  endDate: string;
}

export interface WorkDate {
  date: string; // YYYY-MM-DD
  hour: number;
}

export interface Assignment {
  worker: string;
  operationTask: string;
  startDate: string;
  endDate: string;
  workDateList: WorkDate[];
  planFlexibility: PlanFlexibility;
}

// workload_hours is the canonical field (Timefold output)
// workload_days * HOURS_PER_DAY is the conversion for older YAMLs
export interface OperationTask {
  id: string;
  name?: string;
  description?: string;
  operation: string;
  workloadHours: number;
  recommendsWorkerMin?: number;
  recommendsWorkerMax?: number;
  colorCode?: string;
}

export interface PhaseTask {
  id: string;
  name?: string;
  description?: string;
  phase: string;
  startDate: string;
  endDate: string;
  operationTaskList: OperationTask[];
}

export interface WorkflowTask {
  id: string;
  name?: string;
  description?: string;
  workflow: string;
  fab?: string;
  region?: string;
  colorCode?: string;
  phaseTaskList: PhaseTask[];
}

export interface ScheduleData {
  planRange: PlanRange;
  workflowTaskList: WorkflowTask[];
  assignmentList: Assignment[];
}
