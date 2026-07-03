// Mirror of frontend types — keep in sync with src/types/

export interface WorkDate { date: string; hour: number }
export interface Assignment {
  worker: string;
  operationTask: string;
  startDate: string;
  endDate: string;
  workDateList: WorkDate[];
  planFlexibility: string;
}
export interface OperationTask {
  id: string;
  name?: string;
  operation: string;
  workloadHours: number;
  recommendsWorkerMin?: number;
  recommendsWorkerMax?: number;
}
export interface PhaseTask {
  id: string;
  name?: string;
  phase: string;
  startDate: string;
  endDate: string;
  operationTaskList: OperationTask[];
}
export interface WorkflowTask {
  id: string;
  name?: string;
  workflow: string;
  fab?: string;
  region?: string;
  phaseTaskList: PhaseTask[];
}
export interface ScheduleData {
  planRange: { startDate: string; endDate: string };
  workflowTaskList: WorkflowTask[];
  assignmentList: Assignment[];
}

export interface Operation {
  id: string;
  name?: string;
  workloadHours?: number;
  minWorkerNum?: number;
  maxWorkerNum?: number;
}
export interface Phase { id: string; name?: string; operationList: Operation[] }
export interface Workflow { id: string; name?: string; phaseList: Phase[] }
export interface Worker {
  id: string;
  name?: string;
  isManager?: boolean;
  workerCompany?: string;
  skillMap?: Record<string, number>;
  unavailableDates: unknown[];
}
export interface Fab { id: string; name?: string; region?: string }
export interface Region { id: string; name?: string }
export interface TransiteDayMap { from: string; to: string; days: number }
export interface EnvConfig {
  workflowList: Workflow[];
  fabList: Fab[];
  regionList: Region[];
  workerList: Worker[];
  transiteDayMap: TransiteDayMap[];
  customerCompanyList: unknown[];
  workerCompanyList: unknown[];
}

export interface Violation {
  type: string;
  assignmentIndices: number[];
  message: string;
  date?: string;
  severity: 'error' | 'warning';
}
