// ─── Run status ────────────────────────────────────────────────────────────
export type RunStatus = 'Completed' | 'Failed' | 'Executing';

// ─── Run (one solver execution = one local folder) ──────────────────────────
// Mock model: folders are simulated. Each Run maps to ./runs/<id>/ with
// input/ and output/ subfolders. No data ever touches the real disk.
export type OutputState = 'none' | 'fetching' | 'ready';

export interface Run {
  id: string;            // e.g. "20251014_003"
  solveDate: string;     // ISO string — when the solve was created/submitted
  label: string;         // optional human label
  folderPath: string;    // "./runs/20251014_003/"
  inputEnvName: string;  // input file name, e.g. "EnvConfig.yaml"
  inputSchedName: string;// input file name, e.g. "Schedule.yaml"
  // Public URL of the folder that actually holds the two YAML files
  // (e.g. "/data/runs/20260521/input"). When set, the Gantt fetches and
  // renders REAL data via the yaml_to_suother_like_excel.py port.
  // When null, the Gantt uses the mock fallback grid.
  inputDir: string | null;
  output: OutputState;   // result folder state
  // Whether the output folder actually contains 2 YAML files.
  // Only when true does the *result* Gantt button become enabled.
  outputHasYaml: boolean;
}

// ─── Core domain ───────────────────────────────────────────────────────────
export interface Dataset {
  id: string;
  name: string;
  description: string;
  createdAt: string;
  updatedAt: string;
  runCount: number;
  latestStatus: RunStatus | null;
}

export interface RunLog {
  id: string;
  datasetId: string;
  runNumber: number;
  status: RunStatus;
  label: string;
  startedAt: string;      // ISO string
  finishedAt: string | null;
  hardScore: number | null;
  softScore: number | null;
  outputPath: string | null;
}

export interface Comment {
  id: string;
  datasetId: string;
  author: string;
  body: string;
  createdAt: string;
}

// ─── Gantt domain ──────────────────────────────────────────────────────────
export interface GanttEmployee {
  id: string;
  name: string;
  company: string;
  companyColor: string;
  role: string;
  isManager: boolean;
}

export interface GanttModule {
  code: string;       // e.g. "530N02621A"
  baseCode: string;   // text before first underscore, used for color grouping
  color: string;      // hex color
}

export interface GanttCell {
  type: 'work' | 'unavailable' | 'empty';
  moduleCode?: string;
  moduleColor?: string;
  operationId?: string;
  isCutoff?: boolean;   // draw right border
  isToday?: boolean;    // draw left border (orange)
}

export interface GanttData {
  employees: GanttEmployee[];
  dates: Date[];
  cutoffDate: Date | null;
  todayDate: Date;
  // cells[employeeIndex][dateIndex]
  cells: GanttCell[][];
  modules: GanttModule[];
  planStart: Date;
  planEnd: Date;
}

// ─── Raw YAML shapes (match EnvConfig.yaml + Schedule.yaml exactly) ──────────
// These mirror the structures read by yaml_to_suother_like_excel.py.

// EnvConfig.yaml ─────────────────────────────────────────────────────────────
export interface RawWorkerCompany {
  id: string;
  name?: string;
}

export interface RawWorker {
  id: string;
  name?: string;
  worker_company?: string;   // → worker_company_list[].id
  is_manager?: boolean;
  role?: string;
  unavailable_dates?: unknown[];
}

export interface RawEnvironment {
  worker_company_list?: RawWorkerCompany[];
  worker_list?: RawWorker[];
}

// Root may be wrapped in `environment:` or be flat.
export interface RawEnvConfig {
  environment?: RawEnvironment;
  worker_company_list?: RawWorkerCompany[];
  worker_list?: RawWorker[];
}

// Schedule.yaml ──────────────────────────────────────────────────────────────
export interface RawOperationTask {
  id: string;
  name?: string;
  operation?: string;
}

export interface RawPhaseTask {
  id: string;
  operation_task_list?: RawOperationTask[];
}

export interface RawWorkflowTask {
  id: string;
  name?: string;            // module display name, e.g. "SU 1001A"
  workflow?: string;
  phase_task_list?: RawPhaseTask[];
}

export interface RawWorkDate {
  date: string;
  hour?: number;
}

export interface RawAssignment {
  worker: string;
  operation_task: string;
  plan_flexibility?: string;
  work_date_list?: RawWorkDate[];
  work_date_lsit?: RawWorkDate[];   // tolerate the typo seen in some exports
}

export interface RawScheduleBody {
  plan_range: { start_date: string; end_date: string };
  workflow_task_list?: RawWorkflowTask[];
  assignment_list?: RawAssignment[];
}

// Root may be wrapped in `schedule:` or be flat.
export interface RawSchedule {
  schedule?: RawScheduleBody;
  plan_range?: { start_date: string; end_date: string };
  workflow_task_list?: RawWorkflowTask[];
  assignment_list?: RawAssignment[];
}

// ─── New Run form ──────────────────────────────────────────────────────────
export interface JobOrderRow {
  jobId: string;
  customer: string;
  fab: string;
  region: string;
}

export interface AvailabilityRow {
  employeeId: string;
  dates: string;
  type: 'vacation' | 'restriction';
}

export interface NewRunForm {
  useExistingDataset: boolean;
  existingDatasetId: string;
  useNewFiles: boolean;
  newEnvConfigFile: File | null;
  newScheduleFile: File | null;
  planStart: string;
  planEnd: string;
  cutoffDate: string;
  solveLabel: string;
  solveDurationMinutes: number;
  allowOvertime: boolean;
  jobOrders: JobOrderRow[];
}
