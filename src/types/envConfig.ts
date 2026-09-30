export interface WeeklyUnavailableDate {
  weekdays: string[];
}

export interface SingleUnavailableDate {
  days: string[];
}

export interface UnavailableDateEntry {
  weekly?: WeeklyUnavailableDate;
  single?: SingleUnavailableDate;
}

export type WorkerDescription = {
  '業務形態'?: string;
  'VISA'?: string;
  '海外運転'?: string;
  '備考'?: string;
};

export interface FabSuitabilityEntry {
  kind: string;
  suitability: Record<string, number>;
}

export interface Worker {
  id: string;
  name?: string;
  description?: WorkerDescription;
  workerCompany?: string;
  isManager?: boolean;
  skillMap?: Record<string, number>;
  workerTypeByOperation?: Record<string, string>;
  fabSuitabilityMap?: FabSuitabilityEntry[];
  affinity?: string[];
  unavailableDates: UnavailableDateEntry[];
}

export interface WorkerCompany {
  id: string;
  name?: string;
  annualOvertimeLimit?: number;
  monthlyOvertimeLimit?: number;
  unavailableDates: UnavailableDateEntry[];
}

export interface Fab {
  id: string;
  name?: string;
  region?: string;
  customerCompany?: string;
  unavailableDates: UnavailableDateEntry[];
}

export interface Region {
  id: string;
  name?: string;
  maxStayOn?: number;
  maxAnnualStay?: number;
  stayOffInterval?: number;
  unavailableDates: UnavailableDateEntry[];
}

export interface CustomerCompany {
  id: string;
  name?: string;
  unavailableDates: UnavailableDateEntry[];
}

export interface Operation {
  id: string;
  name?: string;
  workHours?: number[];
  workloadHours?: number;
  minWorkerNum?: number;
  maxWorkerNum?: number;
  /** Minimum worker.skillMap[operation.id] needed to be assigned this operation — see constraintService's SKILL_MISMATCH check. */
  requiredSkillLevel?: number;
}

export interface Phase {
  id: string;
  name?: string;
  operationList: Operation[];
}

export interface Workflow {
  id: string;
  name?: string;
  phaseList: Phase[];
}

export interface TransiteDayMap {
  from: string;
  to: string;
  days: number;
}

// Tag DEFINITIONS (id + weight, weight can be negative) — a top-level
// EnvConfig.yaml section distinct from Worker.affinity, which is just a
// worker's own list of tag id references into this list. Optional (not a
// required array like the other *List fields) so existing EnvConfig
// literals across the codebase don't all need updating for a section that
// may simply be absent from older/simpler YAML.
export interface AffinityTag {
  id: string;
  weight: number;
}

export interface EnvConfig {
  workflowList: Workflow[];
  fabList: Fab[];
  regionList: Region[];
  customerCompanyList: CustomerCompany[];
  workerCompanyList: WorkerCompany[];
  workerList: Worker[];
  transiteDayMap: TransiteDayMap[];
  affinityTagList?: AffinityTag[];
}
