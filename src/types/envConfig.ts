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

export interface Worker {
  id: string;
  name?: string;
  description?: string;
  workerCompany?: string;
  isManager?: boolean;
  skillMap?: Record<string, number>;
  unavailableDates: UnavailableDateEntry[];
  definition?: string;
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
  minWorkerNum?: number;
  maxWorkerNum?: number;
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

export interface EnvConfig {
  workflowList: Workflow[];
  fabList: Fab[];
  regionList: Region[];
  customerCompanyList: CustomerCompany[];
  workerCompanyList: WorkerCompany[];
  workerList: Worker[];
  transiteDayMap: TransiteDayMap[];
}
