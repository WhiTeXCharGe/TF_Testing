/**
 * Fallback mock data used when database.xlsx is not available.
 * Mirrors the structure of what databaseService reads from Excel.
 */
import type { Dataset, RunLog, Comment, RawEnvConfig, RawSchedule } from '@/types';

export const MOCK_DATASETS: Dataset[] = [
  {
    id: '2025SU_OTHER',
    name: '2025 Summer (Other)',
    description: 'Summer season schedule for OTHER region workflows.',
    createdAt: '2025-09-01T09:00:00Z',
    updatedAt: '2025-10-25T14:30:00Z',
    runCount: 4,
    latestStatus: 'Executing',
  },
  {
    id: '2025SU_MAIN',
    name: '2025 Summer (Main)',
    description: 'Summer season schedule for MAIN region workflows.',
    createdAt: '2025-09-01T09:00:00Z',
    updatedAt: '2025-10-22T10:00:00Z',
    runCount: 2,
    latestStatus: 'Completed',
  },
  {
    id: '2025FA_OTHER',
    name: '2025 Fall (Other)',
    description: 'Fall season schedule — initial planning run.',
    createdAt: '2025-11-01T08:00:00Z',
    updatedAt: '2025-11-01T08:00:00Z',
    runCount: 1,
    latestStatus: 'Failed',
  },
];

export const MOCK_RUN_LOGS: RunLog[] = [
  // 2025SU_OTHER
  {
    id: 'r001',
    datasetId: '2025SU_OTHER',
    runNumber: 1,
    status: 'Completed',
    label: 'Initial solve',
    startedAt: '2025-10-15T09:00:00Z',
    finishedAt: '2025-10-15T12:11:00Z',
    hardScore: 0,
    softScore: -142,
    outputPath: 'output/2025SU_OTHER/run-001/',
  },
  {
    id: 'r002',
    datasetId: '2025SU_OTHER',
    runNumber: 2,
    status: 'Completed',
    label: 'Conservative (no overtime)',
    startedAt: '2025-10-18T10:00:00Z',
    finishedAt: '2025-10-18T13:05:00Z',
    hardScore: 0,
    softScore: -198,
    outputPath: 'output/2025SU_OTHER/run-002/',
  },
  {
    id: 'r003',
    datasetId: '2025SU_OTHER',
    runNumber: 3,
    status: 'Failed',
    label: 'Extended plan range',
    startedAt: '2025-10-20T11:00:00Z',
    finishedAt: '2025-10-20T11:08:00Z',
    hardScore: null,
    softScore: null,
    outputPath: null,
  },
  {
    id: 'r004',
    datasetId: '2025SU_OTHER',
    runNumber: 4,
    status: 'Executing',
    label: 'Overtime allowed + new jobs',
    startedAt: new Date(Date.now() - 5025 * 1000).toISOString(),
    finishedAt: null,
    hardScore: null,
    softScore: null,
    outputPath: null,
  },
  // 2025SU_MAIN
  {
    id: 'r005',
    datasetId: '2025SU_MAIN',
    runNumber: 1,
    status: 'Completed',
    label: 'Initial solve',
    startedAt: '2025-10-10T09:00:00Z',
    finishedAt: '2025-10-10T12:22:00Z',
    hardScore: 0,
    softScore: -88,
    outputPath: 'output/2025SU_MAIN/run-001/',
  },
  {
    id: 'r006',
    datasetId: '2025SU_MAIN',
    runNumber: 2,
    status: 'Completed',
    label: 'Refined',
    startedAt: '2025-10-22T09:00:00Z',
    finishedAt: '2025-10-22T12:05:00Z',
    hardScore: 0,
    softScore: -72,
    outputPath: 'output/2025SU_MAIN/run-002/',
  },
  // 2025FA_OTHER
  {
    id: 'r007',
    datasetId: '2025FA_OTHER',
    runNumber: 1,
    status: 'Failed',
    label: 'First attempt',
    startedAt: '2025-11-01T08:00:00Z',
    finishedAt: '2025-11-01T08:02:00Z',
    hardScore: null,
    softScore: null,
    outputPath: null,
  },
];

export const MOCK_COMMENTS: Comment[] = [
  {
    id: 'c001',
    datasetId: '2025SU_OTHER',
    author: 'Yamada',
    body: 'Run 1 looks good. Checking with PM before locking assignments.',
    createdAt: '2025-10-15T14:00:00Z',
  },
  {
    id: 'c002',
    datasetId: '2025SU_OTHER',
    author: 'Tanaka',
    body: 'Run 3 failed due to invalid plan_end date in YAML. Fixed and resubmitted as Run 4.',
    createdAt: '2025-10-20T11:30:00Z',
  },
];

// ─────────────────────────────────────────────────────────────────────────────
// Mock raw YAML — Gantt "mock fallback" for runs with inputDir=null (e.g. brand-
// new runs). Uses the REAL YAML shapes, fed through the SAME buildGanttData()
// port, so the fallback exercises the real logic.
// ─────────────────────────────────────────────────────────────────────────────
export const MOCK_ENV_CONFIG: RawEnvConfig = {
  environment: {
    worker_company_list: [
      { id: 'c1', name: 'Alpha Corp' },
      { id: 'c2', name: 'Beta Tech' },
    ],
    worker_list: [
      { id: 'w1', name: 'Tanaka K.',   worker_company: 'c1', is_manager: true,  role: 'Mech',
        unavailable_dates: [{ single: { days: ['2025/10/16', '2025/10/17'] } }] },
      { id: 'w2', name: 'Suzuki M.',   worker_company: 'c1', is_manager: false, role: 'Elec' },
      { id: 'w3', name: 'Sato R.',     worker_company: 'c1', is_manager: false, role: 'QC' },
      { id: 'w4', name: 'Yamada T.',   worker_company: 'c2', is_manager: false, role: 'Heavy' },
      { id: 'w5', name: 'Ito N.',      worker_company: 'c2', is_manager: false, role: 'Mech' },
      { id: 'w6', name: 'Watanabe S.', worker_company: 'c2', is_manager: true,  role: 'Elec',
        unavailable_dates: [{ single: { days: ['2025/11/03'] } }] },
    ],
  },
};

export const MOCK_SCHEDULE: RawSchedule = {
  schedule: {
    plan_range: { start_date: '2025/10/01', end_date: '2025/11/15' },
    workflow_task_list: [
      { id: 'e1', name: 'SU 1001A', workflow: 'workflow',
        phase_task_list: [{ id: 'e1p1', operation_task_list: [{ id: 'e1p1o1', operation: 'p1o1' }] }] },
      { id: 'e2', name: 'SU 1002B', workflow: 'workflow',
        phase_task_list: [{ id: 'e2p1', operation_task_list: [{ id: 'e2p1o1', operation: 'p1o1' }] }] },
      { id: 'e3', name: 'SU 1003C', workflow: 'workflow',
        phase_task_list: [{ id: 'e3p1', operation_task_list: [{ id: 'e3p1o1', operation: 'p1o1' }] }] },
    ],
    assignment_list: [
      { worker: 'w1', operation_task: 'e1p1o1', work_date_list: workDates('2025/10/01', '2025/10/15') },
      { worker: 'w2', operation_task: 'e1p1o1', work_date_list: workDates('2025/10/01', '2025/10/17') },
      { worker: 'w3', operation_task: 'e2p1o1', work_date_list: workDates('2025/10/05', '2025/10/22') },
      { worker: 'w4', operation_task: 'e2p1o1', work_date_list: workDates('2025/10/01', '2025/10/12') },
      { worker: 'w5', operation_task: 'e3p1o1', work_date_list: workDates('2025/10/08', '2025/10/20') },
      { worker: 'w6', operation_task: 'e3p1o1', work_date_list: workDates('2025/10/20', '2025/11/04') },
    ],
  },
};

/** Inclusive list of { date: "YYYY/MM/DD" } between two dates. */
function workDates(start: string, end: string): { date: string }[] {
  const out: { date: string }[] = [];
  const cur = new Date(start.replace(/\//g, '-'));
  const last = new Date(end.replace(/\//g, '-'));
  while (cur <= last) {
    const y = cur.getFullYear();
    const m = String(cur.getMonth() + 1).padStart(2, '0');
    const d = String(cur.getDate()).padStart(2, '0');
    out.push({ date: `${y}/${m}/${d}` });
    cur.setDate(cur.getDate() + 1);
  }
  return out;
}
