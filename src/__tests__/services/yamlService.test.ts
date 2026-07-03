import { parseScheduleYaml, parseEnvConfigYaml, stringifyScheduleYaml } from '../../services/yamlService';

// ── Minimal valid YAML fixtures ────────────────────────────────────────────────

const SCHEDULE_YAML = `
schedule:
  plan_range:
    start_date: "2025-09-01"
    end_date: "2025-09-30"
  workflow_task_list:
    - id: wt001
      name: "Module A"
      workflow: wf_standard
      fab: fab_osaka
      phase_task_list:
        - id: wt001_p0
          name: "Setup"
          phase: p1
          start_date: "2025-09-01"
          end_date: "2025-09-15"
          operation_task_list:
            - id: wt001_p0_o0
              name: "Heavy"
              operation: p1o1
              workload_hours: 240
              recommends_worker_min: 2
              recommends_worker_max: 3
  assignment_list:
    - worker: w001
      operation_task: wt001_p0_o0
      start_date: "2025-09-01"
      end_date: "2025-09-05"
      plan_flexibility: Flexible
      work_date_list:
        - date: "2025-09-01"
          hour: 8
        - date: "2025-09-02"
          hour: 8
`;

const ENV_CONFIG_YAML = `
environment:
  workflow_list:
    - id: wf_standard
      name: Standard Workflow
      phase_list:
        - id: p1
          name: Module Setup
          operation_list:
            - id: p1o1
              name: Heavy
              work_hours: [8, 10, 12]
              workload_hours: 240
              min_worker_num: 2
              max_worker_num: 3
    - id: wf_misc
      name: Other Work
      phase_list: []
  fab_list:
    - id: fab_osaka
      name: Osaka Fab
      region: region_kansai
  region_list:
    - id: region_kansai
      name: Kansai
  customer_company_list: []
  worker_company_list:
    - id: co001
      name: TechCorp
      unavailable_dates: []
  worker_list:
    - id: w001
      name: Tanaka Taro
      worker_company: co001
      unavailable_dates: []
  transited_day_map: []
`;

// ── parseScheduleYaml ─────────────────────────────────────────────────────────

describe('parseScheduleYaml', () => {
  it('parses plan range correctly', () => {
    const schedule = parseScheduleYaml(SCHEDULE_YAML);
    expect(schedule.planRange.startDate).toBe('2025-09-01');
    expect(schedule.planRange.endDate).toBe('2025-09-30');
  });

  it('parses workflowTaskList', () => {
    const schedule = parseScheduleYaml(SCHEDULE_YAML);
    expect(schedule.workflowTaskList).toHaveLength(1);
    expect(schedule.workflowTaskList[0].id).toBe('wt001');
    expect(schedule.workflowTaskList[0].name).toBe('Module A');
    expect(schedule.workflowTaskList[0].fab).toBe('fab_osaka');
  });

  it('parses phaseTaskList within workflowTask', () => {
    const schedule = parseScheduleYaml(SCHEDULE_YAML);
    const phases = schedule.workflowTaskList[0].phaseTaskList;
    expect(phases).toHaveLength(1);
    expect(phases[0].id).toBe('wt001_p0');
    expect(phases[0].startDate).toBe('2025-09-01');
    expect(phases[0].endDate).toBe('2025-09-15');
  });

  it('parses operationTaskList workloadHours', () => {
    const schedule = parseScheduleYaml(SCHEDULE_YAML);
    const op = schedule.workflowTaskList[0].phaseTaskList[0].operationTaskList[0];
    expect(op.workloadHours).toBe(240);
    expect(op.recommendsWorkerMin).toBe(2);
    expect(op.recommendsWorkerMax).toBe(3);
  });

  it('parses assignmentList', () => {
    const schedule = parseScheduleYaml(SCHEDULE_YAML);
    expect(schedule.assignmentList).toHaveLength(1);
    const a = schedule.assignmentList[0];
    expect(a.worker).toBe('w001');
    expect(a.operationTask).toBe('wt001_p0_o0');
    expect(a.startDate).toBe('2025-09-01');
    expect(a.endDate).toBe('2025-09-05');
    expect(a.planFlexibility).toBe('Flexible');
  });

  it('parses workDateList inside assignment', () => {
    const schedule = parseScheduleYaml(SCHEDULE_YAML);
    const wdl = schedule.assignmentList[0].workDateList;
    expect(wdl).toHaveLength(2);
    expect(wdl[0].date).toBe('2025-09-01');
    expect(wdl[0].hour).toBe(8);
  });
});

// ── parseEnvConfigYaml ────────────────────────────────────────────────────────

describe('parseEnvConfigYaml', () => {
  it('parses workflowList', () => {
    const env = parseEnvConfigYaml(ENV_CONFIG_YAML);
    expect(env.workflowList).toHaveLength(2);
    expect(env.workflowList[0].id).toBe('wf_standard');
    expect(env.workflowList[1].id).toBe('wf_misc');
  });

  it('parses phaseList and operationList', () => {
    const env = parseEnvConfigYaml(ENV_CONFIG_YAML);
    const op = env.workflowList[0].phaseList[0].operationList[0];
    expect(op.id).toBe('p1o1');
    expect(op.workHours).toEqual([8, 10, 12]);
    expect(op.workloadHours).toBe(240);
    expect(op.minWorkerNum).toBe(2);
    expect(op.maxWorkerNum).toBe(3);
  });

  it('wf_misc has empty phaseList', () => {
    const env = parseEnvConfigYaml(ENV_CONFIG_YAML);
    const misc = env.workflowList.find(w => w.id === 'wf_misc')!;
    expect(misc.phaseList).toHaveLength(0);
  });

  it('parses fabList', () => {
    const env = parseEnvConfigYaml(ENV_CONFIG_YAML);
    expect(env.fabList).toHaveLength(1);
    expect(env.fabList[0].id).toBe('fab_osaka');
    expect(env.fabList[0].region).toBe('region_kansai');
  });

  it('parses regionList', () => {
    const env = parseEnvConfigYaml(ENV_CONFIG_YAML);
    expect(env.regionList).toHaveLength(1);
    expect(env.regionList[0].id).toBe('region_kansai');
    expect(env.regionList[0].name).toBe('Kansai');
  });

  it('parses workerList', () => {
    const env = parseEnvConfigYaml(ENV_CONFIG_YAML);
    expect(env.workerList).toHaveLength(1);
    expect(env.workerList[0].id).toBe('w001');
    expect(env.workerList[0].workerCompany).toBe('co001');
  });
});

// ── stringifyScheduleYaml round-trip ─────────────────────────────────────────

describe('stringifyScheduleYaml round-trip', () => {
  it('re-parses to the same plan range', () => {
    const original = parseScheduleYaml(SCHEDULE_YAML);
    const yaml = stringifyScheduleYaml(original);
    const roundTripped = parseScheduleYaml(yaml);
    expect(roundTripped.planRange).toEqual(original.planRange);
  });

  it('re-parses with same number of assignments', () => {
    const original = parseScheduleYaml(SCHEDULE_YAML);
    const yaml = stringifyScheduleYaml(original);
    const roundTripped = parseScheduleYaml(yaml);
    expect(roundTripped.assignmentList).toHaveLength(original.assignmentList.length);
  });
});
