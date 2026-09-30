import { parseScheduleYaml, parseEnvConfigYaml, stringifyScheduleYaml, stringifyEnvConfigYaml } from '../../services/yamlService';

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

// ── stringifyEnvConfigYaml round-trip — regression test for fields silently
// dropped on save: fab/region/customerCompany/workerCompany unavailableDates
// and region's maxStayOn/maxAnnualStay/stayOffInterval were never written out
// at all (so they defaulted back to [] / 0 on the next load), and
// unavailable_dates single.days survived a save→reload as YYYY/MM/DD instead
// of the app's internal YYYY-MM-DD, silently breaking every date comparison
// downstream (isWeeklyDate, the unavailable-date bar, etc).

describe('stringifyEnvConfigYaml round-trip', () => {
  const richEnv = () => {
    const env = parseEnvConfigYaml(ENV_CONFIG_YAML);
    env.fabList[0].unavailableDates = [{ single: { days: ['2026-01-01'] } }];
    env.regionList[0] = {
      ...env.regionList[0],
      maxStayOn: 30, maxAnnualStay: 180, stayOffInterval: 7,
      unavailableDates: [{ weekly: { weekdays: ['sunday'] } }],
    };
    env.customerCompanyList = [{ id: 'cust1', name: 'Cust1', unavailableDates: [{ single: { days: ['2026-02-01'] } }] }];
    env.workerCompanyList[0].unavailableDates = [{ single: { days: ['2026-03-01'] } }];
    env.workerList[0].affinity = ['w2', 'w3'];
    env.workerList[0].unavailableDates = [
      { weekly: { weekdays: ['sunday'] } },
      { single: { days: ['2026-04-01', '2026-04-02'] } },
    ];
    env.affinityTagList = [{ id: 'wct1', weight: 2 }, { id: 'a3', weight: -1 }];
    return env;
  };

  it('round-trips the affinity_tag list (tag definitions, distinct from a worker\'s own affinity references)', () => {
    const roundTripped = parseEnvConfigYaml(stringifyEnvConfigYaml(richEnv()));
    expect(roundTripped.affinityTagList).toEqual([{ id: 'wct1', weight: 2 }, { id: 'a3', weight: -1 }]);
  });

  it('omits affinity_tag entirely when the source EnvConfig never had it', () => {
    const env = parseEnvConfigYaml(ENV_CONFIG_YAML); // no affinity_tag in the fixture
    expect(env.affinityTagList).toBeUndefined();
    const yamlOut = stringifyEnvConfigYaml(env);
    expect(yamlOut).not.toMatch(/affinity_tag/);
  });

  it('round-trips fab unavailableDates', () => {
    const roundTripped = parseEnvConfigYaml(stringifyEnvConfigYaml(richEnv()));
    expect(roundTripped.fabList[0].unavailableDates).toEqual([{ single: { days: ['2026-01-01'] } }]);
  });

  it('round-trips region maxStayOn/maxAnnualStay/stayOffInterval and unavailableDates', () => {
    const roundTripped = parseEnvConfigYaml(stringifyEnvConfigYaml(richEnv()));
    expect(roundTripped.regionList[0]).toMatchObject({
      maxStayOn: 30, maxAnnualStay: 180, stayOffInterval: 7,
      unavailableDates: [{ weekly: { weekdays: ['sunday'] } }],
    });
  });

  it('round-trips customerCompany unavailableDates', () => {
    const roundTripped = parseEnvConfigYaml(stringifyEnvConfigYaml(richEnv()));
    expect(roundTripped.customerCompanyList[0].unavailableDates).toEqual([{ single: { days: ['2026-02-01'] } }]);
  });

  it('round-trips workerCompany unavailableDates', () => {
    const roundTripped = parseEnvConfigYaml(stringifyEnvConfigYaml(richEnv()));
    expect(roundTripped.workerCompanyList[0].unavailableDates).toEqual([{ single: { days: ['2026-03-01'] } }]);
  });

  it('round-trips worker affinity', () => {
    const roundTripped = parseEnvConfigYaml(stringifyEnvConfigYaml(richEnv()));
    expect(roundTripped.workerList[0].affinity).toEqual(['w2', 'w3']);
  });

  it('round-trips an operation required_skill_level, which used to be silently dropped on save', () => {
    const env = richEnv();
    env.workflowList[0].phaseList[0].operationList[0].requiredSkillLevel = 3;
    const roundTripped = parseEnvConfigYaml(stringifyEnvConfigYaml(env));
    expect(roundTripped.workflowList[0].phaseList[0].operationList[0].requiredSkillLevel).toBe(3);
  });

  it('keeps unavailable_dates single.days in internal YYYY-MM-DD format after a round trip, not YYYY/MM/DD', () => {
    const roundTripped = parseEnvConfigYaml(stringifyEnvConfigYaml(richEnv()));
    const workerDays = roundTripped.workerList[0].unavailableDates.find(d => d.single)?.single?.days;
    expect(workerDays).toEqual(['2026-04-01', '2026-04-02']);
  });

  it('does a full deep-equal round trip with every list populated', () => {
    const env = richEnv();
    const roundTripped = parseEnvConfigYaml(stringifyEnvConfigYaml(env));
    expect(roundTripped).toEqual(env);
  });
});

// ── ys() quoting edge cases — regression test for the "-" / "," bug ─────────
// Bug: description/name fields whose value was exactly "-" or "," (used as
// N/A-style placeholders) got written back out unquoted on save, producing
// invalid YAML that failed to parse on the next load.

describe('ys() quoting edge cases (schedule round-trip)', () => {
  const roundTripDescription = (value: string) => {
    const original = parseScheduleYaml(SCHEDULE_YAML);
    original.workflowTaskList[0].description = value;
    const yaml = stringifyScheduleYaml(original);
    const roundTripped = parseScheduleYaml(yaml);
    return roundTripped.workflowTaskList[0].description;
  };

  it('round-trips a task description of "-"', () => {
    expect(roundTripDescription('-')).toBe('-');
  });

  it('round-trips a task description of ","', () => {
    expect(roundTripDescription(',')).toBe(',');
  });

  it('round-trips a task description of "?"', () => {
    expect(roundTripDescription('?')).toBe('?');
  });

  it('round-trips a numeric-looking description "-1" as a string, not a number', () => {
    const result = roundTripDescription('-1');
    expect(typeof result).toBe('string');
    expect(result).toBe('-1');
  });

  it('round-trips a numeric-looking description "007" as a string', () => {
    const result = roundTripDescription('007');
    expect(typeof result).toBe('string');
    expect(result).toBe('007');
  });

  it('round-trips a date-looking description "2025-09-01" as a string, not a Date', () => {
    const result = roundTripDescription('2025-09-01');
    expect(typeof result).toBe('string');
    expect(result).toBe('2025-09-01');
  });

  it('leaves a normal placeholder value "N/A" unquoted-safe and unchanged', () => {
    expect(roundTripDescription('N/A')).toBe('N/A');
  });
});

describe('ys() quoting edge cases (envConfig round-trip)', () => {
  it('round-trips a worker 備考 field of "-"', () => {
    const env = parseEnvConfigYaml(ENV_CONFIG_YAML);
    env.workerList[0].description = { '備考': '-' };
    const yaml = stringifyEnvConfigYaml(env);
    const roundTripped = parseEnvConfigYaml(yaml);
    expect(roundTripped.workerList[0].description?.['備考']).toBe('-');
  });

  it('round-trips a worker 業務形態 field of ","', () => {
    const env = parseEnvConfigYaml(ENV_CONFIG_YAML);
    env.workerList[0].description = { '業務形態': ',' };
    const yaml = stringifyEnvConfigYaml(env);
    const roundTripped = parseEnvConfigYaml(yaml);
    expect(roundTripped.workerList[0].description?.['業務形態']).toBe(',');
  });
});
