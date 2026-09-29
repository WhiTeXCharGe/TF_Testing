import { checkConstraints } from '../../services/constraintService';
import { EnvConfig } from '../../types/envConfig';
import { ScheduleData } from '../../types/schedule';

// Regression tests for the skill-mismatch check. It used to gate on an
// operation-level required_skill_level field, which (a) never actually
// round-tripped through EnvConfig.yaml (separately fixed in yamlService.ts)
// and (b) doesn't exist anywhere in real data at all — the actual scheduling
// engine (TFDocker EmployeeSchedule.java) gates eligibility on a binary
// skill_map[operationId] >= 1, with no per-operation threshold concept. A
// worker with no skillMap entry (or an explicit 0) for an operation is
// unqualified for it, full stop — that's the real-world "put an Elec task on
// a worker with no Elec skill" scenario this check exists to catch.

const baseEnvConfig = (): EnvConfig => ({
  workflowList: [
    {
      id: 'wf1',
      phaseList: [
        {
          id: 'p1',
          operationList: [
            { id: 'op1', minWorkerNum: 1, maxWorkerNum: 3 },
          ],
        },
      ],
    },
  ],
  fabList: [], regionList: [], customerCompanyList: [], transiteDayMap: [],
  workerCompanyList: [],
  workerList: [
    { id: 'w1', skillMap: {}, unavailableDates: [] },
  ],
});

const baseSchedule = (): ScheduleData => ({
  planRange: { startDate: '2026-01-01', endDate: '2026-01-31' },
  workflowTaskList: [
    {
      id: 'wt1', workflow: 'wf1',
      phaseTaskList: [
        {
          id: 'pt1', phase: 'p1', startDate: '2026-01-01', endDate: '2026-01-31',
          operationTaskList: [{ id: 'ot1', operation: 'op1', workloadHours: 8 }],
        },
      ],
    },
  ],
  assignmentList: [
    {
      worker: 'w1', operationTask: 'ot1', startDate: '2026-01-05', endDate: '2026-01-05',
      planFlexibility: 'Flexible', workDateList: [{ date: '2026-01-05', hour: 8 }],
    },
  ],
});

describe('checkConstraints — skill mismatch', () => {
  it('flags a worker with no skillMap entry at all for the assigned operation (the real "Elec on a non-Elec worker" case)', () => {
    const violations = checkConstraints(baseEnvConfig(), baseSchedule());
    const skillViolations = violations.filter(v => v.type === 'SKILL_MISMATCH');
    expect(skillViolations).toHaveLength(1);
    expect(skillViolations[0].assignmentIndices).toEqual([0]);
  });

  it('flags a worker with an explicit skillMap level of 0 for the operation, same as no entry', () => {
    const env = baseEnvConfig();
    env.workerList[0].skillMap = { op1: 0 };
    const violations = checkConstraints(env, baseSchedule());
    expect(violations.filter(v => v.type === 'SKILL_MISMATCH')).toHaveLength(1);
  });

  it('does not flag a worker with any positive skillMap level for the operation', () => {
    const env = baseEnvConfig();
    env.workerList[0].skillMap = { op1: 1 };
    const violations = checkConstraints(env, baseSchedule());
    expect(violations.filter(v => v.type === 'SKILL_MISMATCH')).toHaveLength(0);
  });

  it('honors a stricter required_skill_level if one is explicitly set above the default threshold of 1', () => {
    const env = baseEnvConfig();
    env.workflowList[0].phaseList[0].operationList[0].requiredSkillLevel = 3;
    env.workerList[0].skillMap = { op1: 1 };
    const violations = checkConstraints(env, baseSchedule());
    expect(violations.filter(v => v.type === 'SKILL_MISMATCH')).toHaveLength(1);
  });

  it('does not flag misc (personal-business style) tasks, which have no real operation to check against', () => {
    const env = baseEnvConfig();
    const schedule: ScheduleData = {
      ...baseSchedule(),
      workflowTaskList: [
        { id: 'misc1', workflow: 'wf_personal_business', phaseTaskList: [] },
      ],
      assignmentList: [
        { worker: 'w1', operationTask: 'misc1', startDate: '2026-01-05', endDate: '2026-01-05', planFlexibility: 'Flexible', workDateList: [{ date: '2026-01-05', hour: 8 }] },
      ],
    };
    const violations = checkConstraints(env, schedule);
    expect(violations.filter(v => v.type === 'SKILL_MISMATCH')).toHaveLength(0);
  });
});
