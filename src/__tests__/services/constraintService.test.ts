import { checkConstraints } from '../../services/constraintService';
import { EnvConfig } from '../../types/envConfig';
import { ScheduleData } from '../../types/schedule';

// Regression test for a bug where an operation's required_skill_level was
// never actually loaded from (or saved to) EnvConfig.yaml — see
// yamlService.ts's parseOperation/stringifyEnvConfigYaml — so this check
// could never fire no matter what the source YAML said.

const baseEnvConfig = (): EnvConfig => ({
  workflowList: [
    {
      id: 'wf1',
      phaseList: [
        {
          id: 'p1',
          operationList: [
            { id: 'op1', minWorkerNum: 1, maxWorkerNum: 3, requiredSkillLevel: 3 },
          ],
        },
      ],
    },
  ],
  fabList: [], regionList: [], customerCompanyList: [], transiteDayMap: [],
  workerCompanyList: [],
  workerList: [
    { id: 'w1', skillMap: { op1: 1 }, unavailableDates: [] },
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
  it('flags a worker whose skillMap level is below the operation required_skill_level', () => {
    const violations = checkConstraints(baseEnvConfig(), baseSchedule());
    const skillViolations = violations.filter(v => v.type === 'SKILL_MISMATCH');
    expect(skillViolations).toHaveLength(1);
    expect(skillViolations[0].assignmentIndices).toEqual([0]);
  });

  it('does not flag a worker whose skillMap level meets the requirement', () => {
    const env = baseEnvConfig();
    env.workerList[0].skillMap = { op1: 3 };
    const violations = checkConstraints(env, baseSchedule());
    expect(violations.filter(v => v.type === 'SKILL_MISMATCH')).toHaveLength(0);
  });

  it('does not flag anything when the operation has no required_skill_level (0/unset)', () => {
    const env = baseEnvConfig();
    env.workflowList[0].phaseList[0].operationList[0].requiredSkillLevel = 0;
    const violations = checkConstraints(env, baseSchedule());
    expect(violations.filter(v => v.type === 'SKILL_MISMATCH')).toHaveLength(0);
  });

  it('does not flag a worker with no skillMap entry for the operation at all, when nothing is required', () => {
    const env = baseEnvConfig();
    env.workflowList[0].phaseList[0].operationList[0].requiredSkillLevel = 0;
    env.workerList[0].skillMap = {};
    const violations = checkConstraints(env, baseSchedule());
    expect(violations.filter(v => v.type === 'SKILL_MISMATCH')).toHaveLength(0);
  });
});
