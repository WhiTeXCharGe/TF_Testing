import { describe, it, expect } from 'vitest';
import express from 'express';
import request from 'supertest';
import { constraintsRouter } from './constraints.js';

const app = express();
app.use(express.json());
app.use('/api', constraintsRouter);

const envConfig = {
  workflowList: [], fabList: [], regionList: [], customerCompanyList: [],
  workerCompanyList: [], workerList: [{ id: 'w1', unavailableDates: [] }], transiteDayMap: [],
};

const schedule = (workflowTaskList: unknown[]) => ({
  planRange: { startDate: '2026-01-01', endDate: '2026-01-31' },
  workflowTaskList,
  assignmentList: [],
});

describe('POST /api/check-constraints', () => {
  // misc_task_list entries no longer carry a workflow — the request must
  // not be rejected for lacking one ("制約チェックエラー: Invalid request").
  it('accepts a misc task with no workflow', async () => {
    const res = await request(app)
      .post('/api/check-constraints')
      .send({ envConfig, schedule: schedule([{ id: 'misc_1', name: 'VISA', region: 'r1', phaseTaskList: [] }]) });
    expect(res.status).toBe(200);
    expect(res.body.violations).toEqual(expect.any(Array));
  });

  it('still accepts a regular task with a workflow', async () => {
    const res = await request(app)
      .post('/api/check-constraints')
      .send({ envConfig, schedule: schedule([{ id: 'wt1', workflow: 'wf1', phaseTaskList: [] }]) });
    expect(res.status).toBe(200);
  });

  it('still rejects a structurally invalid body', async () => {
    const res = await request(app).post('/api/check-constraints').send({ envConfig });
    expect(res.status).toBe(400);
  });
});
