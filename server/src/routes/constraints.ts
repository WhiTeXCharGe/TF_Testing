import { Router } from 'express';
import { z } from 'zod';
import { runBackendConstraints } from '../services/backendConstraints.js';

export const constraintsRouter = Router();

// ── Zod schema (permissive — accepts any extra fields) ──────────────────────

const WorkDateSchema = z.object({ date: z.string(), hour: z.number() });
const AssignmentSchema = z.object({
  worker: z.string(),
  operationTask: z.string(),
  startDate: z.string(),
  endDate: z.string(),
  workDateList: z.array(WorkDateSchema),
  planFlexibility: z.string().optional().default('Flexible'),
});
const OperationTaskSchema = z.object({
  id: z.string(),
  name: z.string().optional(),
  operation: z.string(),
  workloadHours: z.number().default(0),
  recommendsWorkerMin: z.number().optional(),
  recommendsWorkerMax: z.number().optional(),
});
const PhaseTaskSchema = z.object({
  id: z.string(),
  name: z.string().optional(),
  phase: z.string(),
  startDate: z.string(),
  endDate: z.string(),
  operationTaskList: z.array(OperationTaskSchema),
});
const WorkflowTaskSchema = z.object({
  id: z.string(),
  name: z.string().optional(),
  workflow: z.string(),
  fab: z.string().optional(),
  region: z.string().optional(),
  phaseTaskList: z.array(PhaseTaskSchema),
});
const ScheduleSchema = z.object({
  planRange: z.object({ startDate: z.string(), endDate: z.string() }),
  workflowTaskList: z.array(WorkflowTaskSchema),
  assignmentList: z.array(AssignmentSchema),
});

const WorkerSchema = z.object({
  id: z.string(),
  name: z.string().optional(),
  isManager: z.boolean().optional(),
  workerCompany: z.string().optional(),
  skillMap: z.record(z.number()).optional(),
  unavailableDates: z.array(z.unknown()).default([]),
});
const FabSchema = z.object({ id: z.string(), name: z.string().optional(), region: z.string().optional() });
const RegionSchema = z.object({
  id: z.string(),
  name: z.string().optional(),
  maxStayOn: z.number().optional(),
  maxAnnualStay: z.number().optional(),
  stayOffInterval: z.number().optional(),
});
const TransitSchema = z.object({ from: z.string(), to: z.string(), days: z.number() });
const WorkerCompanySchema = z.object({
  id: z.string(),
  name: z.string().optional(),
  annualOvertimeLimit: z.number().optional(),
  monthlyOvertimeLimit: z.number().optional(),
});
const WorkflowSchema = z.object({
  id: z.string(),
  name: z.string().optional(),
  phaseList: z.array(z.object({
    id: z.string(),
    name: z.string().optional(),
    operationList: z.array(z.object({
      id: z.string(),
      name: z.string().optional(),
      workHours: z.array(z.number()).optional(),
      workloadHours: z.number().optional(),
      minWorkerNum: z.number().optional(),
      maxWorkerNum: z.number().optional(),
    })).default([]),
  })).default([]),
});
const EnvConfigSchema = z.object({
  workflowList: z.array(WorkflowSchema).default([]),
  fabList: z.array(FabSchema).default([]),
  regionList: z.array(RegionSchema).default([]),
  workerList: z.array(WorkerSchema).default([]),
  transiteDayMap: z.array(TransitSchema).default([]),
  customerCompanyList: z.array(z.unknown()).default([]),
  workerCompanyList: z.array(WorkerCompanySchema).default([]),
});

const RequestSchema = z.object({
  envConfig: EnvConfigSchema,
  schedule: ScheduleSchema,
});

// ── POST /api/check-constraints ──────────────────────────────────────────────

constraintsRouter.post('/check-constraints', (req, res) => {
  const parsed = RequestSchema.safeParse(req.body);
  if (!parsed.success) {
    res.status(400).json({ error: 'Invalid request', details: parsed.error.flatten() });
    return;
  }

  try {
    const violations = runBackendConstraints(parsed.data.envConfig, parsed.data.schedule);
    res.json({ violations, checkedAt: new Date().toISOString() });
  } catch (err) {
    console.error('[check-constraints] error:', err);
    res.status(500).json({ error: 'Constraint check failed' });
  }
});
