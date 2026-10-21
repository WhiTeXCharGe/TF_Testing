/**
 * Solver options offered in the "choose solver" dialog (shown after Submit Run).
 *
 *   timefold            → engine=timefold
 *   sigmai-dwave-scip   → engine=sigmai, stage1=dwave, stage2=scip
 *   sigmai-scip-scip    → engine=sigmai, stage1=scip,  stage2=scip
 *
 * Field names / values match documents/SigmaI/SigmaI_Azure_DetailedDesign20261013.md §5.1
 * (POST /runSolver). Stage 2 is always SCIP — the Sigma-I API rejects anything else.
 */

export type SolverOptionId = 'timefold' | 'sigmai-dwave-scip' | 'sigmai-scip-scip';
export type SolverEngine = 'timefold' | 'sigmai';

export interface SolverSelection {
  option: SolverOptionId;
  /** Sigma-I only. Seconds. */
  stage1TimeoutSec?: number;
  /** Sigma-I only. Seconds. */
  stage2TimeoutSec?: number;
}

export interface SolverOptionDef {
  id: SolverOptionId;
  engine: SolverEngine;
  stage1Solver?: 'scip' | 'dwave';
  stage2Solver?: 'scip';
  defaultStage1TimeoutSec?: number;
  defaultStage2TimeoutSec?: number;
}

/** Guard rails — same defaults as the design doc (Q7 there: final values TBD after P0 testing). */
export const MIN_TIMEOUT_SEC = 1;
export const MAX_TIMEOUT_SEC = 7200;

export const SOLVER_OPTIONS: readonly SolverOptionDef[] = [
  { id: 'timefold', engine: 'timefold' },
  {
    id: 'sigmai-dwave-scip', engine: 'sigmai',
    stage1Solver: 'dwave', stage2Solver: 'scip',
    defaultStage1TimeoutSec: 600, defaultStage2TimeoutSec: 1800,
  },
  {
    id: 'sigmai-scip-scip', engine: 'sigmai',
    stage1Solver: 'scip', stage2Solver: 'scip',
    defaultStage1TimeoutSec: 1800, defaultStage2TimeoutSec: 1800,
  },
];

export const DEFAULT_SOLVER_OPTION: SolverOptionId = 'timefold';

export function getSolverOption(id: SolverOptionId): SolverOptionDef {
  const def = SOLVER_OPTIONS.find(o => o.id === id);
  if (!def) throw new Error(`Unknown solver option: ${id}`);
  return def;
}

/** Default selection for an option (timeouts pre-filled for Sigma-I). */
export function defaultSelection(id: SolverOptionId): SolverSelection {
  const def = getSolverOption(id);
  return {
    option: id,
    stage1TimeoutSec: def.defaultStage1TimeoutSec,
    stage2TimeoutSec: def.defaultStage2TimeoutSec,
  };
}

export function isValidTimeout(n: unknown): n is number {
  return typeof n === 'number' && Number.isInteger(n) && n >= MIN_TIMEOUT_SEC && n <= MAX_TIMEOUT_SEC;
}

/**
 * Extra multipart fields to send with POST /runSolver.
 * Timefold sends only `engine=timefold`; nothing else is added so the request stays
 * identical to today's apart from that one field.
 */
export function toRequestFields(sel: SolverSelection): Record<string, string> {
  const def = getSolverOption(sel.option);
  if (def.engine === 'timefold') return { engine: 'timefold' };
  return {
    engine: 'sigmai',
    stage1Solver: def.stage1Solver!,
    stage2Solver: def.stage2Solver!,
    stage1TimeoutSec: String(sel.stage1TimeoutSec ?? def.defaultStage1TimeoutSec),
    stage2TimeoutSec: String(sel.stage2TimeoutSec ?? def.defaultStage2TimeoutSec),
  };
}
