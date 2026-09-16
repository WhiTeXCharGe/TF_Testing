// Kept separate from moduleViewModel.test.ts, which is excluded from the
// suite (jest.config.cjs testPathIgnorePatterns) because its buildModuleViewModel
// fixtures crash the jest worker with an out-of-memory error — pre-existing,
// unrelated to this helper. unassignedRange doesn't touch buildModuleViewModel
// at all, so it's safe to test on its own.
import { unassignedRange, ModulePhase } from '../../components/GanttChart/moduleViewModel';

const phase = (over: Partial<ModulePhase>): ModulePhase => ({
  moduleId: 'm1', phaseId: 'p', phaseName: 'P', planStartDate: '2025-09-01', planEndDate: '2025-09-10',
  barStartDate: null, barEndDate: null, workerCount: 0, tasks: [], color: '#fff', ...over,
});

describe('unassignedRange', () => {
  it('returns null when every phase already has a worker assigned', () => {
    const phases = [phase({ phaseId: 'a', workerCount: 1 }), phase({ phaseId: 'b', workerCount: 2 })];
    expect(unassignedRange(phases)).toBeNull();
  });

  it('returns null for an empty phase list', () => {
    expect(unassignedRange([])).toBeNull();
  });

  it("spans a single unassigned phase's own planned range", () => {
    const phases = [phase({ phaseId: 'a', planStartDate: '2025-09-05', planEndDate: '2025-09-20', workerCount: 0 })];
    expect(unassignedRange(phases)).toEqual({ start: '2025-09-05', end: '2025-09-20' });
  });

  it('combines multiple unassigned phases into their min-start/max-end envelope', () => {
    const phases = [
      phase({ phaseId: 'a', planStartDate: '2025-09-10', planEndDate: '2025-09-15', workerCount: 0 }),
      phase({ phaseId: 'b', planStartDate: '2025-09-01', planEndDate: '2025-09-05', workerCount: 0 }),
      phase({ phaseId: 'c', planStartDate: '2025-09-12', planEndDate: '2025-09-30', workerCount: 0 }),
    ];
    expect(unassignedRange(phases)).toEqual({ start: '2025-09-01', end: '2025-09-30' });
  });

  it('only folds in unassigned phases, ignoring assigned ones even if their dates are wider', () => {
    const phases = [
      phase({ phaseId: 'assigned', planStartDate: '2025-01-01', planEndDate: '2025-12-31', workerCount: 3 }),
      phase({ phaseId: 'unassigned', planStartDate: '2025-09-05', planEndDate: '2025-09-08', workerCount: 0 }),
    ];
    expect(unassignedRange(phases)).toEqual({ start: '2025-09-05', end: '2025-09-08' });
  });
});
