import { useMemo } from 'react';
import { useAppContext } from '../../context/AppContext';
import { GanttChartArea, GanttRow, GanttBar } from './GanttChartArea';
import { getColorForPhaseIndex } from '../../utils/colorUtils';
import { Worker } from '../../types/envConfig';

interface Props {
  dates: string[];
}

export function DeviceViewGantt({ dates }: Props) {
  const { state, dispatch } = useAppContext();
  const { schedule, envConfig, expandedDeviceIds, selectedAssignmentIndex, violations } = state;

  const violationIndices = useMemo(
    () => new Set(violations.flatMap(v => v.assignmentIndices)),
    [violations],
  );

  const rows = useMemo((): GanttRow[] => {
    if (!schedule) return [];
    const result: GanttRow[] = [];

    for (const wt of schedule.workflowTaskList) {
      const isExpanded = expandedDeviceIds.has(wt.id);
      const deviceLabel = wt.name ? `${wt.name} (${wt.id})` : wt.id;

      if (!isExpanded) {
        // Collapsed: one row per device, show all phase summary bars
        const collapsedBars: GanttBar[] = [];
        wt.phaseTaskList.forEach((pt, phaseIdx) => {
          const phaseColor = getColorForPhaseIndex(phaseIdx);
          const phaseAssignments = schedule.assignmentList
            .map((a, i) => ({ a, i }))
            .filter(({ a }) => pt.operationTaskList.some(ot => ot.id === a.operationTask));

          if (phaseAssignments.length > 0) {
            const phaseStart = phaseAssignments.map(({ a }) => a.startDate).reduce(minStr);
            const phaseEnd = phaseAssignments.map(({ a }) => a.endDate).reduce(maxStr);
            collapsedBars.push({
              id: `${wt.id}_${pt.id}_col`,
              assignmentIndex: phaseAssignments[0].i,
              label: pt.name ?? pt.id,
              startDate: phaseStart,
              endDate: phaseEnd,
              color: phaseColor,
              isSelected: phaseAssignments.some(({ i }) => i === selectedAssignmentIndex),
              hasViolation: phaseAssignments.some(({ i }) => violationIndices.has(i)),
            });
          } else if (pt.startDate && pt.endDate) {
            collapsedBars.push({
              id: `${wt.id}_${pt.id}_empty`,
              assignmentIndex: -1,
              label: pt.name ?? pt.id,
              startDate: pt.startDate,
              endDate: pt.endDate,
              color: `${phaseColor}66`,
              isSelected: false,
              hasViolation: false,
            });
          }
        });

        result.push({
          id: wt.id, label: deviceLabel, indent: 0,
          isHeader: true, isExpandable: true, isExpanded: false,
          bars: collapsedBars,
        });

      } else {
        // Expanded: device header + phase summary row + operation slot rows
        result.push({
          id: wt.id, label: deviceLabel, indent: 0,
          isHeader: true, isExpandable: true, isExpanded: true,
          bars: [],
        });

        // ── Phase summary row: one bar per phase ─────────────────────
        const phaseSummaryBars: GanttBar[] = [];
        wt.phaseTaskList.forEach((pt, phaseIdx) => {
          const phaseColor = getColorForPhaseIndex(phaseIdx);
          const phaseAssignments = schedule.assignmentList
            .map((a, i) => ({ a, i }))
            .filter(({ a }) => pt.operationTaskList.some(ot => ot.id === a.operationTask));

          if (phaseAssignments.length > 0) {
            const phaseStart = phaseAssignments.map(({ a }) => a.startDate).reduce(minStr);
            const phaseEnd = phaseAssignments.map(({ a }) => a.endDate).reduce(maxStr);
            phaseSummaryBars.push({
              id: `${wt.id}_${pt.id}_summary`,
              assignmentIndex: phaseAssignments[0].i,
              label: pt.name ?? pt.id,
              startDate: phaseStart,
              endDate: phaseEnd,
              color: getColorForPhaseIndex(phaseIdx),
              isSelected: phaseAssignments.some(({ i }) => i === selectedAssignmentIndex),
              hasViolation: phaseAssignments.some(({ i }) => violationIndices.has(i)),
            });
          }
        });

        result.push({
          id: `${wt.id}__phases`,
          label: '工程',
          indent: 1,
          isExpandable: false,
          isExpanded: false,
          bars: phaseSummaryBars,
        });

        // ── Operation slot rows: Nth row = Nth operation across phases ─
        const maxOps = wt.phaseTaskList.reduce(
          (m, pt) => Math.max(m, pt.operationTaskList.length), 0,
        );

        for (let slot = 0; slot < maxOps; slot++) {
          const slotBars: GanttBar[] = [];

          // Use the operation name from the first phase that has this slot
          let rowLabel = `作業 ${slot + 1}`;

          wt.phaseTaskList.forEach((pt, phaseIdx) => {
            const opTask = pt.operationTaskList[slot];
            if (!opTask) return;
            const phaseColor = getColorForPhaseIndex(phaseIdx);

            // Look up the operation name from envConfig if available
            const opName = opTask.name ?? lookupOpName(opTask.operation, envConfig?.workflowList ?? []) ?? opTask.operation;
            if (slot < 1 || rowLabel === `作業 ${slot + 1}`) rowLabel = opName;

            const opAssignments = schedule.assignmentList
              .map((a, i) => ({ a, i }))
              .filter(({ a }) => a.operationTask === opTask.id);

            if (opAssignments.length > 0) {
              // Aggregate into one bar spanning all assignments for this operation
              const opStart = opAssignments.map(({ a }) => a.startDate).reduce(minStr);
              const opEnd = opAssignments.map(({ a }) => a.endDate).reduce(maxStr);
              const workerNames = opAssignments
                .map(({ a }) => getWorkerName(a.worker, envConfig?.workerList ?? []))
                .join(', ');

              slotBars.push({
                id: `${wt.id}_${pt.id}_${opTask.id}_slot`,
                assignmentIndex: opAssignments[0].i,
                label: `${opName} (${workerNames})`,
                startDate: opStart,
                endDate: opEnd,
                color: phaseColor,
                isSelected: opAssignments.some(({ i }) => i === selectedAssignmentIndex),
                hasViolation: opAssignments.some(({ i }) => violationIndices.has(i)),
              });
            }
          });

          result.push({
            id: `${wt.id}__slot${slot}`,
            label: rowLabel,
            indent: 2,
            isExpandable: false,
            isExpanded: false,
            bars: slotBars,
          });
        }
      }
    }
    return result;
  }, [schedule, expandedDeviceIds, selectedAssignmentIndex, violationIndices, envConfig]);

  const handleToggle = (id: string) => dispatch({ type: 'TOGGLE_DEVICE', payload: id });

  const handleBarClick = (index: number) => {
    if (index < 0) return;
    dispatch({ type: 'SELECT_ASSIGNMENT', payload: index === selectedAssignmentIndex ? null : index });
  };

  const handleBarDragEnd = (index: number, newStart: string, newEnd: string) => {
    dispatch({ type: 'UPDATE_ASSIGNMENT', payload: { index, updates: { startDate: newStart, endDate: newEnd } } });
  };

  return (
    <GanttChartArea
      rows={rows}
      dates={dates}
      onToggleRow={handleToggle}
      onBarClick={handleBarClick}
      onBarDragEnd={handleBarDragEnd}
    />
  );
}

function minStr(a: string, b: string) { return a < b ? a : b; }
function maxStr(a: string, b: string) { return a > b ? a : b; }

function getWorkerName(workerId: string, workerList: Worker[]): string {
  return workerList.find(w => w.id === workerId)?.name ?? workerId;
}

function lookupOpName(
  operationId: string,
  workflowList: { phaseList: { operationList: { id: string; name?: string }[] }[] }[],
): string | undefined {
  for (const wf of workflowList) {
    for (const ph of wf.phaseList) {
      const op = ph.operationList.find(o => o.id === operationId);
      if (op?.name) return op.name;
    }
  }
  return undefined;
}
