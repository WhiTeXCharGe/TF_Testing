import { useMemo } from 'react';
import { useAppContext } from '../../context/AppContext';
import { GanttChartArea, GanttRow } from './GanttChartArea';
import { getColorForDevice } from '../../utils/colorUtils';
import { WorkflowTask } from '../../types/schedule';

interface Props {
  dates: string[];
}

export function WorkerViewGantt({ dates }: Props) {
  const { state, dispatch } = useAppContext();
  const { schedule, envConfig, selectedAssignmentIndex, violations } = state;

  const violationIndices = useMemo(
    () => new Set(violations.flatMap(v => v.assignmentIndices)),
    [violations],
  );

  // Build a lookup: operationTask id → workflowTask id (for color coding by device)
  const opTaskToDeviceId = useMemo(() => {
    const map = new Map<string, string>();
    schedule?.workflowTaskList.forEach(wt =>
      wt.phaseTaskList.forEach(pt =>
        pt.operationTaskList.forEach(ot => map.set(ot.id, wt.id)),
      ),
    );
    return map;
  }, [schedule]);

  const rows = useMemo((): GanttRow[] => {
    if (!schedule || !envConfig) return [];
    const result: GanttRow[] = [];

    // Build worker rows — use envConfig order, skip workers with no assignments
    for (const worker of envConfig.workerList) {
      const workerAssignments = schedule.assignmentList
        .map((a, i) => ({ a, i }))
        .filter(({ a }) => a.worker === worker.id);

      if (workerAssignments.length === 0) continue;

      const workerLabel = worker.name ? `${worker.name} (${worker.id})` : worker.id;

      result.push({
        id: `worker_${worker.id}`,
        label: workerLabel,
        indent: 0,
        isExpandable: false,
        isExpanded: false,
        bars: workerAssignments.map(({ a, i }) => {
          const deviceId = opTaskToDeviceId.get(a.operationTask) ?? '';
          return {
            id: `assignment_${i}`,
            assignmentIndex: i,
            label: getDeviceLabel(deviceId, schedule.workflowTaskList),
            startDate: a.startDate,
            endDate: a.endDate,
            color: getColorForDevice(deviceId),
            isSelected: i === selectedAssignmentIndex,
            hasViolation: violationIndices.has(i),
          };
        }),
      });
    }

    // Also show workers not in envConfig (edge case)
    const knownWorkerIds = new Set(envConfig.workerList.map(w => w.id));
    const unknownAssignments = schedule.assignmentList
      .map((a, i) => ({ a, i }))
      .filter(({ a }) => !knownWorkerIds.has(a.worker));

    const byUnknownWorker = new Map<string, typeof unknownAssignments>();
    for (const item of unknownAssignments) {
      const group = byUnknownWorker.get(item.a.worker) ?? [];
      group.push(item);
      byUnknownWorker.set(item.a.worker, group);
    }
    for (const [workerId, items] of byUnknownWorker) {
      result.push({
        id: `worker_${workerId}`,
        label: workerId,
        indent: 0,
        isExpandable: false,
        isExpanded: false,
        bars: items.map(({ a, i }) => {
          const deviceId = opTaskToDeviceId.get(a.operationTask) ?? '';
          return {
            id: `assignment_${i}`,
            assignmentIndex: i,
            label: getDeviceLabel(deviceId, schedule.workflowTaskList),
            startDate: a.startDate,
            endDate: a.endDate,
            color: getColorForDevice(deviceId),
            isSelected: i === selectedAssignmentIndex,
            hasViolation: violationIndices.has(i),
          };
        }),
      });
    }

    return result;
  }, [schedule, envConfig, selectedAssignmentIndex, violationIndices, opTaskToDeviceId]);

  const handleBarClick = (index: number) => {
    dispatch({ type: 'SELECT_ASSIGNMENT', payload: index === selectedAssignmentIndex ? null : index });
  };

  const handleBarDragEnd = (index: number, newStart: string, newEnd: string) => {
    dispatch({ type: 'UPDATE_ASSIGNMENT', payload: { index, updates: { startDate: newStart, endDate: newEnd } } });
  };

  return (
    <GanttChartArea
      rows={rows}
      dates={dates}
      onToggleRow={() => {}}
      onBarClick={handleBarClick}
      onBarDragEnd={handleBarDragEnd}
    />
  );
}

function getDeviceLabel(deviceId: string, workflowTasks: WorkflowTask[]): string {
  const wt = workflowTasks.find(w => w.id === deviceId);
  return wt?.name ?? deviceId;
}
