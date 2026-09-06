// Session-local identity for objects that need one but don't have a
// natural stable id in the underlying data (currently just assignments —
// see GanttChartEditor_LiveCollabEdit_UndoConflictDesign20260904.md §3).
// Never persisted to the saved YAML file.
export function generateId(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID();
  }
  return `id-${Date.now()}-${Math.random().toString(36).slice(2)}`;
}
