// Number of working hours in one day — used to convert workload_days → workload_hours
export const HOURS_PER_DAY = 8;

// Pixel width of each day column in the Gantt grid
export const CELL_WIDTH = 40;

// Height of each Gantt row in pixels
export const ROW_HEIGHT = 36;

// Width of the row header (left panel) in pixels
export const ROW_HEADER_WIDTH = 220;

// Maximum number of undo steps
export const MAX_UNDO_STACK = 100;

// Service API base URL (Node.js service, used for webapp integration in Phase 5)
export const SERVICE_BASE_URL = 'http://localhost:3001';

// Whether to shade weekend columns differently in the Gantt grid
export const SHOW_WEEKEND_SHADING = false;
