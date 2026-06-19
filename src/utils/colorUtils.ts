// 20-color palette for workflow_task (device) color coding
const PALETTE = [
  '#4e9af1', // blue
  '#f28b2c', // orange
  '#63c26e', // green
  '#e05c5c', // red
  '#a97de8', // purple
  '#45c8c8', // teal
  '#f2c44a', // yellow
  '#e87da0', // pink
  '#6ab5e8', // light blue
  '#e8a56a', // peach
  '#7dd68a', // light green
  '#c96b6b', // dark red
  '#c2a1e8', // lavender
  '#6acfcf', // cyan
  '#e8d06a', // gold
  '#e89bcf', // rose
  '#5db8e8', // sky
  '#e8b87d', // tan
  '#94d494', // mint
  '#e87070', // coral
];

const colorCache = new Map<string, string>();

export function getColorForDevice(workflowTaskId: string): string {
  if (!colorCache.has(workflowTaskId)) {
    const index = colorCache.size % PALETTE.length;
    colorCache.set(workflowTaskId, PALETTE[index]);
  }
  return colorCache.get(workflowTaskId)!;
}

// Phase colors (lighter variants)
const PHASE_COLORS: Record<string, string> = {
  default0: '#f7a9a8', // salmon (like Module Setup in screenshot)
  default1: '#7ecece', // teal (like Hardware Setup)
  default2: '#7dd68a', // green (like Function Setup)
  default3: '#f2c44a',
  default4: '#c2a1e8',
};

export function getColorForPhaseIndex(index: number): string {
  return PHASE_COLORS[`default${index % 5}`] ?? PALETTE[index % PALETTE.length];
}

// Lighten a hex color for hover state
export function lightenColor(hex: string, amount = 20): string {
  const num = parseInt(hex.slice(1), 16);
  const r = Math.min(255, (num >> 16) + amount);
  const g = Math.min(255, ((num >> 8) & 0xff) + amount);
  const b = Math.min(255, (num & 0xff) + amount);
  return `#${((r << 16) | (g << 8) | b).toString(16).padStart(6, '0')}`;
}
