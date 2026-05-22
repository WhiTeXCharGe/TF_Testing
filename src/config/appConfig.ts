// Application-level constants.
// Do not import from here in config/uiConfig.ts (keep them separate).

export const APP_CONFIG = {
  // Backend / solver API base URL.
  // Set VITE_API_BASE_URL in .env to point at your Cloud Run job endpoint.
  apiBaseUrl: import.meta.env.VITE_API_BASE_URL ?? '',

  // Path under /public where static data files live.
  dataBasePath: '/data',

  // Excel database file name under /public/data/
  databaseFileName: 'database.xlsx',

  // Sheet names inside database.xlsx
  sheets: {
    datasets: 'Datasets',
    runLogs: 'RunLogs',
    comments: 'Comments',
  },

  // Folder name under /public/data/datasets/{datasetId}/
  datasetsFolder: 'datasets',

  // File names expected inside each dataset folder
  envConfigFile: 'EnvConfig.yaml',
  scheduleFile: 'Schedule.yaml',

  // Feature flags
  features: {
    editGantt: false,          // Gantt drag-to-edit (not yet implemented)
    constraintPanel: true,     // Show constraint violations panel
    notifications: false,      // Browser push notifications
  },

  // Local storage keys
  storage: {
    dbCache: 'tfScheduler_dbCache',
    dbCacheTs: 'tfScheduler_dbCacheTs',
  },

  // Cache TTL in milliseconds (5 minutes)
  cacheTtlMs: 5 * 60 * 1000,
} as const;
