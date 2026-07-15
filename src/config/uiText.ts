// All UI label strings in one place — change language here without touching components

export const UI = {
  // App title
  appTitle: 'Gantt Chart Editor',

  // Menu bar
  fileMenu: 'ファイル',
  editMenu: '編集',
  viewMenu: '表示',
  helpMenu: 'ヘルプ',

  // File menu items
  open: '開く',
  save: '上書き保存',
  saveAs: '名前を付けて保存',

  // File open dialog
  fileOpenDialogTitle: 'ファイルを開く',
  envConfigFileLabel: 'EnvConfig ファイル',
  scheduleFileLabel: 'スケジュール ファイル',
  chooseFile: 'ファイル選択',
  noFileChosen: 'ファイルを選択してください',

  // Toolbar — actions
  undo: '元に戻す',
  redo: 'やり直し',
  addTask: '+ 追加',
  bulkFlexEdit: '柔軟性一括編集',
  search: '検索',
  clear: 'クリア',

  // Toolbar — view toggle
  deviceView: '装置ビュー',
  workerView: 'ワーカービュー',

  // Toolbar — date period
  startDateLabel: '開始',
  endDateLabel: '終了',
  periodSeparator: '〜',

  // Search
  workerNamePlaceholder: '作業者名',
  deviceCodeLabel: '製番',
  deviceAttributeLabel: '属性',

  // Side panel
  sidePanelTitle: '詳細情報',
  noSelectionMessage: 'タスクを選択してください',
  sidePanelWorkerDetailTitle: '作業情報',
  workerLabel: '作業者',
  taskLabel: 'タスク',
  startLabel: '開始日',
  endLabel: '終了日',
  fabLabel: '製番 / FAB',
  phaseOperationLabel: '工程 / 作業',
  dailyWorkHoursLabel: '日別作業時間 (MM/DD)',
  violationTitle: '制約違反',
  violationTargetDateLabel: '対象日',
  invalidDateRangeMessage: '開始日と終了日の関係が不正です。',
  saveButton: '保存',
  deleteConfirm: '選択したタスクを削除しますか？',
  flexibilityLabel: '柔軟性',
  workloadLabel: '工数',
  hoursUnit: '時間',
  deleteButton: '削除',

  // Task add dialog
  addDialogTitle: 'Assignment 追加',
  dialogDeviceLabel: '装置',
  dialogPhaseLabel: '工程',
  dialogOperationLabel: '作業',
  dialogWorkerLabel: '作業者',
  dialogStartLabel: '開始日',
  dialogEndLabel: '終了日',
  dialogHoursLabel: '作業時間 (時間/日)',
  dialogFlexLabel: '計画柔軟性',
  dialogOk: 'OK',
  dialogCancel: 'キャンセル',
  dialogConfirm: '追加',

  // Error dialog
  errorTitle: 'エラー',
  errorClose: '閉じる',

  // Status bar
  fileLoaded: 'ファイルを読み込みました',
  noFile: 'ファイル未読み込み',
  undoCount: (n: number) => `Undo: ${n}`,
  redoCount: (n: number) => `Redo: ${n}`,
  shortcutHint: 'Ctrl+O: 開く  Ctrl+S: 保存  Ctrl+Z: Undo  Delete: 削除',

  // Plan flexibility options
  flexible: 'Flexible',
  flexibleDesc: 'Flexible',
  reluctant: 'Reluctant',
  reluctantDesc: 'Reluctant',
  fixed: 'Fixed',
  fixedDesc: 'Fixed',

  // Bulk flexibility dialog
  bulkDialogTitle: '柔軟性一括編集',
  bulkTargetLabel: '対象',
  bulkTargetSelected: '選択中の作業',
  bulkTargetAll: 'すべての作業',
  bulkFlexLabel: '変更先の柔軟性',
  bulkDateFilter: '指定日付以前に開始する作業のみ',
  bulkApply: 'OK',

  // Gantt row labels
  phaseSummaryRowLabel: '工程',

  // Worker grid
  workerGridCompany: '企業名',
  workerGridName: '氏名',
  workerGridRemarks: '備考欄',
  workerGridManager: '責任者',
  filterTitle: 'Filter',
  filterClearAll: 'すべて解除',
  filterNoValues: '値なし',

  // Global filter bar — Worker View
  wvFilterBarName: 'バー名検索',
  wvFilterBarNamePlaceholder: 'バー名で絞り込み...',
  wvFilterModule: '装置',
  wvFilterPhase: '工程',
  wvFilterFab: 'Fab',
  wvFilterRegion: 'Region',
  wvFilterDateRange: '期間',

  // Global filter bar — Module View
  mvFilterWorker: '作業者',
  mvFilterFab: 'Fab',
  mvFilterRegion: 'Region',
  mvFilterDateRange: '期間',

  // Filter shared
  filterStartDate: '開始',
  filterEndDate: '終了',
  filterDateSep: '〜',
  filterClear: 'クリア',
  filterSelectAll: 'すべて選択',
  filterSearchPlaceholder: '検索...',
  filterItemCount: (n: number) => `${n} 件選択`,
} as const;