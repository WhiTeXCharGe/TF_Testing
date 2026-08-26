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
  exportExcel: 'Excelエクスポート',

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
  bulkFlexEdit: 'バー可動性',
  search: '検索',
  clear: 'クリア',
  addBarBtn: '+ バー配置',
  addSeibanBtn: '+ 新規製番追加',
  checkingLabel: '⏳ チェック中...',
  constraintCheckBtn: '☑ 制約チェック',
  flightStintsBtn: '✈ 出入国バー',
  sendToSchedulerBtn: '▶ 計画管理ツールへ送信',

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
  flexibilityLabel: 'バー可動性',
  workloadLabel: '工数',
  hoursUnit: '時間',
  deleteButton: '削除',
  fabRegionLabel: 'FAB / Region',
  periodLabel: '期間',
  barColorLabel: 'バーカラー',
  colorPickerTitle: 'クリックしてカラーを選択',
  remarksLabel: '備考',
  remarksPlaceholder: '備考を入力...',
  miscPanelTitle: 'その他作業情報',
  unavailablePanelTitle: '休日情報',
  deleteUnavailableConfirm: 'この休日を削除しますか？',
  workHourTableTitle: '稼働時間 / 日',
  dateColumnLabel: '日付',

  // Task add dialog (バー配置)
  taskAddDialogTitle: 'バー配置',
  dialogDeviceLabel: '装置 *',
  dialogPhaseLabel: '工程 *',
  dialogOperationLabel: '作業 *',
  dialogWorkerLabel: '作業者',
  dialogStartLabel: '開始日 *',
  dialogEndLabel: '終了日 *',
  dialogHoursLabel: '時間/日',
  dialogFlexLabel: 'バー可動性',
  dialogOk: 'OK',
  dialogCancel: 'キャンセル',
  addTypeSectionTitle: '追加タイプ選択',
  addTypeRegular: '通常ワークフロー',
  addTypeMisc: 'その他作業 (Misc)',
  addTypeUnavailable: '休日・不在設定',
  regularSectionTitle: '装置・工程・作業 選択',
  workerScheduleSectionLabel: '作業者・日程',
  addWorkerBtn: '+ 作業者追加',
  miscSectionTitle: 'その他作業 選択',
  miscExistingTaskLabel: '既存のタスク',
  miscNewTaskLabel: '新規タスク作成',
  miscTaskSelectLabel: 'タスク選択 *',
  miscNewTaskNameLabel: '新タスク名 *',
  miscNewTaskNamePlaceholder: 'タスク名を入力',
  unavailWorkersLabel: '対象作業者（複数選択可） *',
  workerSearchPlaceholder: '作業者を検索...',
  workerSelectedCount: (n: number) => `${n}人選択中`,
  workerSelectPrompt: '作業者を選択してください',
  workerFieldLabel: (n: number) => `作業者 ${n} *`,

  // Error dialog
  errorTitle: 'エラー',
  errorClose: '閉じる',

  // Status bar
  fileLoaded: 'ファイルを読み込みました',
  noFile: 'ファイル未読み込み',
  emptyStateInstruction: (menu: string, action: string) => `「${menu}」→「${action}」でファイルを読み込んでください`,
  shortcutOpenHint: 'ショートカット: Ctrl+O',
  undoCount: (n: number) => `Undo: ${n}`,
  redoCount: (n: number) => `Redo: ${n}`,
  shortcutHint: 'Ctrl+O: 開く  Ctrl+S: 保存  Ctrl+Z: Undo  Delete: 削除',

  // Plan flexibility options
  flexible: '可変',
  flexibleDesc: '可変',
  reluctant: '準固定',
  reluctantDesc: '準固定',
  fixed: '固定',
  fixedDesc: '固定',

  // Bulk flexibility dialog (バー可動性)
  bulkDialogTitle: 'バー可動性',
  bulkTargetLabel: '対象',
  bulkTargetSelected: '選択中の作業',
  bulkTargetAll: 'すべての作業',
  bulkFlexLabel: '変更先の可動性',
  bulkDateFilter: '指定日付以前に開始する作業のみ',
  bulkApply: 'OK',
  bulkNoSelectionSuffix: '(未選択)',

  // Plan range edit dialog
  planRangeEditBtn: '計画期間編集',
  planRangeDialogTitle: '計画期間編集',
  planRangeStartLabel: '開始日',
  planRangeEndLabel: '終了日',

  // Gantt row labels
  phaseSummaryRowLabel: '工程',
  monthLabel: (m: number) => `${m}月`,
  dowLabels: ['日', '月', '火', '水', '木', '金', '土'],
  expandColumnsTitle: '詳細列を展開する',
  collapseColumnsTitle: '詳細列を折りたたむ',
  editInPlaceTitle: 'クリックして編集',

  // Device view (製番) side panel
  planStartDateLabel: '作業開始可能日',
  phaseEndDateLabel: '終了希望日',
  actualPeriodLabel: '実績期間',
  assignedWorkerCountLabel: '割り当て作業者',
  peopleUnit: (n: number) => `${n}名`,
  minWorkerLabel: '最小人数',
  maxWorkerLabel: '最大人数',
  workloadHoursLabel: '工数 (h)',
  workerAssignmentSectionLabel: (assigned: number, max: number) => `作業者割り当て (${assigned} / ${max}名)`,
  noAssignmentsLabel: '割り当てなし',

  // New schedule dialog (新規製番追加)
  newScheduleDialogTitle: '新規製番追加',
  tabImportLabel: 'ファイルからインポート',
  tabFormLabel: 'フォームで追加',
  mergeHintLine1: '既存データにマージします。同じIDの製番・作業者・Fabは無視されます。',
  mergeHintLine2: 'どちらか一方だけでも読み込み可能です。',
  scheduleFileMergeLabel: 'Schedule.yaml（製番・割付を追加）',
  envFileMergeLabel: 'EnvConfig.yaml（作業者・Fab等を追加）',
  fileNotChosenShort: '選択なし',
  seibanEntryTitle: (n: number, name: string) => `製番 ${n}${name ? `：${name}` : ''}`,
  expandTitle: '展開',
  collapseTitle: '折りたたむ',
  seibanNameRequiredLabel: '製番名 *',
  seibanNamePlaceholder: '例: SU 1002B',
  workflowRequiredLabel: 'ワークフロー *',
  selectPlaceholder: '--- 選択 ---',
  noneOptionLabel: '--- 選択なし ---',
  fabOptionalLabel: 'Fab',
  workStartDateRequiredLabel: '作業開始可能日 *',
  phaseSettingsSectionTitle: '工程別設定',
  addSeibanEntryBtn: '+ 製番追加',
  workloadHoursLabelCompact: '工数(h)',
  phaseColumnLabel: '工程',
  loadingLabel: '読み込み中...',
  importBtn: 'インポート',
  mergeErrorMessage: (msg: string) => `マージエラー: ${msg}`,
  fileReadErrorMessage: (name: string) => `読み込み失敗: ${name}`,
  fileLoadErrorMessage: (msg: string) => `ファイル読み込みエラー: ${msg}`,
  pickEnvFilePrompt: 'EnvConfig.yaml を選択してください',
  pickScheduleFilePrompt: 'Schedule.yaml を選択してください',
  noFileSelectedError: 'ファイルが選択されませんでした',
  fileReadFailedMessage: (name: string) => `ファイル読み込み失敗: ${name}`,
  schedulerLaunchFailedError: 'SchedulerWebの起動に失敗しました',
  sendFailedError: '送信に失敗しました',
  schedulerDeliveryFailedError: 'SchedulerWebへの送信に失敗しました',

  // Frontend constraint check violation messages
  workHourRangeViolation: (op: string, date: string, hour: number, allowed: string) =>
    `作業時間違反: ${op} ${date} ${hour}h (許容: ${allowed})`,
  workHourOver24Violation: (op: string, date: string, hour: number) =>
    `作業時間違反: ${op} ${date} ${hour}h (>24h)`,
  skillMismatchViolation: (workerName: string, operationId: string, required: number, actual: number) =>
    `スキル不足: worker=${workerName} operation=${operationId} required=${required} actual=${actual}`,
  workerUnavailableViolation: (workerName: string, date: string) =>
    `Worker ${workerName}: 利用不可日に割り当て (${date})`,
  phaseOverrunViolation: (operationTask: string, startDate: string, endDate: string) =>
    `工程開始日・終了日違反: ${operationTask} (${startDate}..${endDate})`,
  regionSuitabilityViolation: (workerName: string, regionId: string) =>
    `地域適性違反: worker=${workerName} region=${regionId} suitability=0`,
  companySuitabilityViolation: (workerName: string, companyId: string) =>
    `企業適性違反: worker=${workerName} company=${companyId} suitability=0`,

  // Save As dialog
  envFileNameLabel: 'EnvConfig ファイル名',
  scheduleFileNameLabel: 'Schedule ファイル名',
  saveAsElectronHint: '保存先はこの後のダイアログで選択します。',
  saveAsBrowserHint: '両ファイルはブラウザのダウンロードフォルダに保存されます。',
  saveConfirmBtn: '保存',

  // Send to scheduler dialog
  sendToSchedulerDialogTitle: '計画管理ツールへ送信',
  sendToSchedulerConfirmBody: '現在の EnvConfig / Schedule を計画管理ツール（Scheduler Webapp）に送信します。計画管理ツールが起動していない場合は自動的に起動します（最大30秒程度かかることがあります）。送信後は計画管理ツール側の「新規実行」ダイアログに両ファイルが自動でセットされますが、実行の最終確認はそちらで行ってください。',
  sendBtn: '送信',
  sendingStatus: '送信中…（計画管理ツールの起動待ちのため、最大30秒程度かかる場合があります）',
  sendDoneMessage: '送信しました。計画管理ツールの新しいタブが開きます。',
  sendDoneManualLinkHint: '自動で開かない場合はこちらをクリックしてください:',
  closeBtn: '閉じる',
  retryBtn: '再試行',
  unsavedChangesBeforeSendMessage: '未保存の変更があります。送信する前に保存してください。',
  saveAndSendFailedMessage: (msg: string) => `保存に失敗しました: ${msg}`,

  // Constraint result dialog
  constraintDialogTitle: '制約チェック結果',
  constraintCheckingBody: 'バックエンドで制約チェック中...',
  noViolations: '✓ 違反なし',
  errorCount: (n: number) => `✕ エラー ${n}件`,
  warningCount: (n: number) => `⚠ 警告 ${n}件`,
  shownCount: (n: number) => `（表示: ${n}件）`,
  allClearMessage: 'すべての制約チェックをクリアしました',
  noMatchingViolations: '該当する違反がありません',
  errorsSectionLabel: 'エラー',
  warningsSectionLabel: '警告',
  rowClickHint: '行をクリックすると割付が選択されます',
  recheckBtn: '☑ 再チェック',
  noOptionsLabel: '選択肢なし',
  chipClearAll: '✕ すべて解除',
  companyLabel: '会社',
  dateRangeLabel: '日付範囲:',
  clearAllFiltersBtn: '✕ すべてクリア',
  colConstraintLabel: '制約',
  badgeError: 'ERR',
  badgeWarning: 'WARN',
  andNMore: (n: number) => `他${n}件`,
  violationLabels: {
    OVERLAP: '同一日重複',
    WORKER_UNAVAILABLE: '作業不可日',
    PHASE_OVERRUN: '工程日付超過',
    WORK_HOUR_RANGE: '作業時間範囲',
    SKILL_MISMATCH: 'スキル不足',
    TASK_WORKER_COUNT: '作業者数',
    PHASE_SEQUENCE: '工程順序',
    WORKLOAD_TOTAL: '必要作業量',
    RESPONSIBLE_WORKER: '作業責任者',
    TRAVEL_DAYS: '移動日',
    REGION_SUITABILITY: '地域適性',
    COMPANY_SUITABILITY: '企業適性',
    OVERTIME: '残業時間',
    STAY_DURATION: '滞在期間',
  } as Record<string, string>,

  // Menu bar status messages
  savePathUnknownMessage: '保存先パスが不明です。名前を付けて保存を使用してください。',
  collabNoScheduleError: 'スケジュールを読み込んでから開始してください',
  savedMessage: '保存しました',
  saveFailedMessage: (msg: string) => `保存失敗: ${msg}`,
  excelExportedMessage: 'Excelエクスポートしました',
  exportFailedMessage: (msg: string) => `エクスポート失敗: ${msg}`,

  // Common / shared widgets
  selectDefaultPlaceholder: '---',
  noMatchLabel: '該当なし',

  // Excel export
  deviceViewSheetName: '製番ビュー',
  phaseOperationColumnLabel: '工程/作業',

  // Backend constraint check
  constraintCheckErrorMessage: (msg: string) => `制約チェックエラー: ${msg}`,
  backendUnreachableError: 'バックエンドに接続できません。サーバーが起動しているか確認してください。',
  incomingTransferErrorMessage: (msg: string) => `計画管理ツールからのデータ受信に失敗しました: ${msg}`,

  // Gantt row labels
  workerGridCompany: '企業名',
  workerGridId: 'ID',
  workerGridName: '氏名',
  workerGridRemarks: '備考欄',
  workerGridManager: '責任者',
  workerGridManagerYes: 'はい',
  extraColWorkType: '業務形態',
  extraColAssignedDuties: '担当職務',
  extraColVisa: 'VISA',
  extraColOverseasDriving: '海外運転',
  hourUnitSuffix: 'h',
  deviceCornerLabel: '装置 / 工程',
  fabFieldLabel: 'Fab',
  regionFieldLabel: 'Region',
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

  // Live View sharing
  shareLiveViewBtn: 'ライブビューを共有',
  stopSharingBtn: '共有を停止',
  shareViewDialogTitle: 'ライブビューの共有',
  shareViewDialogDesc: 'このリンクを開くと、他の人が現在の内容を閲覧専用で見ることができます（編集は不可）。ホストの操作に応じて自動で更新されます。',
  shareViewLinkLoading: 'リンクを生成中…',
  copyLinkBtn: 'コピー',
  copyLinkCopied: 'コピーしました',
  shareViewClose: '閉じる',
} as const;
