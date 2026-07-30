/**
 * Export the current schedule to a styled Excel (.xlsx) workbook — two tabs,
 * ワーカービュー (Worker view) then 製番ビュー (Device view), mirroring the
 * on-screen Gantt views' rows/columns/colors as closely as a static
 * spreadsheet allows. Reuses the same view-model builders the on-screen
 * views use (buildWorkerTimelineModel / buildModuleViewModel) so colors and
 * labels automatically stay in sync with what's actually rendered.
 *
 * Cell-styling approach adapted from SchedulerWeb's src/utils/excelExport.ts
 * (ExcelJS + file-saver, 3-row month/day/weekday header, frozen panes).
 */
import * as ExcelJS from 'exceljs';
import { saveAs } from 'file-saver';
import { EnvConfig } from '../types/envConfig';
import { ScheduleData } from '../types/schedule';
import {
  buildWorkerTimelineModel,
  WorkerTimelineModel,
  WorkerMetaInfo,
} from '../components/GanttChart/workerViewModel';
import { buildModuleViewModel, ModuleViewModel } from '../components/GanttChart/moduleViewModel';
import { UI } from '../config/uiText';

const DAY_COL_WIDTH = 3.5;
const HEADER_ROWS = 3;
const DOW_JA = UI.dowLabels;

function toArgb(hex: string): string {
  return 'FF' + hex.replace('#', '').toUpperCase();
}

function isWeekend(dateStr: string): boolean {
  const dow = new Date(`${dateStr}T00:00:00`).getDay();
  return dow === 0 || dow === 6;
}

function headerStyle(argb = 'FFE8F0FE'): Partial<ExcelJS.Style> {
  return {
    fill: { type: 'pattern', pattern: 'solid', fgColor: { argb } },
    font: { bold: true, size: 9, name: 'Meiryo' },
    alignment: { horizontal: 'center', vertical: 'middle' },
    border: {
      bottom: { style: 'thin', color: { argb: 'FFDADCE0' } },
      right: { style: 'thin', color: { argb: 'FFDADCE0' } },
    },
  };
}

function dataStyle(fillArgb?: string, opts?: { bold?: boolean; align?: 'left' | 'center' }): Partial<ExcelJS.Style> {
  return {
    fill: fillArgb
      ? { type: 'pattern', pattern: 'solid', fgColor: { argb: fillArgb } }
      : { type: 'pattern', pattern: 'none' },
    font: { size: 9, name: 'Meiryo', bold: !!opts?.bold },
    alignment: { horizontal: opts?.align ?? 'center', vertical: 'middle' },
    border: {
      bottom: { style: 'hair', color: { argb: 'FFDADCE0' } },
      right: { style: 'hair', color: { argb: 'FFDADCE0' } },
    },
  };
}

/** Writes the 3-row month / day-number / day-of-week header shared by both sheets. */
function writeDateHeader(
  ws: ExcelJS.Worksheet,
  metaCols: { label: string; width: number }[],
  monthGroups: { label: string; startIndex: number; span: number }[],
  dates: string[],
): void {
  const metaCount = metaCols.length;
  metaCols.forEach((c, i) => { ws.getColumn(i + 1).width = c.width; });
  dates.forEach((_, di) => { ws.getColumn(metaCount + 1 + di).width = DAY_COL_WIDTH; });

  const row1 = ws.getRow(1);
  const row2 = ws.getRow(2);
  const row3 = ws.getRow(3);
  // Meta columns (Company/ID/Name/...) get one label spanning all 3 header
  // rows — only the date columns actually need 3 separate rows (month/day/weekday).
  metaCols.forEach((c, i) => {
    const col = i + 1;
    ws.mergeCells(1, col, HEADER_ROWS, col);
    const cell = row1.getCell(col);
    cell.value = c.label;
    Object.assign(cell, headerStyle());
  });

  for (const g of monthGroups) {
    const startCol = metaCount + 1 + g.startIndex;
    const endCol = startCol + g.span - 1;
    if (endCol > startCol) ws.mergeCells(1, startCol, 1, endCol);
    const cell = row1.getCell(startCol);
    cell.value = g.label;
    Object.assign(cell, headerStyle('FF1A73E8'));
    cell.font = { ...cell.font, color: { argb: 'FFFFFFFF' } };
  }

  dates.forEach((d, di) => {
    const col = metaCount + 1 + di;
    const we = isWeekend(d);
    const dayNum = new Date(`${d}T00:00:00`).getDate();
    const dowIdx = new Date(`${d}T00:00:00`).getDay();

    const c2 = row2.getCell(col);
    c2.value = dayNum;
    Object.assign(c2, headerStyle(we ? 'FFFFF5F5' : 'FFE8F0FE'));

    const c3 = row3.getCell(col);
    c3.value = DOW_JA[dowIdx];
    Object.assign(c3, headerStyle(we ? 'FFFFF5F5' : 'FFE8F0FE'));
    if (we) c3.font = { ...c3.font, color: { argb: 'FFB54747' } };
  });

  ws.autoFilter = {
    from: { row: HEADER_ROWS, column: 1 },
    to: { row: HEADER_ROWS, column: metaCount + dates.length },
  };
}

// ── Worker sheet ──────────────────────────────────────────────────────────

const WORKER_META_COLS: { key: keyof WorkerMetaInfo; label: string; width: number }[] = [
  { key: 'company', label: UI.companyLabel, width: 14 },
  { key: 'id', label: 'ID', width: 8 },
  { key: 'name', label: UI.workerGridName, width: 12 },
  { key: 'manager', label: UI.workerGridManager, width: 8 },
  { key: 'remarks', label: UI.remarksLabel, width: 14 },
  { key: 'workType', label: UI.extraColWorkType, width: 12 },
  { key: 'assignedDuties', label: UI.extraColAssignedDuties, width: 20 },
  { key: 'visa', label: 'VISA', width: 8 },
  { key: 'overseasDriving', label: UI.extraColOverseasDriving, width: 10 },
];

function buildWorkerSheet(wb: ExcelJS.Workbook, model: WorkerTimelineModel, dates: string[]): void {
  const metaCount = WORKER_META_COLS.length;
  const ws = wb.addWorksheet(UI.workerView, {
    views: [{ state: 'frozen', xSplit: metaCount, ySplit: HEADER_ROWS }],
  });

  writeDateHeader(ws, WORKER_META_COLS, model.monthGroups, dates);

  model.rows.forEach((row, ri) => {
    const exRow = ws.getRow(HEADER_ROWS + 1 + ri);

    WORKER_META_COLS.forEach((c, ci) => {
      const cell = exRow.getCell(ci + 1);
      cell.value = row.meta[c.key];
      Object.assign(cell, dataStyle(undefined, { align: ci <= 2 ? 'left' : 'center' }));
    });

    // Default-fill every day cell first so gaps aren't left unstyled, then
    // overwrite with merged, colored segment cells (mirrors the on-screen
    // merged bars — one styled+labeled cell per contiguous assignment run).
    for (let di = 0; di < dates.length; di++) {
      Object.assign(exRow.getCell(metaCount + 1 + di), dataStyle(undefined));
    }

    for (const seg of row.segments) {
      const startCol = metaCount + 1 + seg.startIndex;
      const endCol = metaCount + 1 + seg.endIndex;
      if (endCol > startCol) ws.mergeCells(exRow.number, startCol, exRow.number, endCol);
      const cell = exRow.getCell(startCol);
      cell.value = seg.label || undefined;
      Object.assign(cell, dataStyle(toArgb(seg.color)));
      cell.font = { ...cell.font, color: { argb: toArgb(seg.textColor) } };
      // Fixed assignments render square-cornered on screen vs. rounded for
      // others — Excel fills can't express that, so a heavier (double)
      // border stands in for "Fixed" instead of silently dropping the signal.
      if (seg.kind === 'work' && seg.planFlexibility === 'Fixed') {
        cell.border = {
          ...cell.border,
          top: { style: 'double', color: { argb: 'FF555555' } },
          bottom: { style: 'double', color: { argb: 'FF555555' } },
        };
      }
    }
  });
}

// ── Device (製番) sheet ───────────────────────────────────────────────────

const DEVICE_META_COLS = [
  { label: UI.deviceCodeLabel, width: 18 },
  { label: UI.deviceAttributeLabel, width: 14 },
  { label: UI.phaseOperationColumnLabel, width: 20 },
];

function dateColIndex(dateStr: string | null, dates: string[]): number {
  if (!dateStr) return -1;
  return dates.indexOf(dateStr);
}

function writeDeviceBar(
  ws: ExcelJS.Worksheet,
  rowNum: number,
  metaCount: number,
  startDate: string | null,
  endDate: string | null,
  dates: string[],
  color: string,
  label: string,
  bold: boolean,
): void {
  let startIdx = dateColIndex(startDate, dates);
  let endIdx = dateColIndex(endDate, dates);
  if (startIdx === -1 && startDate) startIdx = startDate < dates[0] ? 0 : -1;
  if (endIdx === -1 && endDate) endIdx = endDate > dates[dates.length - 1] ? dates.length - 1 : -1;
  if (startIdx === -1 || endIdx === -1 || endIdx < startIdx) return;

  const startCol = metaCount + 1 + startIdx;
  const endCol = metaCount + 1 + endIdx;
  if (endCol > startCol) ws.mergeCells(rowNum, startCol, rowNum, endCol);
  const cell = ws.getRow(rowNum).getCell(startCol);
  cell.value = label;
  Object.assign(cell, dataStyle(toArgb(color), { bold }));
}

function buildDeviceSheet(wb: ExcelJS.Workbook, model: ModuleViewModel, dates: string[]): void {
  const metaCount = DEVICE_META_COLS.length;
  const ws = wb.addWorksheet(UI.deviceViewSheetName, {
    views: [{ state: 'frozen', xSplit: metaCount, ySplit: HEADER_ROWS }],
  });

  writeDateHeader(ws, DEVICE_META_COLS, model.monthGroups, dates);

  let rowNum = HEADER_ROWS + 1;
  for (const mod of model.modules) {
    const moduleRow = ws.getRow(rowNum);
    moduleRow.getCell(1).value = mod.moduleName;
    moduleRow.getCell(2).value = mod.workflowName;
    moduleRow.getCell(3).value = '';
    Object.assign(moduleRow.getCell(1), dataStyle('FFE8EEF5', { bold: true, align: 'left' }));
    Object.assign(moduleRow.getCell(2), dataStyle('FFE8EEF5', { align: 'left' }));
    Object.assign(moduleRow.getCell(3), dataStyle('FFE8EEF5'));
    for (let di = 0; di < dates.length; di++) {
      Object.assign(moduleRow.getCell(metaCount + 1 + di), dataStyle('FFE8EEF5'));
    }
    moduleRow.outlineLevel = 0;
    rowNum += 1;

    for (const phase of mod.phases) {
      const phaseRow = ws.getRow(rowNum);
      phaseRow.getCell(3).value = phase.phaseName;
      Object.assign(phaseRow.getCell(1), dataStyle(undefined));
      Object.assign(phaseRow.getCell(2), dataStyle(undefined));
      Object.assign(phaseRow.getCell(3), dataStyle(undefined, { bold: true, align: 'left' }));
      for (let di = 0; di < dates.length; di++) {
        Object.assign(phaseRow.getCell(metaCount + 1 + di), dataStyle(undefined));
      }
      writeDeviceBar(
        ws, rowNum, metaCount,
        phase.barStartDate ?? phase.planStartDate,
        phase.barEndDate ?? phase.planEndDate,
        dates, phase.color, phase.phaseName, true,
      );
      phaseRow.outlineLevel = 1;
      phaseRow.hidden = true;
      rowNum += 1;

      for (const task of phase.tasks) {
        const taskRow = ws.getRow(rowNum);
        taskRow.getCell(3).value = task.taskName;
        Object.assign(taskRow.getCell(1), dataStyle(undefined));
        Object.assign(taskRow.getCell(2), dataStyle(undefined));
        Object.assign(taskRow.getCell(3), dataStyle(undefined, { align: 'left' }));
        for (let di = 0; di < dates.length; di++) {
          Object.assign(taskRow.getCell(metaCount + 1 + di), dataStyle(undefined));
        }
        if (task.startDate && task.endDate) {
          writeDeviceBar(ws, rowNum, metaCount, task.startDate, task.endDate, dates, task.color, task.taskName, false);
        }
        taskRow.outlineLevel = 2;
        taskRow.hidden = true;
        rowNum += 1;
      }
    }
  }

  ws.properties.outlineLevelRow = 2;
}

// ── Entry point ───────────────────────────────────────────────────────────

/** Builds the workbook without triggering a download — used by exportScheduleToExcel, and directly by tests. */
export function buildScheduleWorkbook(
  envConfig: EnvConfig,
  schedule: ScheduleData,
  dates: string[],
): ExcelJS.Workbook {
  const wb = new ExcelJS.Workbook();

  const workerModel = buildWorkerTimelineModel(envConfig, schedule, dates, dates[0] ?? '');
  buildWorkerSheet(wb, workerModel, dates);

  const deviceModel = buildModuleViewModel(envConfig, schedule, dates);
  buildDeviceSheet(wb, deviceModel, dates);

  return wb;
}

export async function exportScheduleToExcel(
  envConfig: EnvConfig,
  schedule: ScheduleData,
  dates: string[],
  filename = 'Schedule_export.xlsx',
): Promise<void> {
  const wb = buildScheduleWorkbook(envConfig, schedule, dates);
  const buffer = await wb.xlsx.writeBuffer();
  const blob = new Blob([buffer], {
    type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
  });
  saveAs(blob, filename);
}
