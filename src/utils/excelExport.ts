/**
 * Export Gantt data to a styled Excel file.
 * Mirrors the output of yaml_to_suother_like_excel.py.
 *
 * Uses ExcelJS for cell styling (colors, borders, frozen panes).
 * Uses file-saver for browser-side download.
 */
import * as ExcelJS from 'exceljs';
import { saveAs } from 'file-saver';
import type { GanttData } from '@/types';
import { dowAbbr, toKey } from './dateUtils';

const FIXED_COLS = 4; // Company | Name | Role | Mgr

/** Export the Gantt chart to an Excel (.xlsx) file and trigger download. */
export async function exportGanttToExcel(
  data: GanttData,
  filename = 'schedule_gantt.xlsx'
): Promise<void> {
  const wb = new ExcelJS.Workbook();
  const ws = wb.addWorksheet('Gantt', {
    views: [{ state: 'frozen', xSplit: FIXED_COLS, ySplit: 3 }],
  });

  const { employees, dates, cells, cutoffDate, todayDate } = data;
  const todayKey = toKey(todayDate);

  // ── Column widths ──────────────────────────────────────────────
  ws.getColumn(1).width = 14; // Company
  ws.getColumn(2).width = 14; // Name
  ws.getColumn(3).width = 8;  // Role
  ws.getColumn(4).width = 5;  // Mgr
  for (let ci = 0; ci < dates.length; ci++) {
    ws.getColumn(FIXED_COLS + 1 + ci).width = 3.5;
  }

  // ── Helper: cell style ─────────────────────────────────────────
  function headerStyle(color = 'FFE8F0FE'): Partial<ExcelJS.Style> {
    return {
      fill: { type: 'pattern', pattern: 'solid', fgColor: { argb: color } },
      font: { bold: true, size: 9, name: 'Calibri' },
      alignment: { horizontal: 'center', vertical: 'middle' },
      border: {
        bottom: { style: 'thin', color: { argb: 'FFDADCE0' } },
        right:  { style: 'thin', color: { argb: 'FFDADCE0' } },
      },
    };
  }

  function dataStyle(fillArgb?: string): Partial<ExcelJS.Style> {
    return {
      fill: fillArgb
        ? { type: 'pattern', pattern: 'solid', fgColor: { argb: fillArgb } }
        : { type: 'pattern', pattern: 'none' },
      font: { size: 9, name: 'Calibri' },
      alignment: { horizontal: 'center', vertical: 'middle' },
      border: {
        bottom: { style: 'hair', color: { argb: 'FFDADCE0' } },
        right:  { style: 'hair', color: { argb: 'FFDADCE0' } },
      },
    };
  }

  // ── Row 1: Month labels (merged) ───────────────────────────────
  const row1 = ws.getRow(1);
  row1.height = 16;

  // Left header labels
  ['Company', 'Name', 'Role', 'Mgr'].forEach((label, i) => {
    const cell = row1.getCell(i + 1);
    cell.value = label;
    Object.assign(cell, headerStyle());
  });

  // Group dates by month for merging
  let prevMonth = -1;
  let mergeStart = FIXED_COLS + 1;
  dates.forEach((d, di) => {
    const colIdx = FIXED_COLS + 1 + di;
    const mo = d.getMonth();
    if (mo !== prevMonth) {
      if (prevMonth !== -1 && colIdx - 1 >= mergeStart) {
        ws.mergeCells(1, mergeStart, 1, colIdx - 1);
      }
      mergeStart = colIdx;
      prevMonth = mo;
    }
    if (di === dates.length - 1 && colIdx >= mergeStart) {
      ws.mergeCells(1, mergeStart, 1, colIdx);
    }
  });

  // Set month label values
  let writtenMonths = new Set<number>();
  dates.forEach((d, di) => {
    const colIdx = FIXED_COLS + 1 + di;
    const mo = d.getMonth();
    if (!writtenMonths.has(mo)) {
      const cell = row1.getCell(colIdx);
      cell.value = `${d.getFullYear()}/${String(mo + 1).padStart(2, '0')}`;
      Object.assign(cell, headerStyle('FF1A73E8'));
      if (cell.font) cell.font.color = { argb: 'FFFFFFFF' };
      writtenMonths.add(mo);
    }
  });

  // ── Row 2: Day numbers ─────────────────────────────────────────
  const row2 = ws.getRow(2);
  row2.height = 14;
  ['Company', 'Name', 'Role', 'Mgr'].forEach((_, i) => {
    Object.assign(row2.getCell(i + 1), headerStyle());
  });
  dates.forEach((d, di) => {
    const cell = row2.getCell(FIXED_COLS + 1 + di);
    cell.value = d.getDate();
    const isWE = d.getDay() === 0 || d.getDay() === 6;
    Object.assign(cell, headerStyle(isWE ? 'FFF1F3F4' : 'FFE8F0FE'));
  });

  // ── Row 3: Weekday abbreviations ───────────────────────────────
  const row3 = ws.getRow(3);
  row3.height = 14;
  ['Company', 'Name', 'Role', 'Mgr'].forEach((_, i) => {
    Object.assign(row3.getCell(i + 1), headerStyle());
  });
  dates.forEach((d, di) => {
    const cell = row3.getCell(FIXED_COLS + 1 + di);
    const dow = d.getDay();
    cell.value = dowAbbr(d);
    const isWE = dow === 0 || dow === 6;
    Object.assign(cell, headerStyle(isWE ? 'FFF1F3F4' : 'FFE8F0FE'));
    if (dow === 0) cell.font = { ...cell.font, color: { argb: 'FFC5221F' } };
    if (dow === 6) cell.font = { ...cell.font, color: { argb: 'FF1A73E8' } };
  });

  // ── Data rows ──────────────────────────────────────────────────
  const cutoffKey = cutoffDate ? toKey(cutoffDate) : null;

  employees.forEach((emp, ri) => {
    const exRow = ws.getRow(4 + ri);
    exRow.height = 16;

    // Company (with company color)
    const compArgb = 'FF' + emp.companyColor.replace('#', '');
    const compCell = exRow.getCell(1);
    compCell.value = emp.company;
    Object.assign(compCell, dataStyle(compArgb));

    // Name
    const nameCell = exRow.getCell(2);
    nameCell.value = emp.name;
    Object.assign(nameCell, dataStyle());
    nameCell.alignment = { horizontal: 'left', vertical: 'middle' };

    // Role
    const roleCell = exRow.getCell(3);
    roleCell.value = emp.role;
    Object.assign(roleCell, dataStyle());

    // Mgr
    const mgrCell = exRow.getCell(4);
    mgrCell.value = emp.isManager ? 'M' : '';
    Object.assign(mgrCell, dataStyle(emp.isManager ? 'FF1A73E8' : undefined));
    if (emp.isManager) mgrCell.font = { ...mgrCell.font, color: { argb: 'FFFFFFFF' }, bold: true };

    // Date cells
    dates.forEach((d, di) => {
      const colIdx = FIXED_COLS + 1 + di;
      const ganttCell = cells[ri]?.[di];
      const exCell = exRow.getCell(colIdx);

      const dk = toKey(d);
      const isWE = d.getDay() === 0 || d.getDay() === 6;

      let fillArgb: string | undefined;
      if (ganttCell?.type === 'unavailable') {
        fillArgb = 'FFF28B82';
      } else if (ganttCell?.type === 'work' && ganttCell.moduleColor) {
        fillArgb = 'FF' + ganttCell.moduleColor.replace('#', '');
      } else if (isWE) {
        fillArgb = 'FFF1F3F4';
      }

      Object.assign(exCell, dataStyle(fillArgb));

      // Cut-off border (right)
      if (cutoffKey && dk === cutoffKey) {
        exCell.border = {
          ...exCell.border,
          right: { style: 'medium', color: { argb: 'FF1A73E8' } },
        };
      }
      // Today border (left)
      if (dk === todayKey) {
        exCell.border = {
          ...exCell.border,
          left: { style: 'medium', color: { argb: 'FFE37400' } },
        };
      }
    });
  });

  // ── Download ───────────────────────────────────────────────────
  const buffer = await wb.xlsx.writeBuffer();
  const blob = new Blob([buffer], {
    type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
  });
  saveAs(blob, filename);
}
