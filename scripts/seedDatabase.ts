/**
 * Seed script — generates database.xlsx in public/data/.
 * Run with: npx ts-node --esm scripts/seedDatabase.ts
 *
 * This is a Node.js script, not bundled with Vite.
 * Requires: npm install exceljs (dev dep already in package.json).
 */
import ExcelJS from 'exceljs';
import path from 'path';
import { fileURLToPath } from 'url';
import { MOCK_DATASETS, MOCK_RUN_LOGS, MOCK_COMMENTS } from '../src/data/mockData.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname  = path.dirname(__filename);

async function seed() {
  const wb = new ExcelJS.Workbook();

  // ── Datasets sheet ──────────────────────────────────────────────
  const dsSheet = wb.addWorksheet('Datasets');
  dsSheet.columns = [
    { header: 'id',           key: 'id',           width: 20 },
    { header: 'name',         key: 'name',         width: 30 },
    { header: 'description',  key: 'description',  width: 50 },
    { header: 'createdAt',    key: 'createdAt',    width: 24 },
    { header: 'updatedAt',    key: 'updatedAt',    width: 24 },
    { header: 'runCount',     key: 'runCount',     width: 10 },
    { header: 'latestStatus', key: 'latestStatus', width: 14 },
  ];
  MOCK_DATASETS.forEach(d => dsSheet.addRow(d));

  // ── RunLogs sheet ───────────────────────────────────────────────
  const rlSheet = wb.addWorksheet('RunLogs');
  rlSheet.columns = [
    { header: 'id',          key: 'id',          width: 16 },
    { header: 'datasetId',   key: 'datasetId',   width: 20 },
    { header: 'runNumber',   key: 'runNumber',   width: 10 },
    { header: 'status',      key: 'status',      width: 12 },
    { header: 'label',       key: 'label',       width: 36 },
    { header: 'startedAt',   key: 'startedAt',   width: 24 },
    { header: 'finishedAt',  key: 'finishedAt',  width: 24 },
    { header: 'hardScore',   key: 'hardScore',   width: 10 },
    { header: 'softScore',   key: 'softScore',   width: 10 },
    { header: 'outputPath',  key: 'outputPath',  width: 40 },
  ];
  MOCK_RUN_LOGS.forEach(r => rlSheet.addRow(r));

  // ── Comments sheet ──────────────────────────────────────────────
  const cmtSheet = wb.addWorksheet('Comments');
  cmtSheet.columns = [
    { header: 'id',        key: 'id',        width: 16 },
    { header: 'datasetId', key: 'datasetId', width: 20 },
    { header: 'author',    key: 'author',    width: 20 },
    { header: 'body',      key: 'body',      width: 80 },
    { header: 'createdAt', key: 'createdAt', width: 24 },
  ];
  MOCK_COMMENTS.forEach(c => cmtSheet.addRow(c));

  // ── Write file ──────────────────────────────────────────────────
  const outPath = path.resolve(__dirname, '../public/data/database.xlsx');
  await wb.xlsx.writeFile(outPath);
  console.log(`Wrote ${outPath}`);
}

seed().catch(err => { console.error(err); process.exit(1); });
