/**
 * Run service — handles solver job submission and status polling.
 * Replace APP_CONFIG.apiBaseUrl with your Cloud Run Jobs endpoint.
 */
import axios from 'axios';
import { APP_CONFIG } from '@/config/appConfig';
import type { RunLog, NewRunForm } from '@/types';
import { getRunLogs } from './databaseService';

export interface SubmitResult {
  runId: string;
  status: string;
}

/** Submit a new solver run to the backend. */
export async function submitRun(form: NewRunForm): Promise<SubmitResult> {
  if (!APP_CONFIG.apiBaseUrl) {
    // Dev mode: simulate acceptance
    return { runId: `run-${Date.now()}`, status: 'accepted' };
  }
  const res = await axios.post<SubmitResult>(`${APP_CONFIG.apiBaseUrl}/runSolver`, form);
  return res.data;
}

/** Poll status of a run by runId. */
export async function pollRunStatus(runId: string): Promise<{ status: string; outputPath?: string }> {
  if (!APP_CONFIG.apiBaseUrl) {
    return { status: 'Executing' };
  }
  const res = await axios.get(`${APP_CONFIG.apiBaseUrl}/status/${runId}`);
  return res.data;
}

/** Fetch run logs for a dataset (delegates to databaseService). */
export async function fetchRunLogs(datasetId: string): Promise<RunLog[]> {
  return getRunLogs(datasetId);
}
