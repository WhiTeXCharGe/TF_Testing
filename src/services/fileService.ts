import { ScheduleData } from '../types/schedule';
import { EnvConfig } from '../types/envConfig';
import { parseScheduleYaml, parseEnvConfigYaml, stringifyScheduleYaml, stringifyEnvConfigYaml } from './yamlService';
import { buildOpTaskColorMap } from '../components/GanttChart/workerViewModel';
import { UI } from '../config/uiText';

// Bake resolved (auto-generated or explicit) colors into schedule objects before saving
export function resolveScheduleColors(schedule: ScheduleData): ScheduleData {
  const colorMap = buildOpTaskColorMap(schedule);
  return {
    ...schedule,
    workflowTaskList: schedule.workflowTaskList.map(wt => {
      if (wt.phaseTaskList.length === 0) {
        // misc task — color keyed by workflowTask.id
        const hex = colorMap.get(wt.id);
        const code = hex ? hex.replace(/^#/, '') : wt.colorCode;
        return { ...wt, colorCode: code };
      }
      return {
        ...wt,
        phaseTaskList: wt.phaseTaskList.map(pt => ({
          ...pt,
          operationTaskList: pt.operationTaskList.map(ot => {
            const hex = colorMap.get(ot.id);
            const code = hex ? hex.replace(/^#/, '') : ot.colorCode;
            return { ...ot, colorCode: code };
          }),
        })),
      };
    }),
  };
}

export interface LoadedFiles {
  envConfig: EnvConfig;
  schedule: ScheduleData;
  envFileName: string;
  scheduleFileName: string;
}

// Open two YAML files. In the desktop app this uses native OS Open dialogs
// (real absolute paths, so 上書き保存 can later write back to the right file).
// In the browser it falls back to <input type="file"> (only exposes a bare
// filename — overwrite-save can't target a real path in that mode).
export async function openTwoYamlFiles(): Promise<LoadedFiles> {
  if (window.electronAPI) return openTwoYamlFilesElectron();

  const envFile = await pickFile(UI.pickEnvFilePrompt);
  const schedFile = await pickFile(UI.pickScheduleFilePrompt);

  const [envText, schedText] = await Promise.all([
    readFileText(envFile),
    readFileText(schedFile),
  ]);

  return {
    envConfig: parseEnvConfigYaml(envText),
    schedule: parseScheduleYaml(schedText),
    envFileName: envFile.name,
    scheduleFileName: schedFile.name,
  };
}

async function openTwoYamlFilesElectron(): Promise<LoadedFiles> {
  const api = window.electronAPI!;
  const envPick = await api.pickOpenFile();
  if (!envPick) throw new Error(UI.noFileSelectedError);
  const schedPick = await api.pickOpenFile();
  if (!schedPick) throw new Error(UI.noFileSelectedError);

  return {
    envConfig: parseEnvConfigYaml(envPick.content),
    schedule: parseScheduleYaml(schedPick.content),
    envFileName: envPick.path,
    scheduleFileName: schedPick.path,
  };
}

/** Open a single YAML file via native dialog. Used by FileOpenDialog's per-field pickers. */
export async function openSingleYamlElectron(): Promise<{ path: string; content: string } | null> {
  return window.electronAPI!.pickOpenFile();
}

// Trigger browser download of the current schedule as YAML
export function downloadScheduleYaml(data: ScheduleData, filename = 'Schedule.yaml'): void {
  const text = stringifyScheduleYaml(resolveScheduleColors(data));
  const blob = new Blob([text], { type: 'text/yaml' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

// Trigger browser download of EnvConfig as YAML
export function downloadEnvConfigYaml(data: EnvConfig, filename = 'EnvConfig.yaml'): void {
  const text = stringifyEnvConfigYaml(data);
  const blob = new Blob([text], { type: 'text/yaml' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

// Overwrite-save both YAML files to their original imported paths.
// Desktop app: writes directly via Electron's main process (real absolute paths).
// Browser: POSTs to the local Express backend, which resolves the path itself —
// this only worked reliably when envPath/schedulePath were real paths to begin
// with, which the browser file picker can never provide (see openTwoYamlFiles).
export async function overwriteSaveFiles(
  envConfig: EnvConfig,
  schedule: ScheduleData,
  envPath: string,
  schedulePath: string,
): Promise<void> {
  const envYaml = stringifyEnvConfigYaml(envConfig);
  const scheduleYaml = stringifyScheduleYaml(resolveScheduleColors(schedule));

  if (window.electronAPI) {
    await window.electronAPI.writeTextFile(envPath, envYaml);
    await window.electronAPI.writeTextFile(schedulePath, scheduleYaml);
    return;
  }

  const res = await fetch('/api/save-files', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ envPath, schedulePath, envYaml, scheduleYaml }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: 'Unknown error' }));
    throw new Error(err.error ?? 'Save failed');
  }
}

// "Save As" for the desktop app: native Save dialogs for both files, writes
// directly, and returns the newly chosen absolute paths so the caller can
// remember them for the next 上書き保存. Returns null if either dialog was cancelled.
export async function saveYamlFilesAsElectron(
  envConfig: EnvConfig,
  schedule: ScheduleData,
  defaultEnvName: string,
  defaultScheduleName: string,
): Promise<{ envPath: string; schedulePath: string } | null> {
  const api = window.electronAPI!;
  const envPath = await api.pickSaveTarget(defaultEnvName);
  if (!envPath) return null;
  const schedulePath = await api.pickSaveTarget(defaultScheduleName);
  if (!schedulePath) return null;

  await api.writeTextFile(envPath, stringifyEnvConfigYaml(envConfig));
  await api.writeTextFile(schedulePath, stringifyScheduleYaml(resolveScheduleColors(schedule)));
  return { envPath, schedulePath };
}

// Download both YAML files as browser downloads with given filenames
export function downloadBothYamlFiles(
  envConfig: EnvConfig,
  schedule: ScheduleData,
  envFilename: string,
  scheduleFilename: string,
): void {
  downloadEnvConfigYaml(envConfig, envFilename);
  setTimeout(() => downloadScheduleYaml(schedule, scheduleFilename), 200);
}

// Open a single file picker and return the selected File
function pickFile(title: string): Promise<File> {
  return new Promise((resolve, reject) => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.yaml,.yml';
    input.title = title;
    input.onchange = () => {
      const file = input.files?.[0];
      if (file) resolve(file);
      else reject(new Error(UI.noFileSelectedError));
    };
    // Reject if the user closes without selecting (focus returns to window)
    const onFocus = () => {
      window.removeEventListener('focus', onFocus);
      // Small delay so onchange fires first if a file was picked
      setTimeout(() => {
        if (!input.files?.length) reject(new Error(UI.noFileSelectedError));
      }, 500);
    };
    window.addEventListener('focus', onFocus);
    input.click();
  });
}

function readFileText(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = () => reject(new Error(UI.fileReadFailedMessage(file.name)));
    reader.readAsText(file, 'utf-8');
  });
}