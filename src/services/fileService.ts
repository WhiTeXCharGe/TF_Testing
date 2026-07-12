import { ScheduleData } from '../types/schedule';
import { EnvConfig } from '../types/envConfig';
import { parseScheduleYaml, parseEnvConfigYaml, stringifyScheduleYaml, stringifyEnvConfigYaml } from './yamlService';
import { buildOpTaskColorMap } from '../components/GanttChart/workerViewModel';

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

// Open two YAML files via browser <input type="file"> picker — sequential dialogs
export async function openTwoYamlFiles(): Promise<LoadedFiles> {
  const envFile = await pickFile('EnvConfig.yaml を選択してください');
  const schedFile = await pickFile('Schedule.yaml を選択してください');

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

// Overwrite-save both YAML files to their original imported paths via Express backend
export async function overwriteSaveFiles(
  envConfig: EnvConfig,
  schedule: ScheduleData,
  envPath: string,
  schedulePath: string,
): Promise<void> {
  const envYaml = stringifyEnvConfigYaml(envConfig);
  const scheduleYaml = stringifyScheduleYaml(resolveScheduleColors(schedule));
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
      else reject(new Error('ファイルが選択されませんでした'));
    };
    // Reject if the user closes without selecting (focus returns to window)
    const onFocus = () => {
      window.removeEventListener('focus', onFocus);
      // Small delay so onchange fires first if a file was picked
      setTimeout(() => {
        if (!input.files?.length) reject(new Error('ファイルが選択されませんでした'));
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
    reader.onerror = () => reject(new Error(`ファイル読み込み失敗: ${file.name}`));
    reader.readAsText(file, 'utf-8');
  });
}