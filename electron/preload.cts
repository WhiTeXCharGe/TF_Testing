// Preload script — runs in an isolated context, exposes a narrow safe surface
// to the renderer via contextBridge. CommonJS (.cts -> .cjs) so it loads
// correctly regardless of the app's "type": "module" setting.
import { contextBridge, ipcRenderer } from 'electron';

export interface ElectronAPI {
  isElectron: true;
  /** Native "Open" dialog for a single YAML file (New Run input pickers). */
  pickOpenFile: () => Promise<{ path: string; content: string } | null>;
  /**
   * Ensure GanttChartEditor is reachable, launching its installed .exe if needed.
   * Pass transferUrl to deliver a cross-app handoff — GanttChartEditor navigates
   * its one window straight to that URL instead of opening a second window.
   */
  launchGanttEditor: (transferUrl?: string) => Promise<{ ok: boolean; error?: string }>;
}

const api: ElectronAPI = {
  isElectron: true,
  pickOpenFile: () => ipcRenderer.invoke('dialog:pickOpenFile'),
  launchGanttEditor: (transferUrl?: string) => ipcRenderer.invoke('sibling:launchGanttEditor', transferUrl),
};

contextBridge.exposeInMainWorld('electronAPI', api);
