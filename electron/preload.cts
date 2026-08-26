// Preload script — runs in an isolated context with access to a subset of
// Node APIs, and exposes a narrow, safe surface to the renderer via
// contextBridge. Written as CommonJS (.cts -> .cjs) so it loads correctly
// regardless of the app's "type": "module" setting.
import { contextBridge, ipcRenderer } from 'electron';

export interface ElectronAPI {
  isElectron: true;
  /** Native "Open" dialog for a single YAML file. Returns null if cancelled. */
  pickOpenFile: () => Promise<{ path: string; content: string } | null>;
  /** Native "Save As" dialog. Returns the chosen absolute path, or null if cancelled. */
  pickSaveTarget: (defaultName: string) => Promise<string | null>;
  /** Write text content directly to an absolute path. */
  writeTextFile: (path: string, content: string) => Promise<void>;
  /**
   * Ensure SchedulerWeb is reachable, launching its installed .exe if needed.
   * Pass transferUrl to deliver a cross-app handoff — SchedulerWeb navigates
   * its one window straight to that URL instead of opening a second window.
   */
  launchScheduler: (transferUrl?: string) => Promise<{ ok: boolean; error?: string }>;
  /** Tells the main process whether a collab session is active, so it can
   *  keep the embedded server running (hidden to tray) instead of quitting
   *  when the window closes. */
  setCollabSessionActive: (active: boolean) => void;
}

const api: ElectronAPI = {
  isElectron: true,
  pickOpenFile: () => ipcRenderer.invoke('dialog:pickOpenFile'),
  pickSaveTarget: (defaultName: string) => ipcRenderer.invoke('dialog:pickSaveTarget', defaultName),
  writeTextFile: (path: string, content: string) => ipcRenderer.invoke('fs:writeTextFile', path, content),
  launchScheduler: (transferUrl?: string) => ipcRenderer.invoke('sibling:launchScheduler', transferUrl),
  setCollabSessionActive: (active: boolean) => ipcRenderer.send('collab:session-active-changed', active),
};

contextBridge.exposeInMainWorld('electronAPI', api);
