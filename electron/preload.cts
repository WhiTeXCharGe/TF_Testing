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
  /**
   * Main intercepts the window's close button and fires this instead of
   * closing immediately, so the renderer gets a chance to checkpoint an
   * in-progress collab session before its socket is torn down — see
   * AppContext's before-close handler and main.cts's 'close' listener.
   */
  onBeforeClose: (cb: () => void) => void;
  /** Tells main it's safe to actually close the window now. */
  notifyReadyToClose: () => void;
  /**
   * Opens another top-level window, independent of this one — sharing the
   * one embedded server process, but its own renderer/AppContext, so it can
   * show a different local file or a different online session at the same
   * time (comparing two Gantts side by side).
   */
  openNewWindow: () => Promise<void>;
}

const api: ElectronAPI = {
  isElectron: true,
  pickOpenFile: () => ipcRenderer.invoke('dialog:pickOpenFile'),
  pickSaveTarget: (defaultName: string) => ipcRenderer.invoke('dialog:pickSaveTarget', defaultName),
  writeTextFile: (path: string, content: string) => ipcRenderer.invoke('fs:writeTextFile', path, content),
  launchScheduler: (transferUrl?: string) => ipcRenderer.invoke('sibling:launchScheduler', transferUrl),
  onBeforeClose: (cb: () => void) => ipcRenderer.on('app:before-close', () => cb()),
  notifyReadyToClose: () => ipcRenderer.send('app:ready-to-close'),
  openNewWindow: () => ipcRenderer.invoke('window:new'),
};

contextBridge.exposeInMainWorld('electronAPI', api);