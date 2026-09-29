export interface ElectronAPI {
  isElectron: true;
  pickOpenFile: () => Promise<{ path: string; content: string } | null>;
  pickSaveTarget: (defaultName: string) => Promise<string | null>;
  writeTextFile: (path: string, content: string) => Promise<void>;
  launchScheduler: (transferUrl?: string) => Promise<{ ok: boolean; error?: string }>;
  /**
   * Main intercepts the window's close button and fires this instead of
   * closing immediately, so the renderer gets a chance to checkpoint an
   * in-progress collab session before its socket is torn down — see
   * AppContext's before-close handler and electron/main.cts's 'close' listener.
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

declare global {
  interface Window {
    electronAPI?: ElectronAPI;
  }
}