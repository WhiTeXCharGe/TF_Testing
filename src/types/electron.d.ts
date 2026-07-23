export interface ElectronAPI {
  isElectron: true;
  pickOpenFile: () => Promise<{ path: string; content: string } | null>;
  launchGanttEditor: (transferUrl?: string) => Promise<{ ok: boolean; error?: string }>;
}

declare global {
  interface Window {
    electronAPI?: ElectronAPI;
  }
}
