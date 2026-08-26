export interface ElectronAPI {
  isElectron: true;
  pickOpenFile: () => Promise<{ path: string; content: string } | null>;
  pickSaveTarget: (defaultName: string) => Promise<string | null>;
  writeTextFile: (path: string, content: string) => Promise<void>;
  launchScheduler: (transferUrl?: string) => Promise<{ ok: boolean; error?: string }>;
  setCollabSessionActive: (active: boolean) => void;
}

declare global {
  interface Window {
    electronAPI?: ElectronAPI;
  }
}
