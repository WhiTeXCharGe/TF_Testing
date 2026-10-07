import { UI } from '../config/uiText';

// Kept apart from collabService so UI code can `instanceof`-check them without
// pulling in the whole socket/session service (and so tests that mock
// collabService still see the real classes).

/** Can't reach the configured Azure server (config.txt mode=online). Never falls back to local. */
export class ServerUnreachableError extends Error {
  constructor(public readonly url: string) {
    super(UI.serverUnreachableMessage(url));
    this.name = 'ServerUnreachableError';
  }
}

/** config.txt exists but is invalid (e.g. a mode typo, or a missing azure_url). */
export class ConfigError extends Error {
  constructor(detail: string) {
    super(UI.configErrorMessage(detail));
    this.name = 'ConfigError';
  }
}

/** Connection problems that mean "the server isn't there" rather than "the list is empty". */
export function isConnectionError(err: unknown): err is ServerUnreachableError | ConfigError {
  return err instanceof ServerUnreachableError || err instanceof ConfigError;
}
