// ─────────────────────────────────────────────────────────────────────────────
// UI language selector.
//
// All UI-visible text lives in language-specific files in this folder.
// To switch the entire UI between languages, change exactly ONE line below:
//
//   export { UI } from './uiConfig.ja';   ← Japanese (default)
//   export { UI } from './uiConfig.en';   ← English
//
// Both files export an object with identical shape, so no other code needs
// to change. Keys missing from one language would cause a TypeScript error.
// ─────────────────────────────────────────────────────────────────────────────

export { UI } from './uiConfig.ja';
