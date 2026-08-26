import '@testing-library/jest-dom';

// Needed for React 18/19's act() to recognize this as a React test
// environment when a test file opts into jsdom via a per-file
// `@jest-environment jsdom` docblock (the project's default test
// environment is 'node'). Harmless for non-DOM test files.
(globalThis as unknown as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
