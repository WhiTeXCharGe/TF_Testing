/** @type {import('jest').Config} */
module.exports = {
  testEnvironment: 'node',
  transform: {
    '^.+\\.tsx?$': ['ts-jest', { tsconfig: '<rootDir>/tsconfig.jest.json', diagnostics: false }],
  },
  moduleNameMapper: {
    '\\.(css|less|scss|sass|png|jpg|svg)$': '<rootDir>/src/__mocks__/fileMock.cjs',
  },
  setupFilesAfterEnv: ['<rootDir>/src/setupTests.ts'],
  testMatch: [
    '<rootDir>/src/__tests__/**/*.test.ts',
    '<rootDir>/src/__tests__/**/*.test.tsx',
  ],
  testPathIgnorePatterns: [
    '<rootDir>/src/__tests__/gantt/moduleViewModel.test.ts',
  ],
  collectCoverageFrom: [
    'src/utils/**/*.ts',
    'src/services/**/*.ts',
    'src/context/reducer.ts',
    'src/components/GanttChart/workerViewModel.ts',
    'src/components/GanttChart/moduleViewModel.ts',
  ],
  coverageReporters: ['text', 'lcov'],
};
