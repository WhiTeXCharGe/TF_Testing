import * as yamlNs from 'js-yaml';
import type { SessionBaseline } from '../collab/types.js';

// js-yaml is CJS; under NodeNext ESM the default export lands on `.default`
// in some resolvers and on the namespace itself in others.
const yaml: typeof import('js-yaml') =
  (yamlNs as unknown as { default?: typeof import('js-yaml') }).default ?? yamlNs;

// Turns a create-session request into a validated SessionBaseline. Two shapes:
//   - browser: two YAML strings (from an uploaded Schedule.yaml + EnvConfig.yaml)
//   - desktop: schedule/envConfig already parsed as JSON
// Deep schema validation of the schedule/envConfig is the client reducer's job;
// here we only guarantee the baseline is structurally well-formed.

export class IntakeError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'IntakeError';
  }
}

export interface IntakeInput {
  name?: unknown;
  scheduleYaml?: unknown;
  envConfigYaml?: unknown;
  schedule?: unknown;
  envConfig?: unknown;
  currentView?: unknown;
}

export interface IntakeResult {
  name: string;
  baseline: SessionBaseline;
}

const isPlainObject = (v: unknown): v is Record<string, unknown> =>
  typeof v === 'object' && v !== null && !Array.isArray(v);

function parseYaml(label: string, raw: unknown): unknown {
  if (typeof raw !== 'string') throw new IntakeError(`${label} must be a YAML string`);
  try {
    return yaml.load(raw);
  } catch (err) {
    throw new IntakeError(`${label} is not valid YAML: ${(err as Error).message}`);
  }
}

export function intake(input: IntakeInput): IntakeResult {
  const name = typeof input.name === 'string' ? input.name.trim() : '';
  if (name.length < 1 || name.length > 120) {
    throw new IntakeError('name is required and must be 1-120 characters');
  }

  const hasYamlPair = input.scheduleYaml !== undefined || input.envConfigYaml !== undefined;
  const hasJsonPair = input.schedule !== undefined || input.envConfig !== undefined;

  let schedule: unknown;
  let envConfig: unknown;

  if (hasYamlPair) {
    if (input.scheduleYaml === undefined || input.envConfigYaml === undefined) {
      throw new IntakeError('both scheduleYaml and envConfigYaml are required');
    }
    schedule = parseYaml('scheduleYaml', input.scheduleYaml);
    envConfig = parseYaml('envConfigYaml', input.envConfigYaml);
  } else if (hasJsonPair) {
    if (input.schedule === undefined || input.envConfig === undefined) {
      throw new IntakeError('both schedule and envConfig are required');
    }
    schedule = input.schedule;
    envConfig = input.envConfig;
  } else {
    throw new IntakeError('supply either (scheduleYaml + envConfigYaml) or (schedule + envConfig)');
  }

  if (!isPlainObject(schedule)) throw new IntakeError('schedule must be an object');
  if (!isPlainObject(envConfig)) throw new IntakeError('envConfig must be an object');

  const currentView = input.currentView ?? 'worker';
  if (currentView !== 'worker' && currentView !== 'device') {
    throw new IntakeError("currentView must be 'worker' or 'device'");
  }

  return { name, baseline: { schedule, envConfig, currentView } };
}
