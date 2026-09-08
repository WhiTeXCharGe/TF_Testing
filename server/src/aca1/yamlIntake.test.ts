import { describe, it, expect } from 'vitest';
import { intake, IntakeError } from './yamlIntake.js';

describe('intake', () => {
  it('accepts a pre-parsed JSON payload (desktop path)', () => {
    const r = intake({ name: 'Plan', schedule: { a: 1 }, envConfig: { b: 2 }, currentView: 'device' });
    expect(r).toEqual({ name: 'Plan', baseline: { schedule: { a: 1 }, envConfig: { b: 2 }, currentView: 'device' } });
  });

  it('parses YAML strings (browser path) and defaults currentView to worker', () => {
    const r = intake({ name: 'Plan', scheduleYaml: 'a: 1\n', envConfigYaml: 'b: 2\n' });
    expect(r.baseline).toEqual({ schedule: { a: 1 }, envConfig: { b: 2 }, currentView: 'worker' });
  });

  it('trims the name', () => {
    expect(intake({ name: '  Plan  ', schedule: {}, envConfig: {} }).name).toBe('Plan');
  });

  it('rejects a missing / blank name', () => {
    expect(() => intake({ name: '  ', schedule: {}, envConfig: {} })).toThrow(IntakeError);
  });

  it('rejects a name over 120 chars', () => {
    expect(() => intake({ name: 'x'.repeat(121), schedule: {}, envConfig: {} })).toThrow(IntakeError);
  });

  it('rejects when neither YAML nor JSON pair is supplied', () => {
    expect(() => intake({ name: 'x' })).toThrow(IntakeError);
  });

  it('rejects a half-supplied pair', () => {
    expect(() => intake({ name: 'x', schedule: {} })).toThrow(IntakeError);
    expect(() => intake({ name: 'x', scheduleYaml: 'a: 1' })).toThrow(IntakeError);
  });

  it('rejects malformed YAML', () => {
    expect(() => intake({ name: 'x', scheduleYaml: ':\n:\n  -', envConfigYaml: 'b: 2' })).toThrow(IntakeError);
  });

  it('rejects a non-object schedule or envConfig', () => {
    expect(() => intake({ name: 'x', schedule: 5, envConfig: {} })).toThrow(IntakeError);
    expect(() => intake({ name: 'x', schedule: {}, envConfig: null })).toThrow(IntakeError);
    expect(() => intake({ name: 'x', scheduleYaml: '- 1\n- 2', envConfigYaml: 'b: 2' })).toThrow(IntakeError);
  });

  it('rejects an invalid currentView', () => {
    expect(() => intake({ name: 'x', schedule: {}, envConfig: {}, currentView: 'nope' })).toThrow(IntakeError);
  });
});
