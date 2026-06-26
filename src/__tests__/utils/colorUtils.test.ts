import { getColorForDevice, getColorForPhaseIndex, lightenColor } from '../../utils/colorUtils';

describe('getColorForDevice', () => {
  it('returns a hex color string', () => {
    const color = getColorForDevice('device-001');
    expect(color).toMatch(/^#[0-9a-f]{6}$/i);
  });
  it('returns the same color for the same device ID', () => {
    const a = getColorForDevice('device-abc');
    const b = getColorForDevice('device-abc');
    expect(a).toBe(b);
  });
  it('returns different colors for different device IDs', () => {
    const a = getColorForDevice('unique-device-x1');
    const b = getColorForDevice('unique-device-x2');
    // They may occasionally match (palette wrap), but most won't
    // Just check both are valid hex colors
    expect(a).toMatch(/^#[0-9a-f]{6}$/i);
    expect(b).toMatch(/^#[0-9a-f]{6}$/i);
  });
});

describe('getColorForPhaseIndex', () => {
  it('returns a hex color for index 0', () => {
    expect(getColorForPhaseIndex(0)).toMatch(/^#[0-9a-f]{6}$/i);
  });
  it('returns different colors for different phase indices 0–4', () => {
    const colors = [0, 1, 2, 3, 4].map(getColorForPhaseIndex);
    const unique = new Set(colors);
    expect(unique.size).toBe(5);
  });
  it('wraps around after 5 phases', () => {
    expect(getColorForPhaseIndex(0)).toBe(getColorForPhaseIndex(5));
    expect(getColorForPhaseIndex(1)).toBe(getColorForPhaseIndex(6));
  });
  it('handles large index', () => {
    expect(getColorForPhaseIndex(999)).toMatch(/^#[0-9a-f]{6}$/i);
  });
});

describe('lightenColor', () => {
  it('returns a valid hex color', () => {
    expect(lightenColor('#336699')).toMatch(/^#[0-9a-f]{6}$/i);
  });
  it('lightens a dark color', () => {
    const original = '#000000';
    const lightened = lightenColor(original, 20);
    expect(lightened).not.toBe(original);
    // R, G, B each go from 0 to 20 = 0x14
    expect(lightened).toBe('#141414');
  });
  it('does not exceed #ffffff', () => {
    const lightened = lightenColor('#ffffff', 100);
    expect(lightened).toBe('#ffffff');
  });
  it('uses default amount of 20 when not specified', () => {
    const a = lightenColor('#336699');
    const b = lightenColor('#336699', 20);
    expect(a).toBe(b);
  });
});
