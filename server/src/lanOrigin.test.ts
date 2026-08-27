import { describe, it, expect } from 'vitest';
import { isLocalOrLanOrigin } from './lanOrigin.js';

describe('isLocalOrLanOrigin', () => {
  it('allows the origins the collab feature actually needs', () => {
    // The host's own dev server / packaged app
    expect(isLocalOrLanOrigin('http://localhost:5173')).toBe(true);
    expect(isLocalOrLanOrigin('http://localhost:5174')).toBe(true); // sibling SchedulerWeb
    expect(isLocalOrLanOrigin('http://127.0.0.1:5173')).toBe(true);
    expect(isLocalOrLanOrigin('http://[::1]:5173')).toBe(true);
    // A LAN joiner who opened the shared link, across all three RFC1918 ranges
    expect(isLocalOrLanOrigin('http://192.168.1.23:5173')).toBe(true);
    expect(isLocalOrLanOrigin('http://10.0.0.5:5173')).toBe(true);
    expect(isLocalOrLanOrigin('http://172.16.4.4:5173')).toBe(true);
    expect(isLocalOrLanOrigin('http://172.31.255.254:5173')).toBe(true);
    // Two PCs on a direct cable with no DHCP
    expect(isLocalOrLanOrigin('http://169.254.10.20:5173')).toBe(true);
  });

  it('rejects public and non-private addresses', () => {
    expect(isLocalOrLanOrigin('https://attacker.example')).toBe(false);
    expect(isLocalOrLanOrigin('http://8.8.8.8')).toBe(false);
    expect(isLocalOrLanOrigin('http://172.32.0.1')).toBe(false); // just past the RFC1918 block
    expect(isLocalOrLanOrigin('http://172.15.0.1')).toBe(false); // just before it
    expect(isLocalOrLanOrigin('http://11.0.0.1')).toBe(false);
    expect(isLocalOrLanOrigin('http://192.169.0.1')).toBe(false);
  });

  // The reason this is a parsed host match and not a prefix regex over the raw
  // origin string: every one of these is a registrable domain (hostnames may
  // start with a digit) that a naive /^https?:\/\/(localhost|10\.|…)/ lets
  // through — which would leave POST /api/save-files reachable from any site.
  it('rejects lookalike domains that merely start with an allowed prefix', () => {
    expect(isLocalOrLanOrigin('https://localhost.evil.example')).toBe(false);
    expect(isLocalOrLanOrigin('https://127.0.0.1.evil.example')).toBe(false);
    expect(isLocalOrLanOrigin('https://10.evil.example')).toBe(false);
    expect(isLocalOrLanOrigin('https://192.168.evil.example')).toBe(false);
    expect(isLocalOrLanOrigin('https://172.16.evil.example')).toBe(false);
    expect(isLocalOrLanOrigin('https://notlocalhost')).toBe(false);
  });

  it('rejects malformed, opaque and non-http origins', () => {
    expect(isLocalOrLanOrigin('null')).toBe(false); // sandboxed iframe
    expect(isLocalOrLanOrigin('')).toBe(false);
    expect(isLocalOrLanOrigin('not a url')).toBe(false);
    expect(isLocalOrLanOrigin('file://localhost')).toBe(false);
    expect(isLocalOrLanOrigin('ftp://192.168.1.1')).toBe(false);
    expect(isLocalOrLanOrigin('javascript:alert(1)')).toBe(false);
    // Out-of-range octets must not sneak through the shape check
    expect(isLocalOrLanOrigin('http://192.168.1.999')).toBe(false);
    expect(isLocalOrLanOrigin('http://999.168.1.1')).toBe(false);
  });
});
