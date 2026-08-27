// Which browser Origins the local/LAN collab feature is allowed to serve.
//
// This gate matters more than it looks: POST /api/save-files takes an absolute
// path straight from the request body and writes it. Because express.json()
// requires Content-Type: application/json — not a CORS-simple type — that
// endpoint is always preflighted, so whether the preflight succeeds is the
// only thing standing between a drive-by website and overwriting any file the
// OS user can write while the app happens to be running.
//
// Deliberately NOT a prefix regex over the raw origin string. Something like
// /^https?:\/\/(localhost|10\.|192\.168\.)/ matches https://localhost.evil.com,
// https://10.evil.com and friends — hostnames may begin with digits and those
// are all registrable domains an attacker can point anywhere, which would
// leave the hole wide open. The host is parsed out and matched exactly.
export function isLocalOrLanOrigin(origin: string): boolean {
  let url: URL;
  try {
    url = new URL(origin);
  } catch {
    return false; // opaque/malformed origin (e.g. "null" from a sandboxed iframe)
  }
  if (url.protocol !== 'http:' && url.protocol !== 'https:') return false;

  // URL keeps IPv6 literals bracketed in .hostname
  const host = url.hostname.replace(/^\[(.*)\]$/, '$1').toLowerCase();

  if (host === 'localhost' || host === '::1') return true;

  const m = /^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$/.exec(host);
  if (!m) return false;
  const octets = m.slice(1, 5).map(Number);
  if (octets.some(o => o > 255)) return false;

  const [a, b] = octets;
  return (
    a === 127 ||                          // loopback
    a === 10 ||                           // RFC1918
    (a === 192 && b === 168) ||           // RFC1918
    (a === 172 && b >= 16 && b <= 31) ||  // RFC1918
    (a === 169 && b === 254)              // link-local — two PCs on a direct cable, no DHCP
  );
}
