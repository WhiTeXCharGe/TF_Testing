import type { RequestHandler } from 'express';

// Guards ACA2's /internal/* routes: only ACA1 (which shares the key) may call
// them. ACA2's socket endpoint stays public; this is just the control plane.
export function internalAuth(expectedKey: string): RequestHandler {
  return (req, res, next) => {
    if (req.get('x-internal-key') !== expectedKey) {
      res.status(401).json({ ok: false, error: 'unauthorized' });
      return;
    }
    next();
  };
}
