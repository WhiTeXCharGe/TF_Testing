import { Router } from 'express';
import { getLanAddresses } from '../lan/lanAddresses.js';

export const networkInfoRouter = Router();

// Lets the client discover its own machine's LAN-facing address(es) — a
// browser/renderer has no way to do this on its own.
networkInfoRouter.get('/network-info', (_req, res) => {
  res.json({ ok: true, addresses: getLanAddresses() });
});
