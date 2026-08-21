import { Router } from 'express';
import os from 'node:os';

export const networkInfoRouter = Router();

// Lets the client build a LAN-reachable share link (e.g. http://192.168.x.x:5173/?view=1)
// without the browser being able to discover its own machine's LAN-facing address itself.
networkInfoRouter.get('/network-info', (_req, res) => {
  const interfaces = os.networkInterfaces();
  const addresses: string[] = [];
  for (const ifaceList of Object.values(interfaces)) {
    for (const iface of ifaceList ?? []) {
      if (iface.family === 'IPv4' && !iface.internal) {
        addresses.push(iface.address);
      }
    }
  }
  res.json({ ok: true, addresses });
});
