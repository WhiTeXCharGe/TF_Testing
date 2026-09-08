Phase 1 — Mock harness + storage abstraction (the "mock test server" — build first)
 web/mock-azure/ scaffold: aca1/, aca2/ runners, mock-blob/ (gitignored), dev:mock script via concurrently
 storageClient interface (putJson/getJson/listPrefix/delete) + fs impl + unit tests
 Stub ACA1 :4000 and ACA2 :4010 with /api/health + a hardcoded session, wired to fs storage; verify all three processes boot together
 Smoke script: create → list → open → socket connect, against stubs
Phase 2 — ACA2 relay cloud refactor
 Add persistence to sessionStore: async load baseline+log from storage on activate; make store operations async; keep in-memory hot path
 Periodic flush (every N actions / T seconds) + flush-on-last-leave; evict session from memory when empty; write status=not-active
 Internal endpoints: POST /internal/sessions/:id/activate, /lock, /unlock, GET /internal/sessions/:id/live; gate with INTERNAL_KEY
 join handler: reject unknown/not-loaded session; when locked force role=view and drop inbound actions
 Cloud build route split (no save-files / network-info / handoff); env-driven config (PORT, STORAGE, ACA1_URL, INTERNAL_KEY, WEB_ORIGIN)
 Real CORS (WEB_ORIGIN + no-Origin) replacing the LAN allowlist for the cloud build
 Adapt collabSocket.test.ts / sessionStore.test.ts for async store; add load/replay/flush + lock-gating tests
 Dockerfile for server/
Phase 3 — ACA1 session API
 GET /api/sessions — list from storage meta.json by prefix, merge live participant counts from ACA2 /internal/.../live
 POST /api/sessions — multipart 2-YAML upload; parse/validate (reuse yamlService schema logic server-side); write baseline.json + meta.json; return { sessionId, ownerToken }
 POST /api/sessions JSON variant for the desktop app (current in-memory schedule)
 POST /api/sessions/:id/open — status check + ACA2 activate orchestration; return { relayUrl, role }
 POST /api/sessions/:id/lock + /unlock — ownerToken check, update meta.json, notify ACA2
 DELETE /api/sessions/:id — ownerToken; delete storage keys + tell ACA2 to evict
 GET /api/sessions/:id/name — keep for compat
 Idle/absolute TTL sweep (skip locked for idle, honor 8 h absolute)
 Dockerfile for ACA1
 Unit tests: status transitions, open orchestration (stubbed ACA2), ownerToken gating, YAML validation rejects
Phase 4 — Full local integration test
 Vitest integration suite over the running mock: create-from-Test_data/*.yaml → list → open (2 clients) → action propagates → last leave → snapshot on disk → re-open replays to latest → lock → action rejected → unlock → delete
 dev:mock boots ACA1 + ACA2 + Vite with VITE_ACA1_URL; README with run steps
Phase 5 — Client: session list + open + YAML upload
 collabService.ts: add listSessions(), createSessionFromYaml(files), openSession(id, role); socket origin from ACA1's relayUrl (drop fetchCollabLink / network-info)
 New SessionListPage (landing): status badges, Open, Create (2-file upload), owner Lock/Unlock
 Rework SessionDialog.tsx — drop start/join tabs + link rows; keep desktop "push current schedule" as Create
 AppContext.tsx: startCollabSession/joinCollabSession → new open flow; store ownerToken
 Locked session: read-only banner + reuse existing viewer-gating; hide edit affordances
 Handle ACA2 cold-start latency in open (spinner / retry)
 Client tests: list render, open flow (mocked fetch), lock gating (extend Cypress viewer specs)
Phase 6 — Security & abuse limits
 express-rate-limit on POST /api/sessions (10/hr/IP); max concurrent sessions (100); YAML size cap
 Action payload cap 1 MB (align ACA2 maxHttpBufferSize)
 INTERNAL_KEY enforced on every ACA2 /internal/*
 ownerToken stored hashed in meta.json
 Secrets via env / Key Vault ref; document the set
Phase 7 — Azure deploy (when reachable) + fallback
 az scripts: RG, ACA env, ACR, acr build both images, Blob account/container, Key Vault secret
 Deploy ACA1 (minReplicas:1) + ACA2 (minReplicas:0, maxReplicas:N, --affinity sticky), env vars, internal ingress for ACA2 /internal
 Static Web Apps for the web client; custom domain + managed cert (ACA1 + web origin)
 GitHub Actions deploy workflow
 Documented switch: STORAGE=blob + Azure URLs vs STORAGE=fs + localhost mock
 Smoke test against deployed stack (or note it's blocked by your VPN and hand off)
