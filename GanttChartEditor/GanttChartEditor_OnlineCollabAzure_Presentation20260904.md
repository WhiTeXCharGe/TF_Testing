# GanttChartEditor — Online Collaboration on Azure

### Team meeting · 2026-09-04

Goal: **collaborators no longer need the same network** · keep **cost near zero** for
bursty, meeting-time use · auth stays **anonymous share-link**.

Full engineering detail: `GanttChartEditor_OnlineCollabAzure_Design20260904.md`

---

## 1. Today — LAN only

```mermaid
flowchart LR
  subgraph creator["Creator PC"]
    E["Electron app<br/>(React frontend)"]
    S["Node relay<br/>Express + Socket.IO :3010<br/>in-memory sessions"]
    E --- S
  end
  J["Joiner<br/>http://192.168.x.x:5173"]
  J -- "must be on the same LAN" --> S

  style J stroke-dasharray: 4 4
```

| Works | Doesn't |
|---|---|
| Event-sourced model: baseline + ordered action log, client reducer is source of truth | Joiner must be on the **same LAN** (`192.168.x.x` link) |
| Clients already replay the log on every reconnect | Sessions live in **one process's memory** — no restart survival |
| Roles: `edit` / `view` | **No authentication** — `join` only checks the session exists |
| | CORS locked to LAN; share link is a LAN IP |

---

## 2. Target — hosted relay on Azure

```mermaid
flowchart LR
  D["Desktop app<br/>(RELAY_URL set)"]
  W["Browser join<br/>Azure Static Web Apps (Free)"]
  subgraph azure["Azure"]
    R["Relay<br/>compute option A / B / C / D"]
    ST["Session store<br/>Blob / Cosmos / Redis"]
    KV["Key Vault<br/>join-token secret"]
    R --- ST
    R --- KV
  end
  D -- "wss + REST + signed token" --> R
  W -- "wss + REST + signed token" --> R
```

**Unchanged:** event-sourced model, `reducer.ts`, roles.
**Added:** recoverable session store · signed anonymous join token · real CORS ·
desktop-only routes removed from the public build · TLS + custom domain · abuse limits.

---

## 3. Four compute options

```mermaid
flowchart TD
  Q{"Who operates<br/>connection scaling?"}
  Q -- "We do (self-host Socket.IO)" --> SIO
  Q -- "Azure does (Web PubSub)" --> WPS
  SIO --> A["A · Container Apps<br/>run existing container"]
  SIO --> B["B · App Service<br/>classic Node PaaS"]
  WPS --> C["C · Web PubSub + small API"]
  WPS --> D["D · Functions + Web PubSub<br/>fully serverless"]
```

| | A · Container Apps | B · App Service | C · Web PubSub + API | D · Functions + Web PubSub |
|---|---|---|---|---|
| **Client change** | none | none | small (C1) / rewrite (C2) | rewrite `collabService` |
| **Server change** | none (+Dockerfile) | none | swap 1 transport line | rewrite as Functions |
| **Idle cost** | ~1 small replica | **full plan 24/7** | Free tier **$0** / $50 unit | **storage only** + WPS |
| **Scales connections** | you (Redis + affinity) | you (Redis + affinity) | **Azure** | **Azure** |
| **Cold starts** | none (min 1) | none (Always On) | small API only | HTTP Functions |
| **Ops burden** | low–medium | low–medium | medium | medium–high |
| **Lock-in** | low | low–medium | medium / high | high |
| **Big sync payload** | fine (20 MB) | fine (20 MB) | **1 MB cap → Blob+SAS** | **1 MB cap → Blob+SAS** |
| **Local dev** | `node` | `node` | `+ awps-tunnel` | `+ Core Tools + awps-tunnel` |
| **Best when** | least rework, clear scale path | App Service is the team standard | don't want to run realtime infra | near-zero idle cost is mandatory |

**Rejected:** AKS (too much ops) · bare VM (re-own everything) · Azure SignalR (.NET-first
sibling — analysis transfers if preferred).

---

## 4. Protocol fork — the decision to make in the room

```mermaid
flowchart LR
  subgraph sio["Self-hosted Socket.IO  (A / B)"]
    s1["Client: unchanged"]
    s2["Server holds connections"]
    s3["Scale-out: you add Redis adapter + sticky sessions"]
    s4["No message-size limit beyond yours (20 MB)"]
    s5["Lock-in: low (portable container)"]
  end
  subgraph wps["Azure Web PubSub  (C / D)"]
    w1["Client: small change (C1) or rewrite (C2)"]
    w2["Azure holds connections"]
    w3["Scale-out: built in, no Redis, no affinity"]
    w4["1 MB message cap → big baseline via Blob + SAS"]
    w5["Lock-in: medium–high (Azure-specific)"]
  end
```

| | Socket.IO (self-host) | Azure Web PubSub |
|---|---|---|
| Client change | none | small (C1) / rewrite (C2) |
| Connection scaling & fan-out | **you** (Redis adapter + sticky) | **Azure** (built in) |
| Idle cost floor | ~1 replica always on | Free tier $0, else ~$50 / 1,000-conn unit |
| Message size | 20 MB (as configured now) | **1 MB hard cap** |
| Reconnect + backfill | your `seq` log (already built) | your `seq` log (same) |
| Local dev | run `node` | run `node` + `awps-tunnel` |
| Maturity in our stack | **in production on LAN today** | "for Socket.IO" shim is newer |

---

## 5. Session store (every option needs one)

| Store | Cost | Use for |
|---|---|---|
| **Blob Storage** | cents/mo | baseline snapshot + log flush — small scale, 1 replica |
| **Cosmos DB serverless** | ~$1–10/mo bursty | durable action log at medium scale (pairs with Functions) |
| **Azure Managed Redis** | ~$16–60/mo, always-on | Socket.IO multi-replica backplane — only once you scale past 1 replica |

Progression: **Blob only → add Redis when Socket.IO goes multi-replica → Cosmos if Web PubSub path.**

---

## 6. Sequence — create session + join (Socket.IO path)

```mermaid
sequenceDiagram
  autonumber
  participant C as Creator (desktop)
  participant R as Relay (Azure)
  participant BL as Store (Blob)
  participant P as Participant (browser, anywhere)

  C->>R: POST /collab/sessions {schedule, envConfig, view}
  R->>BL: write baseline
  R-->>C: { sessionId, editToken, viewToken }
  C->>C: build link  ...&session=<id>&role=edit&t=<token>

  C->>R: ws join {sessionId, role:edit, token}
  R->>R: verify token (sig, exp, role)
  R-->>C: sync-init { baseline, actions:[] }

  P->>R: open link → ws join {sessionId, role, token}
  R->>R: verify token
  R->>BL: load baseline + log (if not in memory)
  R-->>P: sync-init { baseline, actions:[...] }
  P->>P: reducer replays baseline + log
  R-->>C: presence [C, P]
  R-->>P: presence [C, P]
```

---

## 7. Sequence — live edit + reconnect

```mermaid
sequenceDiagram
  autonumber
  participant P as Participant (edit)
  participant R as Relay
  participant BL as Store
  participant O as Other participants

  P->>R: action {type, payload}
  R->>R: role == edit? append + stamp seq
  R-->>O: action {type, payload}
  R--)BL: flush log (async, batched)
  O->>O: reducer applies in seq order

  Note over P,R: later — network blip / relay redeploy
  P-xR: socket drops
  P->>R: auto-reconnect → join {token}
  R->>BL: reload baseline + full log if needed
  R-->>P: sync-init { baseline, full log }
  P->>P: reducer rebuilds (idempotent by seq)
```

---

## 8. Sequence — Web PubSub variant (C / D)

```mermaid
sequenceDiagram
  autonumber
  participant P as Participant
  participant N as API /negotiate
  participant W as Azure Web PubSub
  participant S as Server logic
  participant BL as Store

  P->>N: GET /negotiate?session=<id>&t=<token>
  N->>N: verify join token
  N-->>P: { endpoint, accessToken }
  P->>W: ws connect (Azure holds the connection)
  W->>S: "connected" webhook
  S->>BL: load baseline + log
  S->>W: sendToConnection(sync-init)  %% SAS URL if > 1 MB
  W-->>P: sync-init

  P->>W: "action" {type, payload}
  W->>S: "action" webhook
  S->>S: verify role, append seq
  S->>W: sendToGroup(session, action)
  W-->>P: action  (Azure fans out to the whole group)
```

---

## 9. Anonymous link — hardened

```mermaid
flowchart LR
  CR["Create session"] --> TK["Mint signed token<br/>HMAC( sid + role + exp )"]
  TK --> LK["Share link<br/>...?session=sid&role=edit&t=token"]
  LK --> JN["join / negotiate"]
  JN --> VF{"verify<br/>sig · exp · role"}
  VF -- ok --> IN["allowed into session"]
  VF -- fail --> RJ["rejected"]
```

| Control | Value (start) |
|---|---|
| Session-create rate limit | 10 / hour / IP |
| Max participants / session | 25 |
| Action payload cap | 1 MB (matches Web PubSub; big baseline → Blob) |
| Session TTL | 30 min idle (today) + 8 h absolute |
| CORS | web origin only + allow no-Origin (desktop) |
| Public routes | `/api/collab/*` + `/api/health` only — `save-files` etc. removed |
| Phase-2 hook | swap anonymous for Entra ID at the **same** token check |

---

## 10. Cost — approximate, single region, JSON traffic

| Path | Small (internal, ≤25 online) | Medium (100–500 online) |
|---|---|---|
| **A · Container Apps** + Blob + Static Web Apps | **~$5–20 / mo** | ~$50–100 / mo (+ Managed Redis) |
| **B · App Service B1** + Blob | ~$15 / mo (always on) | ~$110+ / mo (S1 + Redis) |
| **C · Web PubSub** + small API | ~$1 / mo (Free tier) · ~$55 (Standard) | ~$65 / mo |
| **D · Functions + Web PubSub** | ~$5 / mo (Free tier) · ~$55–70 (Standard) | ~$65 / mo |

Static Web Apps (browser join) = **Free**. Verify on the Azure Pricing Calculator before committing.

---

## 11. Recommendation

```mermaid
flowchart TD
  R1["Start: Option A — Container Apps<br/>min 1 replica · Blob store · no Redis<br/>+ Static Web Apps for browser join"]
  R1 --> R2["No socket-code change · ~$5–20/mo · low lock-in<br/>clear path to scale (add Redis + affinity later)"]
  ALT["If near-zero idle cost is mandatory →<br/>Option D (Functions + Web PubSub), accept a server rewrite"]
  AVOID["Avoid Option B unless App Service is already the team standard"]
```

**Decide in the room:**

1. **Protocol** — self-hosted Socket.IO (less change, we operate scaling) **vs** Azure Web PubSub (more change + 1 MB workaround, Azure operates scaling)
2. Is **~$15/mo always-on** acceptable, or is **scale-to-zero** a hard requirement?
3. Expected **concurrency** at 6 / 12 months → small vs medium sizing
4. **Custom domain** for browser-join origin + API?
5. **Azure subscription / resource group / cost owner**
6. **Region** (Japan East assumed) — any data-residency constraint?
7. Does **browser-join ship in v1**, or desktop-only pointing at the cloud relay first?
