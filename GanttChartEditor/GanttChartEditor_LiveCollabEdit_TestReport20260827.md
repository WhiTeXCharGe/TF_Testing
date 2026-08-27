# GanttChartEditor Live Collaborative Editing — Test Report & Bug Log

**Companion to:** `GanttChartEditor_LiveCollabEdit_Design20260826.md` (design) and `GanttChartEditor_LiveCollabEdit_Plan20260826.md` (12-task implementation plan).
**Branch merged:** `live-collab-edit` → `GanttChartEditor`, 20 commits.
**Final state:** client 131/131 tests passing (8 suites), server 20/20 tests passing (3 files).

This document records what was actually built, every automated test added, the one live end-to-end manual verification pass, and every bug found (and fixed) along the way — including the ones caught only after the feature looked "done."

---

## 1. What Was Implemented

| Piece | Where | Purpose |
|---|---|---|
| Session store | `server/src/collab/sessionStore.ts` | Pure in-memory store: a session's baseline data (schedule + envConfig), an ordered action log, and the current participant list. No networking, no disk I/O. |
| Socket.IO relay | `server/src/collab/collabSocket.ts` | Wires the store to real connections: `join` (returns baseline + full action log + participants), `action` (edit-role only, appended to the log and broadcast to everyone else), `leave`/disconnect (updates presence). |
| Session create endpoint | `server/src/routes/collab.ts` | `POST /api/collab/sessions` — a host's current schedule/envConfig becomes a new session's baseline. |
| LAN origin guard | `server/src/lanOrigin.ts` | Restricts which browser origins may call the REST API (added during the security fix described in §4). |
| Client transport | `src/services/collabService.ts` | Thin wrapper around `socket.io-client` + `fetch`: create/join a session, send an action, fetch a shareable link. |
| Session state & actions | `src/types/appState.ts`, `src/context/reducer.ts` | `SessionState`/`SessionParticipant` types; reducer cases for session lifecycle (`SET_SESSION`, `SET_SESSION_BASELINE`, presence/status updates). |
| Dispatch wrapping | `src/context/AppContext.tsx` | The architectural core: every existing component still just calls `dispatch(...)` — this one file decides whether an action needs to go out to the server, and applies incoming actions from other participants without ever echoing them back. |
| Join/Start UI | `src/components/Dialogs/SessionDialog.tsx` | Start a session, or join one by pasting a link/ID and picking Edit/View. |
| Menu bar entry | `src/components/Toolbar/MenuBar.tsx` | New "共同編集" (Collaboration) menu, replacing the old standalone share button; shows a live participant count. |
| Browser join | `src/App.tsx` | Opening a shared link directly in a browser tab (no app install) prompts for a name and joins, rendering the full editor (edit role) or a read-only mirror (view role). |
| Electron lifecycle | `electron/main.cts`, `electron/preload.cts` | Closing the host's window during a session hides it to a tray icon instead of ending the session; a real quit (any path) still works. |

Retired entirely: the old read-only "Live View" broadcast feature (`ShareViewButton`, `ShareViewDialog`, `viewBroadcastService.ts`, `useHostBroadcast`, `useViewerSync`, `server/src/collab/viewBroadcast.ts`) — confirmed fully removed, no leftover imports or dead code.

---

## 2. Automated Tests Added

### Server — 20 tests across 3 files

**`server/src/collab/sessionStore.test.ts` (11 tests)** — the pure data layer
| Test | What it checks |
|---|---|
| creates a session with the given baseline and no actions or participants | A new session starts empty and holds exactly what it was given |
| returns null for an unknown session id | Looking up a nonexistent session fails safely, doesn't throw |
| returns defensive copies — mutations to returned data do not affect internal state | *(added after a review finding — see §4)* Reading a session's data can't corrupt the store from outside |
| assigns increasing sequence numbers and stores the action | Actions get an ever-increasing order number as they're logged |
| returns null for an unknown session id (append) | Logging an action against a nonexistent session fails safely |
| adds a participant and returns the full list | Joining adds you to the roster |
| removes a participant and returns the remaining list | Leaving removes you from the roster |
| returns null for an unknown session id (participants) | Same safety check for participant operations |
| removes sessions with zero participants past the idle threshold | The empty-room cleanup sweep actually reclaims memory |
| keeps sessions that still have participants | ...but never reclaims a room someone's still in |
| keeps sessions inside the idle threshold | ...and never reclaims a room too early |

**`server/src/collab/collabSocket.test.ts` (5 tests)** — real Socket.IO connections, not mocks
| Test | What it checks |
|---|---|
| replies with the baseline, empty action log, and participant list for a fresh session | A brand-new session's `join` reply has everything a client needs to render it |
| replies with ok:false for an unknown session id | Joining a bad/expired link fails cleanly instead of hanging |
| broadcasts an edit-role action to other participants but not back to the sender | The core "no echo" guarantee, tested with two real connected sockets |
| ignores actions from view-role participants | A viewer's edits (if somehow sent) never reach anyone |
| notifies remaining participants when someone disconnects | Presence updates live when someone leaves |

**`server/src/lanOrigin.test.ts` (4 tests)** — the CORS security fix (see §4)
| Test | What it checks |
|---|---|
| allows the origins the collab feature actually needs | localhost, 127.0.0.1, and private-range LAN IPs (192.168.x, 10.x, 172.16-31.x, 169.254.x) all work |
| rejects public and non-private addresses | A random public IP or unrelated domain is refused |
| rejects lookalike domains that merely start with an allowed prefix | `10.evil.example`, `192.168.evil.example`, `localhost.evil.example` etc. do **not** sneak past — this is the exact bug described in §4 |
| rejects malformed, opaque and non-http origins | `file://`, `javascript:`, empty strings, garbage input all fail safely |

### Client — added across 4 files (baseline was 103 tests / 5 suites; now 131 / 8 suites)

**`src/__tests__/context/reducer.test.ts`** (+7 tests, new "Live collaboration session" section)
`SET_SESSION` (sets/clears), `SET_SESSION_BASELINE` (replaces schedule + resets undo/redo/selection), `SET_SESSION_CONNECTION_STATUS` (updates it, and correctly no-ops with no session), `SET_SESSION_PARTICIPANTS` (updates the roster), `OPEN_SESSION_DIALOG`/`CLOSE_SESSION_DIALOG`.

**`src/__tests__/context/AppContext.test.tsx`** (new file, 13 tests) — the dispatch-wrapping core
| Test | What it checks |
|---|---|
| forwards a syncable action to the server while in an edit session, but not a local-only one | Only real edits get sent; UI-only actions (filters, dialogs) never leave the client |
| applies a remote action via the raw dispatch without forwarding it back to the server | The no-echo-loop guarantee, on the client side |
| skips re-applying the baseline for the session creator | The person who *started* the session doesn't get their own data replayed back at them |
| forwards undo as the resulting SET_SCHEDULE snapshot, not the bare UNDO token | Undo converges everyone on the same content instead of confusing other people's undo history |
| forwards redo as the resulting SET_SCHEDULE snapshot, not the bare REDO token | Same guarantee for redo |
| does not forward actions when joined as a view-only participant | A viewer's local edits (if UI allowed one through) never sync out |
| ignores an inbound action whose type is not syncable | *(added after a review finding — see §4)* A malformed or unexpected action from the wire can't touch app state |
| ignores an inbound LOAD_FILES, which would otherwise bypass the mid-session load block | *(same finding)* Closes a specific bypass of the "don't swap documents mid-session" rule |
| filters non-syncable action types out of the sync-init log replay too | The same filter applies when a late joiner replays the whole history, not just live actions |
| rejects starting a session when no schedule is loaded yet | Can't start a session with nothing to share |
| lets a participant leave a session, clearing session state | Leaving actually clears everything and returns to normal solo editing |
| blocks LOAD_FILES while a session is active, regardless of role, and surfaces an error | *(added after a review finding — see §4)* Opening a different file mid-session is blocked centrally, with a clear error message |
| still allows LOAD_FILES normally when no session is active (solo mode is unaffected) | The safety guard above never interferes with normal solo use |

**`src/__tests__/services/collabService.test.ts`** (new file, 4 tests)
| Test | What it checks |
|---|---|
| skips the baseline replay for the creator on their first sync-init | Matches the AppContext-level guarantee above |
| replays the baseline for the creator on a later (reconnect) sync-init | *(added after a review finding — see §4)* If the creator's connection drops and reconnects, they now catch back up instead of silently missing everything |
| replays the baseline for a non-creator on every sync-init | Everyone else always gets the full current state, including on reconnect |
| reports disconnected and replays nothing when sync-init comes back not-ok | A bad/expired link fails cleanly |

**`src/__tests__/App.test.tsx`** (new file, 4 tests)
| Test | What it checks |
|---|---|
| renders the editor in solo mode (no session) | Solo use is completely unaffected — confirmed by an actual test, not just inspection |
| renders the editor for an edit-role session | Editors get the full interactive app |
| renders the read-only ViewPage for a view-role session joined from inside the app | *(added after a review finding — see §4)* Joining as "view" from inside the app, not just via a browser link, actually renders read-only |
| does not mount the editor-only hooks (Ctrl+S / Ctrl+O) for a view-role session | A viewer can't accidentally trigger editor-only keyboard shortcuts |

---

## 3. Live End-to-End Manual Verification

Automated tests prove each piece works in isolation. Before calling this done, I also ran the actual dev server (client + collab server) and drove two real browser tabs against it, to prove the pieces work together, live, with real network calls:

| Scenario | Result |
|---|---|
| Solo mode, no session | App loads and behaves exactly as before this feature existed |
| Real session created via a live API call; one tab joined as editor (pasting the link into the app), a second tab joined as viewer (opening a raw `?session=...&role=view` link) | Both correctly loaded the shared schedule |
| Presence | Participant count updated live in both tabs the instant the second person joined (1 → 2) |
| Live sync | A plan-range edit made in the editor tab appeared instantly in the read-only viewer tab, no reload |
| Undo convergence | Ctrl+Z in the editor tab correctly reverted **both** tabs together |
| Open-File guard | Clicking "Open File" while in a session did nothing (correctly blocked) |
| Leave session | One participant leaving reverted their own tab to normal solo editing with their data intact, while the other tab's session was completely unaffected |

Not verified live (no packaged-app/GUI environment available to test with): the Electron tray icon and quit behavior. That part was checked by careful code reading instead, and is lower-risk since it only affects what happens when the host's window is closed, not the sync logic itself.

---

## 4. Bugs Found (All Fixed)

Every task in the plan went through its own review before moving to the next; a handful of real bugs surfaced there and were fixed immediately. A larger, final review of the *whole* branch — after every individual piece had already passed its own review — caught several more that only show up when you look at how the pieces fit together. Both groups are listed here.

### Found and fixed during individual task reviews

| # | Bug | Where | Fix |
|---|---|---|---|
| 1 | `getSession()` returned live references to internal data, not copies — a caller could mutate the session's action log directly and desync it | `server/src/collab/sessionStore.ts` | Return shallow copies instead |
| 2 | `fetchCollabLink()` didn't check the network-info request actually succeeded before using its response | `src/services/collabService.ts` | Added the same defensive check its sibling function already had |
| 3 | The plan's own sample test code for the Socket.IO relay had a race condition — two client sockets' event listeners were attached in the wrong order, causing intermittent hangs | `server/src/collab/collabSocket.test.ts` | Fixed the test's listener ordering (the actual server code was already correct) |
| 4 | Opening a different file mid-session (via the menu, Ctrl+O, or the separate SchedulerWeb hand-off feature) could silently swap out the shared document while other people kept editing the old one | `src/context/AppContext.tsx` | Blocked centrally, in one place, regardless of which of the three ways triggered it |
| 5 | The Start-Session/Join-Session dialog fetched its share links on every render instead of once, with no error handling — could refire network calls repeatedly every time someone joined or left | `src/components/Dialogs/SessionDialog.tsx` | Moved into a proper one-time effect with error handling |
| 6 | Opening a browser-link join with a bad or expired session ID never showed an error — the user was just silently dropped into an empty editor | `src/App.tsx` | Added a failure screen — which then had a follow-up bug of its own (next row) |
| 7 | The fix for #6 was *too* aggressive — it also showed the error screen for a brief, harmless network blip on an already-connected, actively-in-use session, kicking the user out with no way back | `src/App.tsx` | Only treat a disconnect as a failure if the session never successfully connected in the first place |

### Found only in the final whole-branch review (after every task already passed its own review)

These are the ones worth reading carefully — they're exactly the kind of bug that "looks fine when you check each piece" but only shows up once everything is wired together.

| # | Severity | Bug | Where | Fix |
|---|---|---|---|---|
| 8 | **Critical** | Joining as "閲覧のみ" (view-only) *from inside the app* didn't actually show a read-only screen — only joining via a raw browser link did. Someone in this situation got a fully interactive editor whose edits silently went nowhere, with zero indication anything was wrong — and if they saved, they could write a document that had quietly diverged from the shared one | `src/App.tsx` | Made "is this session read-only?" a single decision based on the session's role, checked in one place, so both ways of joining (in-app and by link) land on the same answer |
| 9 | Important | The server only accepted requests from `localhost:5173`/`:5174` — but a joiner on the LAN (the entire point of this feature) loads the page from a different address (their local IP), so their browser's requests to create/join sessions were silently rejected | `server/src/index.ts` | Allow LAN-reachable origins too (see #12 for the follow-up this caused) |
| 10 | Important | Incoming actions from other participants were applied with no check on what *kind* of action they were — only outgoing actions were filtered. A different or buggy client version (or a deliberately misbehaving one) could send an action type that bypassed the mid-session file-load block (bug #4) or injected something else | `src/context/AppContext.tsx` | Apply the exact same filter to incoming actions as outgoing ones |
| 11 | Important | Closing the app entirely (Cmd/Ctrl+Q, the app's own quit menu, a system shutdown) while a session was active got silently swallowed by the "keep running in the background" fix — there was no way to tell "the user wants to quit" apart from "the user just closed a window." Combined with a placeholder tray icon that was fully transparent (invisible), a host in this situation could end up with no window, no visible way back, and no way to quit except Task Manager | `electron/main.cts` | Added a real "the app is actually quitting" flag that always lets a real quit through; replaced the invisible placeholder with a real, visible icon |
| 12 | Important | The session creator's connection dropping and reconnecting (which happens automatically) meant they permanently stopped receiving anyone else's edits from that point on, while the app reported everything as fine | `src/services/collabService.ts` | Only skip the very first sync after creating a session — every reconnect after that catches up normally, like everyone else |
| 13 | Important | A React rule ("hooks must run in the same order every time") was being broken in a way that happened to work today but would break the moment a small, unrelated change was made elsewhere | `src/components/Dialogs/SessionDialog.tsx` | Reordered the code to follow the rule properly |
| 14 | Important | The server's own test files were being compiled into the packaged app that ships to users, and running the test suite was silently running every test twice (once from source, once from a stale old build) | `server/tsconfig.json`, new `server/vitest.config.ts` | Excluded test files from what gets built/shipped, and told the test runner to only look at the real source files |
| 15 | Important | "Add a new schedule from scratch" (a separate, destructive action) wasn't blocked during a session, even though it wipes out what everyone's currently looking at, with no warning | `src/components/Toolbar/Toolbar.tsx` | Disabled that button while a session is active, matching the existing "Open File" guard |
| 16 | **Important — introduced by the fix for #9 above** | Fixing the LAN-access problem (#9) by allowing origins broadly also accidentally reopened a *different*, pre-existing endpoint (`/api/save-files`, which writes to whatever file path it's told) to being called from **any website**, not just this app or LAN devices — meaning simply having this app open in the background while browsing the web elsewhere could, in theory, let some other page silently trigger a file write on your machine | `server/src/index.ts`, new `server/src/lanOrigin.ts` | Replaced the broad allowance with a real check for "is this actually localhost or a private LAN address?" — and while writing that check, caught and closed one more subtle bypass (see #17) |
| 17 | Minor, caught during the fix for #16 | The first draft of the "is this a LAN address?" check would have accepted a domain name crafted to merely *start with* a LAN-like prefix, e.g. `10.evil.example` or `192.168.evil.example` — which are real, purchasable domain names an attacker controls, not LAN addresses at all | `server/src/lanOrigin.ts` | Rewrote the check to parse the actual address properly and match it exactly, not just check if the text starts with the right characters |

**Total: 17 bugs found, 17 fixed.** Every one was caught by either an automated test, a task-scoped review, or the final whole-branch review — none were found by a user after the fact, because none had shipped yet.

---

## 5. Known Limitations & Frequently Asked Questions

### Why do I have to type a name to join a session?

Right now, honestly: **it's captured, but nothing in the app actually displays it yet.** The participant count you see next to the menu ("🟢 2人が参加中") is just a number — the server and client both already track each person's name internally (it's sent when you join, stored, and broadcast to everyone), but no screen currently lists names, only the total count.

This was intentional groundwork, not a finished feature: the original design called for "an avatar list of who's currently in the session," and the underlying data for that already exists and works — displaying it is just a small UI addition that wasn't part of this build. It would be a quick follow-up (e.g., hovering the participant count to see who's in, or a small list in the session panel) since no new data plumbing is needed.

### How do I let someone join who's *not* on the same network or VPN as the host?

As built, this only works for people who can actually reach the host machine's network address — same Wi-Fi/LAN, or the same VPN. The shared link is built from the host's local network IP (e.g. `192.168.1.23`), which simply isn't reachable from outside that network; someone elsewhere trying to open it will just get a connection failure, not an error message.

Three ways to extend this, in order of how well they fit the current design:

1. **Put both people on the same VPN (recommended).** A mesh VPN like Tailscale (free, very quick to set up, no router configuration) puts both machines on what is effectively a private LAN reachable from anywhere. Once connected, the existing link mechanism works exactly as it does today — no code changes needed, beyond possibly making sure the link picks the VPN's IP address rather than the office LAN's if a machine has both.
2. **Forward a port on the host's router and share the public IP.** Technically possible, but **not recommended without adding real authentication first.** This app has no login by design (that was an explicit, deliberate choice for how it's meant to be used — trusted people on a trusted network). Exposing it to the raw internet means anyone who finds or guesses the address could interact with it directly — the CORS restriction added in this release (bug #16/#17 above) only stops a *browser's JavaScript* from doing that, not a person using a script or command-line tool. Doing this safely would need a proper access-control feature added first, which is a separate piece of work.
3. **A tunnel service (ngrok, Cloudflare Tunnel, etc.)** gets you a public URL without touching the router, but carries the same authentication risk as option 2, and reintroduces exactly the kind of third-party relay the original design deliberately avoided for this version.

**Bottom line:** for anyone outside the LAN today, get them onto the same VPN as the host. If sharing with people over the open internet (no VPN) becomes a real need, that calls for adding actual authentication as its own follow-up feature — happy to scope that out when it's needed.
