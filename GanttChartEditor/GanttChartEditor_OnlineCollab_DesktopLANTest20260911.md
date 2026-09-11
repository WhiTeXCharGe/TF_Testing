# Desktop-app LAN test — online collaboration (before Azure)

> **Date:** 2026-09-11 · **Branch:** `GanttChartEditor`
> Test the online-collab feature with the packaged desktop app over your office network — no Azure, no Docker, no browser.

---

## How it works now

`ROLE=local` (what the packaged app runs) is a **self-contained ACA1 + ACA2 + web** on one port (`3010`):

- The app's bundled server hosts the session API, the live relay, and the built UI, all on `:3010`.
- The Electron window loads `http://localhost:3010` — so it talks to its **own** server.
- **No one needs to type an address.** Every app on `ROLE=local` quietly UDP-broadcasts itself on the LAN and listens for others doing the same. The 参加 dialog lists other machines it's heard by name as clickable chips, and auto-selects one if there's only one — the **接続先サーバー** field is a manual fallback (different subnet, firewalled), not the normal path.

Session state is in memory — a meeting ends when the host closes the app. (On Azure it will survive; that's the difference that motivated the cloud move. On Azure the address is also automatic, baked into the build — discovery is specifically a LAN-testing convenience.)

---

## 1. Build the installer

```bash
cd C:/Users/PC_USER/OneDrive/Desktop/work/Timefold/web/GanttChartEditor
npm install
npm run electron:build
```

Output: `release/GanttChartEditor Setup <version>.exe` (Windows NSIS). Install it on every machine that will take part (host + participants).

> Do **not** set `VITE_ACA1_URL` for a LAN build — leaving it unset makes the app use its own origin, which is exactly what you want here. (`VITE_ACA1_URL` is only for the Azure build later.)

---

## 2. Firewall (only thing to check up front)

Discovery and joining both need the host reachable on the LAN. The first time, Windows will prompt for **Node.js** — allow it on the **Private** network (or open UDP `41237` + TCP `3010` manually) on every machine that will host.

If you ever need the host's address by hand (discovery didn't find it, different subnet): ファイル → オンラインセッションを作成 shows **他の参加者は次のアドレスを…**, e.g. `http://192.168.1.5:3010` — or `ipconfig` for the IPv4 address (port is always `3010`).

---

## 3. Run a session

**Host**

1. ファイル → 開く → load `EnvConfig.yaml` + `Schedule.yaml` (or `Test_data/*`).
2. ファイル → **オンラインセッションを作成** → leave 接続先サーバー **blank** (= this PC) → enter 表示名 + セッション名 → 作成して開始 (or upload the two YAML files).
3. You're in the editor; 共同編集 menu appears.

**Each participant**

1. ファイル → **オンラインセッションに参加**.
2. If only the host is on the LAN, **接続先サーバー auto-fills** and the list shows the host's sessions immediately. With more than one machine around, click its name under 「同じネットワークで見つかったPC」 instead — no typing either way. (Manual entry is still there as a fallback.)
3. Enter 表示名 (remembered next time), pick 編集 / 閲覧のみ, select the session row, click **参加**.
4. Edit together — changes propagate in ~1 s.

**Lock / unlock:** 共同編集 → セッション情報 → ロックする / ロック解除. Any participant can toggle it; while locked, everyone is read-only.

**Leave:** 共同編集 → セッションを終了. When the last person leaves, the session goes 停止中; re-opening it replays the edits (as long as the host app is still running).

---

## 4. What to verify

- [ ] Installer builds and installs on 2+ machines
- [ ] Host creates a session; the create dialog shows its `http://<ip>:3010` address (fallback path)
- [ ] A participant's join dialog discovers the host by name and lists its session, with no address typed
- [ ] Bar drags / date edits / undo-redo sync both ways within ~1 s
- [ ] 表示名 is pre-filled on the participant's second join
- [ ] Lock from a non-creator freezes editing for everyone; unlock restores it
- [ ] 最終参加 time in the list updates after each join; list is sorted most-recent-first
- [ ] Participant leaves and rejoins → sees the current state (replayed)
- [ ] Closing the host app ends the session (expected — memory only until Azure)

---

## 5. Quick check without building (optional)

```bash
cd GanttChartEditor
npm run dev:all        # vite :5173 + local server :3010
```

Open `http://localhost:5173` (and `http://<lan-ip>:5173` on another machine, 接続先サーバー blank — the Vite dev server proxies `/api` to `:3010`). Same behaviour, no installer.

---

## Notes / limits

- **In-memory state.** Restarting the host app clears all sessions. Set `STORAGE=fs` + `MOCK_BLOB_DIR=<path>` as env on the server if you want them to survive a restart during testing (not wired into the packaged app by default).
- **One host at a time.** Every installed app *can* host, but a session lives on whichever machine created it; participants point at that one.
- **The Azure build** is the same code with `VITE_ACA1_URL=<aca1 FQDN>` baked in and `STORAGE=blob` — see `GanttChartEditor_OnlineCollab_AzurePrep20260908.md`.
