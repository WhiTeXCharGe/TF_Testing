# GanttChart Editor — Setup Guide

---

## Prerequisites

Install these before starting:

| Tool | Version | Download |
|---|---|---|
| **Node.js** | v18 or later (v20 recommended) | https://nodejs.org |
| **Git** | Any recent version | https://git-scm.com |
| **VS Code** | (Recommended) | https://code.visualstudio.com |

To check if already installed, run in terminal:
```
node --version
git --version
```

---

## 1. Get the Code

### If cloning from GitHub

```bash
git clone https://github.com/WhiTeXCharGe/TF_Testing.git
cd TF_Testing
git checkout GanttChartEditor
cd GanttChartEditor
```

### If you copied the folder directly

Just open a terminal inside the `GanttChartEditor` folder.

---

## 2. Install Dependencies

Run this once after cloning (or whenever `package.json` changes):

```bash
npm install
```

---

## 3. Start the Dev Server

```bash
npm run dev
```

Then open your browser at:
```
http://localhost:5173
```

---

## 4. Load a Schedule

1. Click **ファイル** → **ファイルを開く** (or press `Ctrl+O`)
2. Select `EnvConfig.yaml` first, then `Schedule.yaml`
3. The Gantt chart will load

Sample YAML files are in `GanttChartEditor/documents/`:
- `EnvConfig.yaml`
- `Schedule.yaml`

---

## 5. Other Useful Commands

| Command | What it does |
|---|---|
| `npm run dev` | Start development server (hot reload) |
| `npm run build` | Build for production (output to `dist/`) |
| `npm run preview` | Preview the production build locally |

---

## Notes

- The app runs entirely in the browser — no backend server needed for Phase 1–2
- All data is in memory; refreshing the page clears loaded data (save first with `Ctrl+S`)
- Tested on Node.js 18 and 20, Windows 10/11
