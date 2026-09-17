# gantt-collab-container

Docker build + Azure deploy for GanttChartEditor's online-collaboration
backend (ACA1 session API + ACA2 live relay). Deliberately a **separate repo**
from `GanttChartEditor` — this holds the container packaging, not app code.

> Full walkthrough (prerequisites, every command explained, what to send the
> Azure admin, what happens on a feature update): see
> **`../documents/GanttChartEditor/GanttChartEditor_ACA_ContainerBuildAndPush.md`**.
> Resource-planning / RBAC decisions: `..._OnlineCollab_AzurePrep20260908.md` in
> the same folder.

## Layout

```
web/
├─ GanttChartEditor/          ← app source (own git repo). server/ has NO
│                                Dockerfile — just the Node.js source.
└─ gantt-collab-container/    ← THIS repo: Dockerfile + deploy.sh only.
   ├─ Dockerfile                Build context is ../GanttChartEditor/server,
   ├─ deploy.sh                 not this folder — see Dockerfile's own header.
   └─ README.md
```

`GanttChartEditor/server/.dockerignore` stays where it is — `.dockerignore`
must sit at the build **context** root, which is `server/`, regardless of
where the Dockerfile itself lives.

## Quick start (once `az login` + resource decisions are done)

```bash
cd gantt-collab-container
export SUFFIX=abc123           # short, globally unique (ACR + storage names)
export WEB_ORIGIN=https://<static-web-apps-url-or-custom-domain>
./deploy.sh
```

`GANTT_SERVER_DIR` (default `../GanttChartEditor/server`) is the only thing
to override if your checkout isn't two sibling folders.

## Just rebuild + push (after a GanttChartEditor code change)

```bash
cd gantt-collab-container
TAG=$(git -C ../GanttChartEditor rev-parse --short HEAD)
az acr build -r <acr-name> -t gantt-collab:$TAG -f Dockerfile ../GanttChartEditor/server
az containerapp update -n ca-gantt-aca2 -g <rg> --image <acr-name>.azurecr.io/gantt-collab:$TAG
az containerapp update -n ca-gantt-aca1 -g <rg> --image <acr-name>.azurecr.io/gantt-collab:$TAG
```

Yes — **every server code change needs a rebuild + push + `containerapp
update`** to actually reach Azure; nothing auto-syncs from a local edit or a
`git commit`. See the full doc's "What happens when you update a feature"
section for why, and for the desktop-app half of the answer (it doesn't need
any of this).

## Local testing without Azure

Doesn't need this repo at all — `GanttChartEditor/server/docker-compose.mock.yml`
builds from this same Dockerfile (via a `dockerfile:` override pointing back
here) against a filesystem stand-in for Blob. See `GanttChartEditor/server/MOCK_AZURE.md`.
