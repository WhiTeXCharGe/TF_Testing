# One image, two roles: set ROLE=aca1 or ROLE=aca2 (see GanttChartEditor/server/src/config.ts).
#
# Lives in this separate repo, not inside GanttChartEditor, but the build
# CONTEXT is still GanttChartEditor/server — every COPY below is relative to
# that context, not to this file's own location. Build it with:
#
#   az acr build -r <registry> -t gantt-collab:<tag> \
#     -f Dockerfile ../GanttChartEditor/server
#
# (run from this folder — see deploy.sh, or GanttChartEditor_ACA_ContainerBuildAndPush.md
# in documents/GanttChartEditor/ for the full walkthrough).
FROM node:20-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY tsconfig.json ./
COPY src ./src
RUN npm run build

FROM node:20-alpine
WORKDIR /app
ENV NODE_ENV=production
COPY package.json package-lock.json ./
RUN npm ci --omit=dev
COPY --from=build /app/dist ./dist
# Default port; compose / ACA override PORT per role.
EXPOSE 4000
CMD ["node", "dist/index.js"]
