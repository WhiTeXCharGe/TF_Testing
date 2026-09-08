# One image, two roles: set ROLE=aca1 or ROLE=aca2 (see src/config.ts).
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
