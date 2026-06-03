# ─── Stage 1: build the JAR + collect runtime deps ───────────────────────────
# Maven image that already has JDK 17, so `mvn package` works out of the box.
FROM maven:3.9-eclipse-temurin-17 AS builder

WORKDIR /build

# Use the STANDALONE pom (the real pom.xml has a <parent> pointing at a file
# that isn't in the Docker build context, which would fail "Non-resolvable
# parent POM"). The standalone pom is functionally identical for our purposes.
COPY docker/pom-standalone.xml ./pom.xml

# Pull dependencies first so source-only edits reuse this cached layer.
RUN mvn -B -ntp -DskipTests dependency:go-offline

# Copy the source after deps for better layer caching.
COPY src ./src

# Builds the jar AND (via maven-dependency-plugin) copies runtime deps to
# target/dependency/. Both go into /app/lib in the runtime image.
RUN mvn -B -ntp -DskipTests clean package

# ─── Stage 2: tiny runtime image with just the JRE + our jars ────────────────
FROM eclipse-temurin:17-jre

LABEL org.opencontainers.image.title="timefold-scheduler"
LABEL org.opencontainers.image.description="Timefold employee scheduler — reads /work/input/{EnvConfig,Schedule}.yaml, writes /work/output/result_Schedule.yaml"
LABEL org.opencontainers.image.source="https://github.com/WhiTeXCharGe/TF_Testing"

# Non-root user (matches what ACA / k8s will give us in production).
RUN useradd --create-home --uid 10001 solver

WORKDIR /app

# Application jar + all transitive runtime dependency jars in one folder.
# entrypoint.sh sets the classpath to /app/lib/*.
COPY --from=builder /build/target/*.jar             /app/lib/
COPY --from=builder /build/target/dependency/*.jar  /app/lib/

# Wrapper script: writes status.json, runs solver, handles SIGTERM (cancel).
COPY docker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh && chown -R solver:solver /app

USER solver

# /work is the conventional mount point. Bind-mount your local folders here:
#   /work/input   ← EnvConfig.yaml + Schedule.yaml (read-only is recommended)
#   /work/output  ← result_Schedule.yaml written by the solver
#   /work/status  ← status/<RUN_ID>.json updated throughout the run
ENV INPUT_DIR=/work/input \
    OUTPUT_DIR=/work/output \
    STATUS_DIR=/work/status \
    JVM_MAX_HEAP=6g

ENTRYPOINT ["/app/entrypoint.sh"]
