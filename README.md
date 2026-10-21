# Timefold employee scheduler

Refactored copy of `TFDocker` (Timefold Solver 1.27.0 employee scheduler).
Main class: `com.yourorg.scheduler.Main`.

## Build

    mvn -B -DskipTests package

Produces `target/timefold-scheduler-0.1.0.jar` and the runtime dependency jars in `target/dependency/`.

Where Maven is not installed locally, run it in Docker instead:

    docker run --rm -v "$(pwd):/w" -v tf-m2:/root/.m2 -w /w maven:3.9-eclipse-temurin-17 mvn -B -DskipTests package
    docker run --rm -v "$(pwd):/w" -v tf-m2:/root/.m2 -w /w maven:3.9-eclipse-temurin-17 mvn -B test

On Git Bash for Windows prefix the command with `MSYS_NO_PATHCONV=1` and use `$(pwd -W)` instead of `$(pwd)`.

## Docker

    docker build -t timefold-scheduler:local .

Mount `EnvConfig.yaml` and `Schedule.yaml` at `/work/input`; the result is written to `/work/output/result_Schedule.yaml`
and run status to `/work/status/<RUN_ID>.json`. Sample inputs are in `src/main/resource/`.
