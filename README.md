# Timefold employee scheduler

Refactored copy of `TFDocker` (Timefold Solver 1.27.0 employee scheduler).
Main class: `com.yourorg.scheduler.EmployeeSchedule`.

## Build

    mvn -B -DskipTests package

Produces `target/timefold-scheduler-0.1.0.jar` and the runtime dependency jars in `target/dependency/`.

## Docker

    docker build -t timefold-scheduler:local .

Mount `EnvConfig.yaml` and `Schedule.yaml` at `/work/input`; the result is written to `/work/output/result_Schedule.yaml`
and run status to `/work/status/<RUN_ID>.json`. Sample inputs are in `src/main/resource/`.
