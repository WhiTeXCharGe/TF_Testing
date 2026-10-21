package com.yourorg.scheduler.config;

/** Fixed numbers shared by the plan builder, the parsers and the constraints. */
public final class SolverConstants {
    private SolverConstants() {}

    public static final int DAILY_CAP = 12;
    public static final int REQUIRED_HOURS_PER_WORKLOAD_DAY = 10;

    public static final boolean TRIM_FINISHED_MODULES = false;
    // if module ended more than this many days before cutOffDate,
    // it is considered finished and can be trimmed.
    public static final int MODULE_TRIM_GRACE_DAYS = 6;

    public static final int BASE_HOURS_PER_DAY = 8;
}
