package com.yourorg.scheduler;

import java.util.Arrays;
import java.util.List;

public class IncrementalConfig {

    // ---- from config_base.py ----
    public static final int WORKER_NUM = 400;

    public static final double EQ_PER_DAYS = 2.5;
    public static final double EQ_PER_DAYS_SIGMA = 2.5;
    public static final int EQ_NUM = 300;           // total modules target

    public static final List<Integer> SKILL_LEVEL_LIST =
            Arrays.asList(1, 2, 3, 4, 5);
    public static final List<Double> SKILL_LEVEL_WEIGHTS =
            Arrays.asList(0.05, 0.1, 0.5, 0.25, 0.1);

    public static final double MANAGER_RATE = 0.5;
    public static final int WORKER_COMPANY_NUM = 2;

    // kept for fallback when region unavailable_dates are not set at all
    public static final boolean IS_SKIP_WEEKEND = true;

    // levels: 0, 1, 2
    public static final List<Integer> REGION_SUITABILITY_LIST =
            Arrays.asList(0, 1, 2);
    public static final List<Double> REGION_SUITABILITY_WEIGHTS =
            Arrays.asList(0.2, 0.5, 0.3);

    public static final List<Integer> CUSTOMER_SUITABILITY_LIST =
            Arrays.asList(0, 1, 2);
    public static final List<Double> CUSTOMER_SUITABILITY_WEIGHTS =
            Arrays.asList(0.2, 0.5, 0.3);

    // ---- from update_config.py ----
    public static final int PLAN_RANGE_EXTRA_DAYS = 3;

    public static final int ENV_SEED = 152;
    public static final int MODULE_SEED = 173;

    // how many working days per evaluation step (like EQ_EVAL_DAYS)
    public static final int EQ_EVAL_DAYS = 5;

    // worklength patterns (from normal_worklength / vip_worklength in Python)
    // Each inner list is: [ phase_total_days, [workload_days_per_operation...] ]

    public static final List<Object> NORMAL_WORKLENGTH = Arrays.asList(
            Arrays.asList(15, Arrays.asList(30, 20, 20)),
            Arrays.asList(12, Arrays.asList(30, 25)),
            Arrays.asList(6,  Arrays.asList(15)),
            Arrays.asList(7,  Arrays.asList(12, 8))
    );

    public static final List<Object> VIP_WORKLENGTH = Arrays.asList(
            Arrays.asList(12, Arrays.asList(30, 20, 20)),
            Arrays.asList(10, Arrays.asList(30, 25)),
            Arrays.asList(6,  Arrays.asList(15)),
            Arrays.asList(8,  Arrays.asList(12, 8))
    );

    // ---- file paths (relative to project root) ----
    public static final String ENV_PATH = "src/main/resource/EnvConfig.yaml";
    public static final String SCHEDULE_IN_PATH = "src/main/resource/Schedule.yaml";
    public static final String SCHEDULE_OUT_PATH = "src/main/resource/Schedule.yaml";
}
