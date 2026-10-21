package com.yourorg.scheduler.solver;

import java.time.LocalDate;

import com.yourorg.scheduler.model.SinglePassPlan;

/** Outcome of a solve: the best plan and the plan start date the exporter needs. */
public class RunResult { public SinglePassPlan plan; public LocalDate planStart; }
