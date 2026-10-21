package com.yourorg.scheduler.model;

import java.time.LocalDate;

import ai.timefold.solver.core.api.domain.common.PlanningId;

public class DaySlot {
    @PlanningId public int id;
    public LocalDate date;
    public DaySlot() {}
    public DaySlot(int id, LocalDate d) { this.id = id; this.date = d; }
}
