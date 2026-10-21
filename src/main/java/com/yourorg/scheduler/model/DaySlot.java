package com.yourorg.scheduler.model;

import ai.timefold.solver.core.api.domain.lookup.PlanningId;
import java.time.LocalDate;

public class DaySlot {
    @PlanningId public int id;
    public LocalDate date;
    public DaySlot() {}
    public DaySlot(int id, LocalDate d) { this.id = id; this.date = d; }
}
