package com.yourorg.scheduler.model;

import java.time.LocalDate;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class ParsedSchedule {
    public LocalDate planStart;
    public LocalDate planEnd;

    public List<DaySlot> daySlots;
    public List<TaskWindow> windows;
    public Map<String,Integer> requiredByKey;

    // Fixed assignments parsed from Schedule.yaml
    public List<FixedAssign> fixedRows;
    public List<FlexibleAssign> flexibleRows;
    public Map<String,Integer>  flexibleHoursByKey;
    // modules that are still relevant after trim (for dynamic blocks / warm start)
    public Set<String> activeModules;
    public Map<String,Integer> fixedHoursByKeyAll;     // includes history before cutoff
    public Map<String,Integer> fixedHoursByKeyHorizon; // horizon only
    // flexible warm-start rows
    public List<InitialAssign> initialRows;

    // for logging / debugging
    public int cutOffDayId;
    public LocalDate cutOffDate;
}
