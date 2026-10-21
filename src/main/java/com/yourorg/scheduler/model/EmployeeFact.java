package com.yourorg.scheduler.model;

import ai.timefold.solver.core.api.domain.lookup.PlanningId;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

public class EmployeeFact {
    @PlanningId public int id;     // 0 = UNASSIGNED
    public String wid;
    public String name;
    public Map<String,Integer> skills = new HashMap<>();
    public boolean isManager;
    public String workerCompany;

    public Map<String,Integer> regionPreference   = new HashMap<>();
    public Map<String,Integer> customerPreference = new HashMap<>();
    public Map<String,String> workerTypeByOperation = new HashMap<>();
    public List<String> affinityTags = new ArrayList<>();

    // -------- HISTORY (fixed / past schedule summary) --------
    // total fixed hours over ALL history want to consider (before cutoff, etc.)
    public int histFixedTotalHours = 0;

    // fixed hours per dayId (only needed for active horizon days;)
    public Map<Integer,Integer> histFixedHoursByDay = new HashMap<>();

    // fixed factories per dayId (same)
    public Map<Integer,Set<String>> histFixedFactoriesByDay = new HashMap<>();

    public EmployeeFact() {}
    public EmployeeFact(int id, String wid, String name, Map<String,Integer> skills,
                        boolean isManager, String company) {
        this.id = id; this.wid = wid; this.name = name;
        if (skills != null) this.skills = new HashMap<>(skills);
        this.isManager = isManager; this.workerCompany = company;
    }

    public String workerTypeForOperation(String opId) {
        if (opId == null || opId.isBlank()) return "regular";
        String t = workerTypeByOperation.get(opId);
        if (t == null || t.isBlank()) return "regular";
        return t.trim().toLowerCase(Locale.ROOT);
    }

    public boolean isRegularForOperation(String opId) {
        return !"spot".equals(workerTypeForOperation(opId));
    }
    // last fixed assignment info (latest fixed day in history)
    public int histLastFixedDayId = Integer.MIN_VALUE;  // -inf means none
    public String histLastFixedRegion = null;
    public String histLastFixedFactory = null;
}
