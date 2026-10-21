package com.yourorg.scheduler.model;

import java.util.HashSet;
import java.util.Set;

import com.yourorg.scheduler.calendar.SolveContext;

/** Pure rules and lookups over {@link EmployeeFact}: assignment status, skills, affinity, preferences. */
public final class WorkerRules {

    private WorkerRules() {}

    public static boolean isUnassigned(EmployeeFact e) { return e == null || e.id == 0; }

    public static int skill(EmployeeFact e, String opId) { return (e == null) ? 0 : e.skills.getOrDefault(opId, 0); }

    public static boolean isManager(EmployeeFact e) { return e != null && e.isManager; }

    public static String company(EmployeeFact e) { return e == null ? "" : (e.workerCompany == null ? "" : e.workerCompany); }

    public static double avgSkill(String opId) { return SolveContext.current().opAvgSkill.getOrDefault(opId, 3.0); }

    public static int computeAffinityEdge(EmployeeFact e1, EmployeeFact e2) {
        if (e1 == null || e2 == null) return 0;
        if (e1.affinityTags.isEmpty() || e2.affinityTags.isEmpty()) return 0;
        Set<String> e2Tags = new HashSet<>(e2.affinityTags);
        int weight = 0;
        for (String tag : e1.affinityTags) {
            if (e2Tags.contains(tag)) {
                weight += SolveContext.current().affinityTagWeights.getOrDefault(tag, 0);
            }
        }
        return weight;
    }

    public static int positiveSkillCount(EmployeeFact e) {
        if (e == null || e.skills == null) return 0;
        int c = 0;
        for (Integer v : e.skills.values()) {
            if (v != null && v > 0) c++;
        }
        return c;
    }

    public static int workerTypeRank(EmployeeFact e, String opId) {
        if (e == null || e.id == 0) return 999;
        return e.isRegularForOperation(opId) ? 0 : 1;
    }

    public static int regionPref(EmployeeFact e, String regionId) {
        if (e == null || regionId == null || regionId.isBlank()) return 1;
        return e.regionPreference.getOrDefault(regionId, 1);
    }

    public static int customerPref(EmployeeFact e, String customerId) {
        if (e == null || customerId == null || customerId.isBlank()) return 1;
        return e.customerPreference.getOrDefault(customerId, 1);
    }
}
