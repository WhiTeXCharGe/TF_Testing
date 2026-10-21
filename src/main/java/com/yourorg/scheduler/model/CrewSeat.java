package com.yourorg.scheduler.model;

import ai.timefold.solver.core.api.domain.entity.PlanningEntity;
import ai.timefold.solver.core.api.domain.entity.PlanningPin;
import ai.timefold.solver.core.api.domain.lookup.PlanningId;
import ai.timefold.solver.core.api.domain.valuerange.CountableValueRange;
import ai.timefold.solver.core.api.domain.valuerange.ValueRangeProvider;
import ai.timefold.solver.core.api.domain.variable.PlanningVariable;
import ai.timefold.solver.core.impl.domain.valuerange.buildin.collection.ListValueRange;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Set;

@PlanningEntity(difficultyComparatorClass = CrewSeat.DifficultyComparator.class)
public class CrewSeat {
    @PlanningId public int id;

    public int blockId;
    public String module;
    public String workflowId;
    public String factory;
    public String phaseId;
    public int phaseNum;
    public String opId;

    public int seatIndex;
    public boolean needManager;

    public boolean pinnedFixed = false;
    public boolean warmPinned  = false;
    public String  pinnedWid = null;
    public Integer pinnedStart = null;
    public Integer pinnedDays  = null;
    public Integer pinnedHours = null;
    public Set<Integer> pinnedWorkDays = null;
    @PlanningPin
    public boolean isPinned() {
        // Only truly fixed assignments are pinned.
        // Warm-start assignments are NOT pinned.
        return pinnedFixed;
    }
    /**
     * Employee taken over from the input schedule as the initial (warm-start) value, set by PlanBuilder. It may not
     * pass the candidate gates (skill, preference, manager). Timefold 1.35 requires the initial value to be inside
     * the value range, so {@link #eligibleEmployeesForSeat()} appends it when it is missing.
     */
    public EmployeeFact warmStartEmployee = null;

    // Computed candidates (filled once per solve stage)
    private List<EmployeeFact> candidateEmployees = List.of();

    /**
     * Shared "nobody" sentinel. Timefold 1.35 checks once, when the working solution is set, that every
     * assigned planning value is contained in its entity's value range (contains = equals = identity here),
     * so the initial value of an unassigned seat must be THIS instance (not a fresh copy) and must be
     * part of the seat's range, including manager seats (see {@link #eligibleEmployeesForSeat()}).
     */
    public static final EmployeeFact UNASSIGNED =
        new EmployeeFact(0, "__UNASSIGNED__", "__UNASSIGNED__", Map.of(), false, "");

    public CrewSeat() {}

    public void setCandidateEmployees(List<EmployeeFact> list) {
        this.candidateEmployees = (list == null) ? List.of() : list;
    }
    public List<EmployeeFact> getCandidateEmployees() { return candidateEmployees; }

    @ValueRangeProvider(id = "eligibleEmployeesForSeat")
    public CountableValueRange<EmployeeFact> eligibleEmployeesForSeat() {
        // Pinned → pin hard
        if (pinnedFixed && pinnedWid != null) {
            for (EmployeeFact e : candidateEmployees) {
                if (e != null && pinnedWid.equals(e.wid)) {
                    return new ListValueRange<>(List.of(e));
                }
            }
            // For pinned but missing
            return new ListValueRange<>(List.of(UNASSIGNED));
        }

        List<EmployeeFact> base = (candidateEmployees == null) ? List.of() : candidateEmployees;

        if (needManager) {
            // STRICT: manager-only candidates. The UNASSIGNED sentinel is in the range ONLY because every seat
            // starts as UNASSIGNED and Timefold 1.35 rejects an initial value outside the range (it was not
            // checked in 1.27). Appended last, so the order of the real candidates is unchanged.
            List<EmployeeFact> mgr = new ArrayList<>(base.stream().filter(emp -> emp != null && emp.isManager).toList());
            mgr.add(UNASSIGNED);
            addWarmStart(mgr);
            return new ListValueRange<>(mgr);
        } else {
            // Non-manager seat may keep UNASSIGNED as a legal value to ease feasibility
            // if (base.isEmpty()) base = List.of(UNASSIGNED);
            List<EmployeeFact> list = new ArrayList<>(base == null? List.of(): base);
            list.add(UNASSIGNED);
            list = new ArrayList<>(list.stream().distinct().toList());
            addWarmStart(list);
            return new ListValueRange<>(list);
        }
    }

    private void addWarmStart(List<EmployeeFact> range) {
        if (warmStartEmployee != null && !range.contains(warmStartEmployee)) {
            range.add(warmStartEmployee);
        }
    }

    @PlanningVariable(
            valueRangeProviderRefs = "eligibleEmployeesForSeat",
            strengthComparatorClass = EmployeeStrength.class
    )
    public EmployeeFact employee;

    public static class EmployeeStrength implements Comparator<EmployeeFact> {
        @Override
        public int compare(EmployeeFact a, EmployeeFact b) {
            if (a == b) return 0;
            if (a == null) return 1;
            if (b == null) return -1;

            boolean aUnassigned = (a.id == 0);
            boolean bUnassigned = (b.id == 0);
            if (aUnassigned != bUnassigned) return aUnassigned ? 1 : -1;

            int aSkills = WorkerRules.positiveSkillCount(a);
            int bSkills = WorkerRules.positiveSkillCount(b);
            if (aSkills != bSkills) {
                return Integer.compare(aSkills, bSkills); // fewer skills first
            }

            if (a.isManager != b.isManager) {
                return a.isManager ? 1 : -1; // non-manager first in generic order
            }

            return Integer.compare(a.id, b.id);
        }
    }
    public static class DifficultyComparator implements Comparator<CrewSeat> {
        @Override
        public int compare(CrewSeat a, CrewSeat b) {
            if (a == b) return 0;
            if (a == null) return -1;
            if (b == null) return 1;

            // keep only manager-seat priority
            if (a.needManager != b.needManager) {
                return a.needManager ? 1 : -1; // manager seat harder
            }
            //candiates have same priority
            return 0;
        }
    }
}
