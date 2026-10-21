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
    // Computed candidates (filled once per solve stage)
    private List<EmployeeFact> candidateEmployees = List.of();

    private static final EmployeeFact UNASSIGNED =
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
                    return tolerant(new ListValueRange<>(List.of(e)));
                }
            }
            // For pinned but missing
            return tolerant(new ListValueRange<>(List.of(UNASSIGNED)));
        }

        List<EmployeeFact> base = (candidateEmployees == null) ? List.of() : candidateEmployees;

        if (needManager) {
            // STRICT: manager-only, no UNASSIGNED fallback
            base = base.stream().filter(emp -> emp != null && emp.isManager).toList();
            // IMPORTANT: do NOT add UNASSIGNED here
            return tolerant(new ListValueRange<>(base));
        } else {
            // Non-manager seat may keep UNASSIGNED as a legal value to ease feasibility
            // if (base.isEmpty()) base = List.of(UNASSIGNED);
            List<EmployeeFact> list = new ArrayList<>(base == null? List.of(): base);
            list.add(UNASSIGNED);
            list = list.stream().distinct().toList();
            return tolerant(new ListValueRange<>(list));
        }
    }

    /**
     * Timefold 1.35 validates the initial value against the range via {@code contains} only; the initial value may be
     * outside the original range (see {@link ValidationTolerantRange}). The original range is returned unchanged
     * otherwise, so UNASSIGNED is never offered to manager seats.
     */
    private CountableValueRange<EmployeeFact> tolerant(CountableValueRange<EmployeeFact> range) {
        return new ValidationTolerantRange<>(range, employee);
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
