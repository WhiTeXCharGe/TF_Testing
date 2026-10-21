package com.yourorg.scheduler.model;

import ai.timefold.solver.core.api.domain.common.PlanningId;
import ai.timefold.solver.core.api.domain.entity.PlanningEntity;
import ai.timefold.solver.core.api.domain.entity.PlanningPin;
import ai.timefold.solver.core.api.domain.valuerange.ValueRangeProvider;
import ai.timefold.solver.core.api.domain.variable.PlanningVariable;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Set;

@PlanningEntity(comparatorClass = CrewSeat.DifficultyComparator.class)
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

    /**
     * The one shared UNASSIGNED value (id 0). It is the only UNASSIGNED instance in any value range (non-manager
     * seats only); EmployeeFact has no equals(), so Timefold compares employees by identity. PlanBuilder uses it for
     * every seat it creates empty, and {@code PlanBuilder.normalizeInitialValues} replaces any other id-0 copy.
     */
    public static final EmployeeFact UNASSIGNED =
        new EmployeeFact(0, "__UNASSIGNED__", "__UNASSIGNED__", Map.of(), false, "");

    public CrewSeat() {}

    public void setCandidateEmployees(List<EmployeeFact> list) {
        this.candidateEmployees = (list == null) ? List.of() : list;
    }
    public List<EmployeeFact> getCandidateEmployees() { return candidateEmployees; }

    /**
     * Value range of {@link #employee}. Timefold 2.x validates every initial value against this range once, when a
     * working solution is set (pinned entities included, {@code null} skipped), and fails on a value outside it; the
     * range itself must be a plain List (custom ValueRange classes are rejected), so the range is STRICT and
     * {@code PlanBuilder.normalizeInitialValues} brings the initial values into it before solving.
     * <ul>
     *   <li>pinned seat: exactly its own (pinned) employee, so it validates; it never moves;</li>
     *   <li>manager seat: the manager candidates only, never UNASSIGNED;</li>
     *   <li>other seats: the candidates plus the shared {@link #UNASSIGNED}.</li>
     * </ul>
     */
    @ValueRangeProvider(id = "eligibleEmployeesForSeat")
    public List<EmployeeFact> eligibleEmployeesForSeat() {
        // Pinned: the range is the seat's own employee. PlanBuilder never sets pinnedWid; the pinned employee is the
        // seat's initial value, so this is what the 1.x range tolerated through ValidationTolerantRange.
        if (pinnedFixed && employee != null) {
            return List.of(employee);
        }
        return strictEligibleEmployees();
    }

    /** The value range without the pinned-seat rule: the same lists, in the same order, as on 1.27/1.35. */
    public List<EmployeeFact> strictEligibleEmployees() {
        // Pinned → pin hard
        if (pinnedFixed && pinnedWid != null) {
            for (EmployeeFact e : candidateEmployees) {
                if (e != null && pinnedWid.equals(e.wid)) {
                    return List.of(e);
                }
            }
            // For pinned but missing
            return List.of(UNASSIGNED);
        }

        List<EmployeeFact> base = (candidateEmployees == null) ? List.of() : candidateEmployees;

        if (needManager) {
            // STRICT: manager-only, no UNASSIGNED fallback
            return base.stream().filter(emp -> emp != null && emp.isManager).toList();
        } else {
            // Non-manager seat may keep UNASSIGNED as a legal value to ease feasibility
            List<EmployeeFact> list = new ArrayList<>(base);
            list.add(UNASSIGNED);
            return list.stream().distinct().toList();
        }
    }

    @PlanningVariable(
            valueRangeProviderRefs = "eligibleEmployeesForSeat",
            comparatorClass = EmployeeStrength.class
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
