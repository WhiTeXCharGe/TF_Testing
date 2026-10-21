package com.yourorg.scheduler.model;

import ai.timefold.solver.core.api.domain.entity.PlanningEntity;
import ai.timefold.solver.core.api.domain.entity.PlanningPin;
import ai.timefold.solver.core.api.domain.lookup.PlanningId;
import ai.timefold.solver.core.api.domain.valuerange.CountableValueRange;
import ai.timefold.solver.core.api.domain.valuerange.ValueRangeFactory;
import ai.timefold.solver.core.api.domain.valuerange.ValueRangeProvider;
import ai.timefold.solver.core.api.domain.variable.PlanningVariable;
import ai.timefold.solver.core.api.domain.variable.ShadowVariable;
import ai.timefold.solver.core.impl.domain.valuerange.buildin.collection.ListValueRange;
import java.util.Comparator;
import java.util.List;

@PlanningEntity(difficultyComparatorClass = BlockDecision.DifficultyComparator.class)
public class BlockDecision {
    @PlanningId public int id;

    // facts...
    public String module;
    public String workflowId;
    public String factory;
    public String phaseId;
    public int phaseNum;
    public String opId;

    public int windowStart;
    public int windowEnd;
    public int requiredHours;
    public List<Integer> allowed;
    public int minHeads;
    public int maxHeads;
    public boolean warmPinned = false;
    public int recommendMinHeads;
    public int recommendMaxHeads;

    public int fixedAssignedHoursActual = 0;
    @PlanningVariable(valueRangeProviderRefs = "vrStartWithinWindow",
              strengthComparatorClass = StartDayStrength.class)
    public Integer startDay;

    @PlanningVariable(valueRangeProviderRefs = "vrDaysWithinWindow")
    public Integer days;

    public static final class StartDayStrength implements Comparator<Integer> {
        @Override public int compare(Integer a, Integer b) {
            // earlier dates first; nulls last so solver can set startDay early
            if (a == null) return (b == null) ? 0 : 1;
            if (b == null) return -1;
            return Integer.compare(a, b);
        }
    }

    public static final class DifficultyComparator implements Comparator<BlockDecision> {
        @Override
        public int compare(BlockDecision a, BlockDecision b) {
            if (a == b) return 0;
            if (a == null) return -1;
            if (b == null) return 1;

            int aWindow = Math.max(1, a.windowEnd - a.windowStart + 1);
            int bWindow = Math.max(1, b.windowEnd - b.windowStart + 1);

            // Ascending = easy -> hard.
            if (a.requiredHours != b.requiredHours) {
                return Integer.compare(a.requiredHours, b.requiredHours);
            }
            if (aWindow != bWindow) {
                return Integer.compare(bWindow, aWindow); // larger window easier
            }
            if (a.maxHeads != b.maxHeads) {
                return Integer.compare(a.maxHeads, b.maxHeads);
            }
            return Integer.compare(a.id, b.id);
        }
    }

    public static final class HoursStrength implements Comparator<Integer> {
        @Override public int compare(Integer a, Integer b) {
            // prefer smaller hours (8 < 10 < 12)
            if (a == null) return (b == null) ? 0 : 1;
            if (b == null) return -1;
            return Integer.compare(a, b);
        }
    }

    // --- PATCH: ValueRangeFactory instead of hand-built lists ---
    @PlanningVariable(valueRangeProviderRefs = "vrAllowedHours",
                    strengthComparatorClass = HoursStrength.class)
    public Integer hours; // null at start → falls back to first allowed

    @ValueRangeProvider(id = "vrStartWithinWindow")
    public CountableValueRange<Integer> vrStartWithinWindow() {
        // Timefold 1.35: an IntValueRange cannot be sorted by the variable's strengthComparatorClass
        // (UnsupportedOperationException), a ListValueRange can. Same values, same ascending order as before.
        int to = (phaseNum == 1) ? windowStart + 1 : windowEnd + 1;
        List<Integer> l = new java.util.ArrayList<>(Math.max(0, to - windowStart));
        for (int i = windowStart; i < to; i++) l.add(i);
        // 1.35 validates the initial value against the range; a warm start (applyInitialPlan) may put a phase-1 block
        // after windowStart. Offer the CURRENT value when it is missing (1.27 tolerated it).
        if (startDay != null && !l.contains(startDay)) l.add(startDay);
        return new ListValueRange<>(l);
    }
    @ValueRangeProvider(id = "vrDaysWithinWindow")
    public CountableValueRange<Integer> vrDaysWithinWindow() {
        int maxLen = Math.max(1, windowEnd - windowStart + 1);
        return ValueRangeFactory.createIntValueRange(1, maxLen + 1);
    }

    //  discrete list value range for hours (8,10,12, …)
    @ValueRangeProvider(id = "vrAllowedHours")
    public CountableValueRange<Integer> vrAllowedHours() {
        List<Integer> a = (allowed == null || allowed.isEmpty())
                ? List.of(8)
                : allowed.stream().distinct().sorted().toList();
        return new ListValueRange<>(a);
    }

    // SHADOW VARIABLE.
    // coveredWorkingDays is the list of factory-working dayIds inside this
    // block's [startDay, startDay+days-1] span, clamped to the solver
    // horizon. It is a pure function of the planning variables startDay and
    // days (plus the fixed fact factory), so BlockSpanVariableListener
    // recomputes it once whenever either source variable changes. Day-based
    // constraints read this cached list instead of re-deriving the span,
    // which removes the DaySlot-vs-CrewSeat cartesian-product joins that
    // previously dominated incremental score calculation.
    // "removal" is suppressed because the variableListenerClass form of
    // @ShadowVariable is deprecated in Timefold 1.27. It still works on the
    // Community Edition (the version this project pins) and is kept for
    // that reason; the warning is only relevant on a future major upgrade.
    @SuppressWarnings({"deprecation", "removal"})
    @ShadowVariable(variableListenerClass = BlockSpanVariableListener.class,
                    sourceVariableName = "startDay")
    @ShadowVariable(variableListenerClass = BlockSpanVariableListener.class,
                    sourceVariableName = "days")
    public List<Integer> coveredWorkingDays = List.of();

    // helper used in constraints where hours is needed
    public int chosenHours() {
        if (hours != null) return hours;
        // default to smallest allowed to bias toward 8 even before it’s set
        return (allowed == null || allowed.isEmpty()) ? 8
                : allowed.stream().mapToInt(Integer::intValue).min().orElse(8);
    }

    @PlanningPin
    public boolean isPinned() {
        return false;
    }
}
