package com.yourorg.scheduler.constraints;

import java.util.*;

import ai.timefold.solver.core.api.score.buildin.hardmediumsoft.HardMediumSoftScore;
import ai.timefold.solver.core.api.score.stream.Constraint;
import ai.timefold.solver.core.api.score.stream.ConstraintCollectors;
import ai.timefold.solver.core.api.score.stream.ConstraintFactory;
import ai.timefold.solver.core.api.score.stream.Joiners;

import com.yourorg.scheduler.calendar.SolveContext;
import com.yourorg.scheduler.calendar.WorkingDays;
import com.yourorg.scheduler.model.*;
import static com.yourorg.scheduler.config.SolverConstants.DAILY_CAP;

/** Seat-level hard rules: one task per employee per day, daily hour cap (2 registered constraints). */
final class SeatConstraints {

    // ---------- Seat hard rules ----------

    Constraint employeeAvailableAllDays(ConstraintFactory f) {
        return f.forEach(CrewSeat.class)
            .join(f.forEach(BlockDecision.class),
                Joiners.equal((CrewSeat s) -> s.blockId, (BlockDecision b) -> b.id))
            .filter((s, b) -> !WorkerRules.isUnassigned(s.employee))
            .filter((s, b) -> {
                Set<Integer> off = SolveContext.current().calendars.workerOffByWid.getOrDefault(s.employee.wid, Set.of());

                // FIXED seat: check only exact worked days
                if (s.pinnedFixed) {
                    if (s.pinnedWorkDays == null) return false;
                    for (int did : s.pinnedWorkDays) {
                        if (off.contains(did)) return true;
                    }
                    return false;
                }

                // dynamic seat: check range
                final Integer start = b.startDay;
                final Integer days  = b.days;
                if (start == null || days == null || days <= 0) return false;

                for (int di = 0; di < days; di++) {
                    int did = start + di;
                    if (!WorkingDays.isWorkingDay(did, s.factory)) continue;
                    if (off.contains(did)) return true;
                }
                return false;
            })
            .penalize(HardMediumSoftScore.ONE_HARD)
            .asConstraint("seat-worker-available-all-days");
    }

    // helper key (put inside EmployeeSchedule class, anywhere static)
    static final class EmpDayKey {
        final int empId;
        final int dayId;
        EmpDayKey(int empId, int dayId) { this.empId = empId; this.dayId = dayId; }

        @Override public boolean equals(Object o) {
            if (this == o) return true;
            if (!(o instanceof EmpDayKey k)) return false;
            return empId == k.empId && dayId == k.dayId;
        }
        @Override public int hashCode() { return 31 * empId + dayId; }
    }

    // Carries (empId, dayId) for the task-count constraint.
    // Extends EmpDayKey semantics; EmpDayKey already has equals/hashCode.
    static final class EmpDayHours {
        final int empId; final int dayId; final int hours;
        EmpDayHours(int e, int d, int h) { empId=e; dayId=d; hours=h; }
    }

    // Expands one (seat, block) into one EmpDayKey per working day in the block range.
    // Used with flattenLast so Timefold scores only working days, not all 127 DaySlots.
    private static List<EmpDayKey> expandToEmpDays(int empId, int start, int days, String fab) {
        List<EmpDayKey> out = new ArrayList<>(days);
        for (int d = start, end = start + days - 1; d <= end; d++) {
            if (WorkingDays.isWorkingDay(d, fab)) out.add(new EmpDayKey(empId, d));
        }
        return out;
    }

    // Same as expandToEmpDays but also carries hours per day (for the cap constraint).
    private static List<EmpDayHours> expandToEmpDayHours(int empId, int start, int days, String fab, int hrs) {
        List<EmpDayHours> out = new ArrayList<>(days);
        for (int d = start, end = start + days - 1; d <= end; d++) {
            if (WorkingDays.isWorkingDay(d, fab)) out.add(new EmpDayHours(empId, d, hrs));
        }
        return out;
    }
    Constraint oneTaskPerEmpPerDay(ConstraintFactory f) {
        // Strategy: map (seat, block) → List<EmpDayKey> (one entry per working day),
        // then flattenLast(list → list) → UniConstraintStream<EmpDayKey>.
        // This replaces the Joiners.filtering DaySlot×Seat cartesian product that was
        // O(127 DaySlots) per incremental move; now it's O(working_days_in_block).
        return f.forEach(CrewSeat.class)
            .filter(s -> !WorkerRules.isUnassigned(s.employee) && !s.isPinned())
            .join(f.forEach(BlockDecision.class),
                Joiners.equal((CrewSeat s) -> s.blockId, (BlockDecision b) -> b.id))
            .filter((CrewSeat s, BlockDecision b) -> b.startDay != null && b.days != null && b.days > 0)
            .map((CrewSeat s, BlockDecision b) ->
                    expandToEmpDays(s.employee.id, b.startDay, b.days, s.factory))
            .flattenLast(list -> list)
            // UniConstraintStream<EmpDayKey>
            .groupBy(key -> key, ConstraintCollectors.count())
            .filter((key, dynCount) -> {
                int fixedCount = SolveContext.current().fixedTaskcountByEmpDay
                    .getOrDefault(key.empId, Collections.emptyMap())
                    .getOrDefault(key.dayId, 0);
                return (dynCount + fixedCount) > 1;
            })
            .penalize(HardMediumSoftScore.ONE_HARD, (key, dynCount) -> {
                int fixedCount = SolveContext.current().fixedTaskcountByEmpDay
                    .getOrDefault(key.empId, Collections.emptyMap())
                    .getOrDefault(key.dayId, 0);
                return (dynCount + fixedCount) - 1;
            })
            .asConstraint("emp-one-task-per-day");
    }

    Constraint dailyCap12h(ConstraintFactory f) {
        // Same flattenLast pattern. Pinned seats excluded — their hours are in FIXED_HOURS_BY_EMP_DAY.
        return f.forEach(CrewSeat.class)
            .filter(s -> !WorkerRules.isUnassigned(s.employee) && !s.isPinned())
            .join(f.forEach(BlockDecision.class),
                Joiners.equal((CrewSeat s) -> s.blockId, (BlockDecision b) -> b.id))
            .filter((CrewSeat s, BlockDecision b) -> b.startDay != null && b.days != null && b.days > 0)
            .map((CrewSeat s, BlockDecision b) ->
                    expandToEmpDayHours(s.employee.id, b.startDay, b.days, s.factory, b.chosenHours()))
            .flattenLast(list -> list)
            // UniConstraintStream<EmpDayHours>
            .groupBy(
                edh -> Arrays.asList(edh.empId, edh.dayId),
                ConstraintCollectors.sum(edh -> edh.hours))
            .filter((key, dynamicHours) -> {
                int empId = (int) key.get(0);
                int dayId = (int) key.get(1);
                int fixedHours = SolveContext.current().fixedHoursByEmpDay
                        .getOrDefault(empId, Collections.emptyMap())
                        .getOrDefault(dayId, 0);
                return dynamicHours + fixedHours > DAILY_CAP;
            })
            .penalize(HardMediumSoftScore.ONE_HARD, (key, dynamicHours) -> {
                int empId = (int) key.get(0);
                int dayId = (int) key.get(1);
                int fixedHours = SolveContext.current().fixedHoursByEmpDay
                        .getOrDefault(empId, Collections.emptyMap())
                        .getOrDefault(dayId, 0);
                int total = dynamicHours + fixedHours;
                return total - DAILY_CAP;
            })
            .asConstraint("seat-daily-cap-12h");
    }

    Constraint[] all(ConstraintFactory f) {
        return new Constraint[] {
            // seat-level hard rules
            // employeeAvailableAllDays(f),
            // oneFactoryPerEmpPerDay(f),
            oneTaskPerEmpPerDay(f),
            dailyCap12h(f),
        };
    }
}
