package com.yourorg.scheduler.constraints;

import java.util.*;

import ai.timefold.solver.core.api.score.HardMediumSoftScore;
import ai.timefold.solver.core.api.score.stream.Constraint;
import ai.timefold.solver.core.api.score.stream.ConstraintCollectors;
import ai.timefold.solver.core.api.score.stream.ConstraintFactory;
import ai.timefold.solver.core.api.score.stream.Joiners;

import com.yourorg.scheduler.calendar.SolveContext;
import com.yourorg.scheduler.calendar.WorkerCalendar;
import com.yourorg.scheduler.calendar.WorkingDays;
import com.yourorg.scheduler.model.*;
import static com.yourorg.scheduler.constraints.ConstraintSupport.clampToInt;

/** Block-level feasibility and production rules (5 constraints). */
final class BlockConstraints {

    // ---------- Block-level guards ----------
    
    Constraint endWithinWindow(ConstraintFactory f) {
        return f.forEach(BlockDecision.class)
            .filter(b -> b.startDay != null && b.days != null
                    && (b.startDay + b.days - 1) > b.windowEnd)
            .penalize(HardMediumSoftScore.ONE_HARD,
                b -> ((b.startDay + b.days - 1) - b.windowEnd) * TIME_VIOLATION_MULT)
            .asConstraint("block-end-within-window");
    }

    Constraint hoursValueAllowed(ConstraintFactory f) {
        return f.forEach(BlockDecision.class)
            .filter(b -> b.allowed == null || b.allowed.isEmpty() || !b.allowed.contains(b.chosenHours()))
            .penalize(HardMediumSoftScore.ONE_HARD)
            .asConstraint("block-hours-in-allowed");
    }

    Constraint phaseOrder(ConstraintFactory f) {
        return f.forEach(BlockDecision.class)
            .filter(a -> a.phaseNum > 0)
            .join(f.forEach(BlockDecision.class).filter(b -> b.phaseNum > 0),
                Joiners.equal((BlockDecision a) -> a.module, (BlockDecision b) -> b.module),
                Joiners.equal((BlockDecision a) -> a.phaseNum + 1, (BlockDecision b) -> b.phaseNum))
            .filter((a,b) -> a.startDay != null && a.days != null && b.startDay != null
                    && (a.startDay + a.days - 1) >= b.startDay)
            .penalize(HardMediumSoftScore.ONE_HARD, (a,b) -> ((a.startDay + a.days - 1) - b.startDay + 1) * TIME_VIOLATION_MULT)
            .asConstraint("phase-order");
    }

    private static boolean isWorkerUnavailable(EmployeeFact e, int dayId) {
        if (e == null || e.wid == null) return false;
        return SolveContext.current().calendars.workerOffByWid.getOrDefault(e.wid, Set.of()).contains(dayId);
    }

    private static int producedHoursForBlock(BlockDecision b, List<CrewSeat> seats) {
        if (b == null || seats == null || seats.isEmpty()) return 0;

        int total = b.fixedAssignedHoursActual;
        int hours = b.chosenHours();
        
        for (CrewSeat s : seats) {
            if (s == null || WorkerRules.isUnassigned(s.employee)) continue;
            if (s.pinnedFixed) continue;

            if (b.startDay == null || b.days == null || b.days <= 0) continue;

            boolean marks = WorkerCalendar.hasMarks(s.employee.id);
            for (int i = 0; i < b.days; i++) {
                int did = b.startDay + i;
                if (!WorkingDays.isWorkingDay(did, s.factory)) continue;
                if (isWorkerUnavailable(s.employee, did)) continue; // zero on personal off-day
                if (marks && !WorkerCalendar.isOpen(s.employee.id, did, s.regionIdx)) continue; // zero on a trip-closed day
                total += hours;
            }
        }
        return total;
    }
    // ---------- Production from staffed seats ----------
    /**
     * Hard penalty per missing hour of production. Kept larger than the worker rules (1 per clash day / hour over the cap)
     * so that, when a plan cannot be made feasible, the solver still staffs every block and breaks worker rules
     * instead of leaving blocks empty.
     */
    static final int UNDERFILL_MULT = 10;
    /**
     * Hard penalty per day of a phase-order or window violation: 10 times the underfill weight, about one working day
     * of hours. A late block can only be fixed by shortening it AND adding a worker. Shortening alone leaves hours
     * missing, which costs hours x UNDERFILL_MULT; with a day weighing only 1 the solver never took that step and
     * stayed on the violation for good (5-module sample: stuck at -98 hard, 0 hard in 13 to 30 s with this ratio).
     * See Timefold_WeightFinding5Modules20261023.md.
     */
    static final int TIME_VIOLATION_MULT = 100;
    Constraint noUnderfillByBlock(ConstraintFactory f) {
        var perBlock = f.forEach(BlockDecision.class)
            .join(f.forEachIncludingUnassigned(CrewSeat.class),
                Joiners.equal((BlockDecision b) -> b.id, (CrewSeat s) -> s.blockId))
            .groupBy((b, s) -> b,
                    ConstraintCollectors.toList((b, s) -> s));

        return perBlock
            .filter((b, seats) -> {
                int prod = producedHoursForBlock(b, seats);
                return prod < b.requiredHours;
            })
            .penalize(HardMediumSoftScore.ONE_HARD,
                (b, seats) -> {
                    int prod = producedHoursForBlock(b, seats);
                    int gap = b.requiredHours - prod;
                    return clampToInt((long) gap * (long) UNDERFILL_MULT);
                })
            .asConstraint("block-no-underfill");
    }

    Constraint overfillAtMostOneDayByBlock(ConstraintFactory f) {
        var perBlock = f.forEach(BlockDecision.class)
            .join(f.forEachIncludingUnassigned(CrewSeat.class),
                Joiners.equal((BlockDecision b) -> b.id, (CrewSeat s) -> s.blockId))
            .groupBy((b, s) -> b,
                    ConstraintCollectors.toList((b, s) -> s));

        return perBlock
            .filter((b, seats) -> {
                int hours = b.chosenHours();
                int staffed = (int) seats.stream()
                        .filter(s -> s != null && !WorkerRules.isUnassigned(s.employee) && !s.pinnedFixed)
                        .count();
                int prod = producedHoursForBlock(b, seats);
                int over = prod - b.requiredHours;
                return over > staffed * hours;
            })
            .penalize(HardMediumSoftScore.ONE_HARD,
                (b, seats) -> {
                    int hours = b.chosenHours();
                    int staffed = (int) seats.stream()
                            .filter(s -> s != null && !WorkerRules.isUnassigned(s.employee) && !s.pinnedFixed)
                            .count();
                    int prod = producedHoursForBlock(b, seats);
                    int over = prod - b.requiredHours;
                    return Math.max(0, over - staffed * hours);
                })
            .asConstraint("block-overfill-at-most-one-day");
    }

    Constraint[] all(ConstraintFactory f) {
        return new Constraint[] {
            // block feasibility
            endWithinWindow(f),
            hoursValueAllowed(f),
            phaseOrder(f),

            // production & capacity (count only staffed seats)
            noUnderfillByBlock(f),
            overfillAtMostOneDayByBlock(f),
        };
    }
}
