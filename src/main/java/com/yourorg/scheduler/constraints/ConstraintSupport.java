package com.yourorg.scheduler.constraints;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import com.yourorg.scheduler.calendar.WorkerCalendar;
import com.yourorg.scheduler.model.BlockDecision;
import com.yourorg.scheduler.model.CrewSeat;
import com.yourorg.scheduler.model.EmployeeFact;

/** Helpers shared by more than one constraint group. */
final class ConstraintSupport {
    private ConstraintSupport() {}

    static int clampToInt(long v) {
        if (v > Integer.MAX_VALUE) return Integer.MAX_VALUE;
        if (v < Integer.MIN_VALUE) return Integer.MIN_VALUE;
        return (int) v;
    }

    // (Removed seatCoversDayAndWorking: the day-based constraints no longer
    // join DaySlot. Per-seat coverage now comes from seatCoveredDays, which
    // reads the BlockDecision.coveredWorkingDays shadow variable.)

    // Returns the dayIds a staffed seat actually works. Dynamic seats reuse
    // the block's coveredWorkingDays shadow variable; pinned (fixed-history)
    // seats use their explicit pinnedWorkDays set. Both collections already
    // exist, so this allocates nothing per call.
    //
    // Trip marks: for a dynamic seat whose worker has Fixed trips, the days
    // WorkerCalendar.isOpen closes (trip start/end, other region inside a
    // trip) are dropped, so every rule built on this list ignores them. Only
    // that path allocates; workers without trips get the shared list as
    // before. Pinned seats are facts and are never filtered.
    static Collection<Integer> seatCoveredDays(CrewSeat s, BlockDecision b) {
        if (s.pinnedFixed) {
            return (s.pinnedWorkDays == null) ? List.of() : s.pinnedWorkDays;
        }
        List<Integer> days = (b == null || b.coveredWorkingDays == null) ? List.of() : b.coveredWorkingDays;
        EmployeeFact e = s.employee;
        if (e == null || days.isEmpty() || !WorkerCalendar.hasMarks(e.id)) return days;
        List<Integer> open = new ArrayList<>(days.size());
        for (Integer d : days) {
            if (WorkerCalendar.isOpen(e.id, d, s.regionIdx)) open.add(d);
        }
        return open;
    }
}
