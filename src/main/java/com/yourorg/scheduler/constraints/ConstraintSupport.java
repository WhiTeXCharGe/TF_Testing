package com.yourorg.scheduler.constraints;

import java.util.Collection;
import java.util.List;

import com.yourorg.scheduler.model.BlockDecision;
import com.yourorg.scheduler.model.CrewSeat;

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
    static Collection<Integer> seatCoveredDays(CrewSeat s, BlockDecision b) {
        if (s.pinnedFixed) {
            return (s.pinnedWorkDays == null) ? List.of() : s.pinnedWorkDays;
        }
        return (b == null || b.coveredWorkingDays == null) ? List.of() : b.coveredWorkingDays;
    }
}
