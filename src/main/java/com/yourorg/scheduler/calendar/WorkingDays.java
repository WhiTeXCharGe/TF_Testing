package com.yourorg.scheduler.calendar;

import java.util.ArrayList;
import java.util.List;

import com.yourorg.scheduler.model.BlockDecision;

/** Working-day and worker-availability queries over the current {@link SolveContext} calendars. */
public final class WorkingDays {

    private WorkingDays() {}

    public static boolean isWorkingDay(int dayId, String fabId) {
        return SolveContext.current().calendars.isWorkingDay(dayId, fabId);
    }

    // IMPORTANT: Integer (nullable) parameters here
    public static int workingDaysCount(Integer startDay, Integer dayCount, String fabId) {
        if (startDay == null || dayCount == null || startDay < 0 || dayCount == 0) return 0;
        int end = startDay + dayCount - 1;
        int n = 0;
        for (int d = startDay; d <= end; d++) if (isWorkingDay(d, fabId)) n++;
        return n;
    }

    // Returns the list of working day IDs in [startDay, startDay+days-1] for fabId.
    // Used by flattenLast-based constraints to avoid the Joiners.filtering cartesian product.
    public static List<Integer> getWorkingDayList(int startDay, int days, String fabId) {
        List<Integer> result = new ArrayList<>(days);
        int end = startDay + days - 1;
        for (int d = startDay; d <= end; d++) {
            if (isWorkingDay(d, fabId)) result.add(d);
        }
        return result;
    }

    // Builds the value cached by the BlockDecision.coveredWorkingDays shadow
    // variable: every factory-working dayId inside the block's current span,
    // clamped to the solver horizon [SOLVE_MIN_DAY, SOLVE_MAX_DAY]. The horizon
    // clamp reproduces the old behaviour where constraints iterated DaySlot
    // facts (which only existed inside the horizon), so scores are unchanged.
    public static List<Integer> computeCoveredWorkingDays(BlockDecision b) {
        if (b == null || b.startDay == null || b.days == null || b.days <= 0) {
            return List.of();
        }
        int end = b.startDay + b.days - 1;
        List<Integer> out = new ArrayList<>(Math.max(0, b.days));
        for (int d = b.startDay; d <= end; d++) {
            if (d < SolveContext.current().solveMinDay || d > SolveContext.current().solveMaxDay) continue;
            if (isWorkingDay(d, b.factory)) out.add(d);
        }
        return out;
    }

    public static boolean isWorkerUnavailableByWid(String wid, int dayId) {
        return SolveContext.current().calendars.isWorkerUnavailable(wid, dayId);
    }
}
