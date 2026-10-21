package com.yourorg.scheduler.calendar;

/**
 * The single answer to "may this worker work on this day for a task of this region?" as far as Fixed trips are
 * concerned (design section 4, "One shared test"). Every day-based consumer of a dynamic seat goes through it:
 * {@code ConstraintSupport.seatCoveredDays}, {@code SeatConstraints.expandToEmpDays} / {@code expandToEmpDayHours},
 * {@code BlockConstraints.producedHoursForBlock} and the per-day loops of {@code ScheduleExporter}.
 *
 * <p>Personal {@code unavailable_dates} are NOT part of this test; they keep their own (older) handling. Pinned
 * (Fixed) seats never ask it: Fixed rows are facts.
 */
public final class WorkerCalendar {

    private WorkerCalendar() {}

    /**
     * False when {@code day} is the start or end day of an enforced Fixed trip of the worker, or lies strictly
     * inside one whose region is not {@code regionIdx} (a region-less task, {@code regionIdx == -1}, is closed
     * there too); true otherwise. Workers without trips take the fast path (one array read).
     *
     * @param empIdx    {@code EmployeeFact.id}
     * @param day       dayId (days since the plan start)
     * @param regionIdx region index of the task ({@code CrewSeat.regionIdx}), -1 = none
     */
    public static boolean isOpen(int empIdx, int day, int regionIdx) {
        TripCalendar c = SolveContext.current().tripCalendar;
        if (!c.hasMarks(empIdx)) return true;
        if (c.isBlocked(empIdx, day)) return false;
        int lock = c.lockRegion(empIdx, day);
        return lock == -1 || lock == regionIdx;
    }

    /** True when the worker has any trip mark (callers use it to skip per-day work for everybody else). */
    public static boolean hasMarks(int empIdx) {
        return SolveContext.current().tripCalendar.hasMarks(empIdx);
    }
}
