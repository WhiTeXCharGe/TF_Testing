package com.yourorg.scheduler.trip;

import java.time.LocalDate;

/** One worked day of one worker that has a region (non-null, non-blank). Region-less days are never passed in. */
public record WorkedDay(String worker, LocalDate date, String region) {
}
