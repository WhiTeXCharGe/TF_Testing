package com.yourorg.scheduler.trip;

import java.time.LocalDate;
import java.util.Set;

/** What the trip list is validated against: known workers and regions, and the plan range. */
public record TripContext(Set<String> workerIds, Set<String> regionIds, LocalDate planStart, LocalDate planEnd) {
}
