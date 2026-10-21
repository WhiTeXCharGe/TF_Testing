package com.yourorg.scheduler.trip;

import java.util.List;

/**
 * Result of {@link TripDeriver#derive}.
 *
 * @param derived  new FLEXIBLE trips, sorted by worker then start
 * @param cutFixed Fixed trips removed because no same-region worked day lies strictly inside (the input instances)
 * @param warnings {@code region_conflict_day}, {@code cut_fixed_trip} and {@code tight_transit} messages
 */
public record DeriveResult(List<Trip> derived, List<Trip> cutFixed, List<String> warnings) {
}
