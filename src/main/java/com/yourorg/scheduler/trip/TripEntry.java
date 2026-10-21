package com.yourorg.scheduler.trip;

import java.util.Map;

/**
 * A trip kept by the parser.
 *
 * @param trip     the parsed trip; null when the input trip is too broken to build one (missing worker, missing or
 *                 unparseable start_date / end_date, missing region)
 * @param raw      the trip map exactly as read (a LinkedHashMap copy, original key order); used to copy Fixed trips to
 *                 the output unchanged. In the nested form it is the map inside {@code trips:} (no worker key).
 * @param enforced true only for a valid FIXED trip that survived the overlap check
 */
public record TripEntry(Trip trip, Map<String, Object> raw, boolean enforced) {
}
