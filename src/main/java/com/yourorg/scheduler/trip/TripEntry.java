package com.yourorg.scheduler.trip;

import java.util.Map;

/**
 * A trip kept by the parser.
 *
 * @param trip     the parsed trip; null when the input trip is too broken to build one (missing worker, missing or
 *                 unparseable start_date / end_date, missing region)
 * @param raw      the trip map as read (a LinkedHashMap copy, original key order); used to copy Fixed trips to the
 *                 output unchanged. In the nested form it is the map inside {@code trips:} (no worker key). Values
 *                 that SnakeYAML turned into dates (java.util.Date, LocalDate) are converted to {@code yyyy/MM/dd}
 *                 strings; everything else is as read.
 * @param enforced true only for a valid FIXED trip that survived the overlap check
 * @param worker   the owner of the trip: the worker of the nested parent, else the trip's own {@code worker} key,
 *                 as a trimmed string (a numeric id 1 becomes "1"); null = no owner. Exporters must use this, never
 *                 {@code raw.get("worker")}: a nested raw has no worker key, a flat raw has it.
 */
public record TripEntry(Trip trip, Map<String, Object> raw, boolean enforced, String worker) {

    /** Owner taken from the trip (null when there is no trip). */
    public TripEntry(Trip trip, Map<String, Object> raw, boolean enforced) {
        this(trip, raw, enforced, trip == null ? null : trip.worker());
    }
}
