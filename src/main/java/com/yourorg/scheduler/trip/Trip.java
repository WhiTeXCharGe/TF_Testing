package com.yourorg.scheduler.trip;

import java.time.LocalDate;
import java.util.Map;

/**
 * One trip of one worker as read from {@code trip_list}: FI on {@code start}, FO on {@code end}, in {@code region}.
 *
 * @param extra every key of the input trip that is not worker / start_date / end_date / region / plan_flexibility /
 *              flexibility / description (never null)
 */
public record Trip(String worker, LocalDate start, LocalDate end, String region, Flexibility flexibility,
                   String description, Map<String, Object> extra) {
}
