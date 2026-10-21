package com.yourorg.scheduler.trip;

import java.util.List;

/** Entries in input order, and one human-readable line per problem found. */
public record TripParseResult(List<TripEntry> entries, List<String> warnings) {
}
