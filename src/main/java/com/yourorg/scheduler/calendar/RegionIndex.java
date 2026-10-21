package com.yourorg.scheduler.calendar;

import java.util.Collection;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.Map;

/**
 * Dense {@code int} index over region ids, so that hot-path code (constraints) can carry a region as a plain int on a
 * {@code CrewSeat} instead of looking a string up per call. Immutable. Index {@code -1} means "no region" (null,
 * blank or unknown id).
 */
public final class RegionIndex {

    private final String[] ids;
    private final Map<String, Integer> byId;

    private RegionIndex(String[] ids, Map<String, Integer> byId) {
        this.ids = ids;
        this.byId = byId;
    }

    /** Index over {@code regionIds} in first-seen order; null, blank and duplicate entries are skipped. */
    public static RegionIndex of(Collection<String> regionIds) {
        LinkedHashSet<String> distinct = new LinkedHashSet<>();
        if (regionIds != null) {
            for (String id : regionIds) {
                if (id != null && !id.isBlank()) distinct.add(id);
            }
        }
        String[] ids = distinct.toArray(new String[0]);
        Map<String, Integer> byId = new HashMap<>(Math.max(16, ids.length * 2));
        for (int i = 0; i < ids.length; i++) byId.put(ids[i], i);
        return new RegionIndex(ids, byId);
    }

    /** Index of {@code id}, or -1 when it is null, blank or unknown. */
    public int indexOf(String id) {
        if (id == null) return -1;
        Integer i = byId.get(id);
        return i == null ? -1 : i;
    }

    /** Region id of {@code idx}, or null for -1 / out of range. */
    public String idOf(int idx) {
        return (idx < 0 || idx >= ids.length) ? null : ids[idx];
    }

    public int size() {
        return ids.length;
    }
}
