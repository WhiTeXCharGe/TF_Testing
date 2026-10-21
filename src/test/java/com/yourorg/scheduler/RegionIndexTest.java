package com.yourorg.scheduler;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

import java.util.Arrays;
import java.util.List;

import org.junit.jupiter.api.Test;

import com.yourorg.scheduler.calendar.RegionIndex;

class RegionIndexTest {

    @Test
    void known_ids_get_dense_indexes_in_first_seen_order() {
        RegionIndex ri = RegionIndex.of(List.of("r_other", "r1", "r2"));
        assertEquals(3, ri.size());
        assertEquals(0, ri.indexOf("r_other"));
        assertEquals(1, ri.indexOf("r1"));
        assertEquals(2, ri.indexOf("r2"));
    }

    @Test
    void unknown_null_and_blank_are_minus_one() {
        RegionIndex ri = RegionIndex.of(List.of("r1", "r2"));
        assertEquals(-1, ri.indexOf("nope"));
        assertEquals(-1, ri.indexOf(null));
        assertEquals(-1, ri.indexOf(""));
        assertEquals(-1, ri.indexOf("   "));
    }

    @Test
    void idOf_round_trips_and_is_null_for_no_region() {
        RegionIndex ri = RegionIndex.of(List.of("r1", "r2", "r3"));
        for (String id : List.of("r1", "r2", "r3")) assertEquals(id, ri.idOf(ri.indexOf(id)));
        assertNull(ri.idOf(-1));
        assertNull(ri.idOf(3));
    }

    @Test
    void duplicates_blanks_and_nulls_in_the_input_are_ignored() {
        RegionIndex ri = RegionIndex.of(Arrays.asList("r1", null, "r1", " ", "r2"));
        assertEquals(2, ri.size());
        assertEquals(0, ri.indexOf("r1"));
        assertEquals(1, ri.indexOf("r2"));
    }

    @Test
    void empty_index() {
        RegionIndex ri = RegionIndex.of(List.of());
        assertEquals(0, ri.size());
        assertEquals(-1, ri.indexOf("r1"));
        assertNull(ri.idOf(0));
    }
}
