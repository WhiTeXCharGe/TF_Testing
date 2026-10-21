package com.yourorg.scheduler;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import com.yourorg.scheduler.calendar.Calendars;
import com.yourorg.scheduler.calendar.SolveContext;

class SolveContextTest {

    static final Path SMALL = Path.of("src", "test", "resources", "fixtures", "small");
    static final Path TINY = Path.of("src", "test", "resources", "fixtures", "tiny");

    @TempDir
    Path tmp;

    @BeforeEach
    @AfterEach
    void resetContext() {
        SolveContext.clear();
    }

    @Test
    void clear_resets_state() throws Exception {
        Characterization.buildFrom(SMALL, tmp.resolve("small"));
        SolveContext c = SolveContext.current();
        assertNotNull(c.calendars);
        assertTrue(c.solveMaxDay > 0);
        assertFalse(c.fixedHoursByEmpDay.isEmpty());
        assertFalse(c.fixedTotalHoursByEmp.isEmpty());
        assertFalse(c.dayToYear.isEmpty());
        assertFalse(c.opAvgSkill.isEmpty());

        SolveContext.clear();

        SolveContext d = SolveContext.current();
        assertNull(d.calendars);
        assertEquals(0, d.solveMinDay);
        assertEquals(0, d.solveMaxDay);
        assertEquals(0.0, d.targetHoursPerEmp);
        assertTrue(d.affinityTagWeights.isEmpty());
        assertTrue(d.opCapacity.isEmpty());
        assertTrue(d.opAvgSkill.isEmpty());
        assertTrue(d.dayToYear.isEmpty());
        assertTrue(d.dayToYm.isEmpty());
        assertTrue(d.fixedHoursByEmpDay.isEmpty());
        assertTrue(d.fixedFactoriesByEmpDay.isEmpty());
        assertTrue(d.fixedTotalHoursByEmp.isEmpty());
        assertTrue(d.fixedAnnualOtByEmpYear.isEmpty());
        assertTrue(d.fixedMonthlyOtByEmpYm.isEmpty());
        assertTrue(d.fixedTaskcountByEmpDay.isEmpty());
    }

    @Test
    void building_another_fixture_does_not_leak_values() throws Exception {
        Characterization.buildFrom(TINY, tmp.resolve("b1"));
        Map<String, Object> first = snapshot(SolveContext.current());
        assertFalse(SolveContext.current().calendars.fabToRegion.isEmpty());

        Characterization.buildFrom(SMALL, tmp.resolve("a"));
        Map<String, Object> afterSmall = snapshot(SolveContext.current());
        assertFalse(first.equals(afterSmall), "fixtures must differ, otherwise the test proves nothing");

        Characterization.buildFrom(TINY, tmp.resolve("b2"));
        assertEquals(first, snapshot(SolveContext.current()));
    }

    /** Sizes and scalar values of everything a build fills. */
    static Map<String, Object> snapshot(SolveContext c) {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("fixedHoursByEmpDay", c.fixedHoursByEmpDay.toString());
        m.put("fixedFactoriesByEmpDay", c.fixedFactoriesByEmpDay.toString());
        m.put("fixedTotalHoursByEmp", c.fixedTotalHoursByEmp.toString());
        m.put("fixedAnnualOtByEmpYear", c.fixedAnnualOtByEmpYear.toString());
        m.put("fixedMonthlyOtByEmpYm", c.fixedMonthlyOtByEmpYm.toString());
        m.put("fixedTaskcountByEmpDay", c.fixedTaskcountByEmpDay.toString());
        m.put("dayToYear.size", c.dayToYear.size());
        m.put("dayToYm.size", c.dayToYm.size());
        m.put("opCapacity", c.opCapacity.toString());
        m.put("opAvgSkill", c.opAvgSkill.toString());
        m.put("affinityTagWeights", c.affinityTagWeights.toString());
        m.put("solveMinDay", c.solveMinDay);
        m.put("solveMaxDay", c.solveMaxDay);
        m.put("targetHoursPerEmp", c.targetHoursPerEmp);
        Calendars cal = c.calendars;
        m.put("calendars.fabs", cal.fabToRegion.size());
        m.put("calendars.regions", cal.regionOff.size());
        m.put("calendars.customers", cal.customerOff.size());
        m.put("calendars.workers", cal.workerOffByWid.size());
        m.put("calendars.transit", cal.transitDays.size());
        return m;
    }
}
