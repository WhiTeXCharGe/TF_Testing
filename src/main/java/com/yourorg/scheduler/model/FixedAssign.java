package com.yourorg.scheduler.model;

import java.util.HashMap;
import java.util.Map;

public class FixedAssign {
    public String module; public String opId; public String factory; public String wid;
    public int startDayId; public int endDayId;
    public Map<Integer,Integer> hoursByDay = new HashMap<>();
    public String phaseId; public int phaseNum;
    /**
     * Region of a misc row: the misc task's own {@code region}, or null when it has none / blank / is listed in
     * {@code misc_neutral_regions}. Always null for module rows (their region comes from the fab).
     */
    public String region;
}
